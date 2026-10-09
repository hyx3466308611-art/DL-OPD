import torch, torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,  # <- Qwen2.5-VL
)

from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from converseg_paths import resolve_plm_dir



class LanguageAdapter(nn.Module):
    """
    Uses Qwen/Qwen2.5-VL-7B-Instruct as a multimodal encoder and
    projects features to SAM2's decoder dims.
    Produces:
      sparse: [B, N_text_tokens, C]
      dense:  [B, C, H, W]   (text-conditioned bias map)
    """
    def __init__(
        self,
        model_name=None,
        transformer_dim=256,
        n_sparse_tokens=0,
        use_dense_bias=True,
        dtype=torch.bfloat16,
        device="cuda",
        # ---- LoRA knobs ----
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_bias="none",
        lora_target_modules="auto",
        gradient_checkpointing=False,
        # ---- NEW ----
        use_image_input=True,
        max_txt_len=256,     # cap token length to save memory
        qwen_image_size=256, # Qwen vision resize (lower on V100 if CUDA conv fails)
    ):
        super().__init__()

        self.max_txt_len = max_txt_len
        self.qwen_image_size = int(qwen_image_size)

        plm_dir = str(resolve_plm_dir(model_name))
        load_kw = dict(local_files_only=True)

        # --- tokenizer & (optional) processor ---
        self.tok = AutoTokenizer.from_pretrained(plm_dir, use_fast=True, **load_kw)
        self.tok.padding_side = "right"

        # Cache which token IDs are *not* plain text (special, image placeholders, etc.)
        self._non_text_token_ids = None
        self._init_non_text_token_ids()

        self.processor = (
            AutoProcessor.from_pretrained(plm_dir, **load_kw) if use_image_input else None
        )
        if self.processor is not None and hasattr(self.processor, "image_processor"):
            ip = self.processor.image_processor
            try:
                ip.do_resize = True
            except Exception:
                pass
            self._apply_qwen_image_size(ip)

        # --- backbone: Qwen2.5-VL conditional generation model ---
        self.backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            plm_dir, dtype=dtype, device_map=None, **load_kw
        ).to(device)

        # Start frozen; LoRA will re-enable a tiny set
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Wire up LoRA (optional)
        self.peft_enabled = False
        if use_lora:
            target_modules = self._infer_lora_targets(self.backbone) if lora_target_modules == "auto" else lora_target_modules
            if len(target_modules) == 0:
                raise RuntimeError("Could not find any LoRA target modules; set `lora_target_modules` explicitly.")
            self.lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias=lora_bias,
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.backbone = get_peft_model(self.backbone, self.lora_cfg)
            self.peft_enabled = True

            if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
                try:
                    if hasattr(self.backbone, "config") and hasattr(self.backbone.config, "use_cache"):
                        self.backbone.config.use_cache = False
                except Exception:
                    pass
                self.backbone.gradient_checkpointing_enable()
            if hasattr(self.backbone, "enable_input_require_grads"):
                self.backbone.enable_input_require_grads()

        # Hidden size on the text side (Qwen2.5-VL has text_config)
        cfg = getattr(self.backbone, "config", None)
        D_t = getattr(getattr(cfg, "text_config", None), "hidden_size", None)
        if D_t is None:
            raise RuntimeError("Could not infer text hidden_size from model config.")

        self.to_sparse = nn.Linear(D_t, transformer_dim)
        self.to_dense = nn.Sequential(
            nn.Linear(D_t, transformer_dim),
            nn.SiLU(),
            nn.Linear(transformer_dim, transformer_dim),
        ) if use_dense_bias else None

        nn.init.xavier_uniform_(self.to_sparse.weight); nn.init.zeros_(self.to_sparse.bias)
        if self.to_dense is not None:
            for m in self.to_dense:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        self.n_sparse_tokens = n_sparse_tokens
        self.use_dense_bias = use_dense_bias
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.txt_norm = nn.LayerNorm(D_t)
        self.temp = nn.Parameter(torch.tensor(1.0))
        self._grad_ckpt_enabled = bool(gradient_checkpointing and self.peft_enabled)

        # ensure module dtypes/devices match
        self.to(device=device, dtype=dtype)

    def _qwen_images_kwargs(self) -> dict:
        """Resize kwargs compatible with fast and slow Qwen2-VL image processors."""
        s = int(self.qwen_image_size)
        ip = getattr(self.processor, "image_processor", None)
        if ip is not None and "Fast" not in type(ip).__name__:
            return {"do_resize": True, "size": {"shortest_edge": s, "longest_edge": s}}
        return {"do_resize": True, "size": {"height": s, "width": s}}

    def _apply_qwen_image_size(self, image_processor) -> None:
        s = int(self.qwen_image_size)
        if "Fast" in type(image_processor).__name__:
            image_processor.size = {"height": s, "width": s}
        else:
            image_processor.size = {"shortest_edge": s, "longest_edge": s}


    # ---- token filters -------------------------------------------------------
    def _init_non_text_token_ids(self):
        """
        Build a list of token IDs that should NOT count as text positions.
        Includes:
          - all special tokens (BOS/EOS, role tokens, etc.)
          - added vocab entries that look like image/vision placeholders
        """
        ids = set(getattr(self.tok, "all_special_ids", []) or [])
        # Grab added vocab and heuristically include any image/vision markers
        try:
            added = getattr(self.tok, "get_added_vocab", lambda: {})()
            for tok, tid in added.items():
                tl = tok.lower()
                if any(s in tl for s in ("image", "vision", "<img", "picture", "video")):
                    ids.add(int(tid))
        except Exception:
            pass
        # store as a 1D LongTensor; move to device on use
        if len(ids) == 0:
            # keep a sentinel so equality checks never broadcast against empty
            ids = {-(10**9)}
        self._non_text_token_ids = torch.tensor(sorted(ids), dtype=torch.long)

    def _text_positions_mask(self, ids: torch.Tensor, attn: torch.Tensor, eos_pos: torch.Tensor) -> torch.Tensor:
        """
        Return [B, T] mask where True = positions that correspond to *plain text* tokens.
        We exclude:
          - padding (already excluded by attn)
          - EOS position
          - any token ID in _non_text_token_ids (special/image placeholders)
        """
        device = ids.device
        bad = self._non_text_token_ids.to(device)  # [K]
        is_bad = (ids.unsqueeze(-1) == bad.view(1, 1, -1)).any(dim=-1)         # [B, T]
        idxs = torch.arange(ids.shape[1], device=device).unsqueeze(0).expand_as(ids)
        return (attn.bool() & ~is_bad & (idxs != eos_pos.unsqueeze(1)))        # [B, T]


    # ---- LoRA utilities -----------------------------------------------------
    def _infer_lora_targets(self, model: nn.Module):
        """
        Heuristic for LLaMA/decoder stacks:
        prefer attention proj + MLP proj layers.
        Returns base names that PEFT will match in module paths.
        """
        common = ["q_proj", "k_proj", "v_proj", "o_proj",  # attn
                  "wq", "wk", "wv", "wo",                 # alt naming
                  "gate_proj", "up_proj", "down_proj"]    # MLP
        # Keep only those that actually occur
        present = set()
        for name, _ in model.named_modules():
            base = name.split(".")[-1].lower()
            for t in common:
                if base == t:
                    present.add(t)
        # If nothing matches (unusual naming), fall back to all Linear in attention/MLP blocks
        if not present:
            for name, mod in model.named_modules():
                if isinstance(mod, nn.Linear) and any(s in name.lower() for s in ["attn", "attention", "mlp", "ffn"]):
                    present.add(name.split(".")[-1])
        return sorted(list(present))

    # --- text-only ---
    def encode_text(self, texts: list[str]):
        toks = self.tok(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_txt_len
        )
        toks = {k: v.to(self.backbone.device) for k, v in toks.items()}
        out = self._encoder_forward(
            input_ids=toks["input_ids"],
            attention_mask=toks["attention_mask"],
        )
        seq = self._final_token_features(out)                    # [B, T, D_t]
        attn = toks["attention_mask"].bool()
        ids  = toks["input_ids"].long()
        return seq, attn, ids


    def _vl_encoder(self):
        """Qwen2-VL inner model (vision + language), without lm_head (saves large GPU memory)."""
        m = self.backbone
        if hasattr(m, "base_model"):
            m = m.base_model
        if hasattr(m, "model"):
            inner = m.model
            if hasattr(inner, "model") and hasattr(inner, "lm_head"):
                return inner.model
            return inner
        return m

    @staticmethod
    def _install_embedding_grad_hooks(module: nn.Module) -> None:
        """Qwen2-VL + checkpoint: embedding outputs must require grad for LoRA."""
        if getattr(module, "_converseg_embed_grad_hook", False):
            return

        def _hook(_mod, _inp, out):
            if torch.is_tensor(out):
                out.requires_grad_(True)
            elif isinstance(out, (list, tuple)):
                for t in out:
                    if torch.is_tensor(t):
                        t.requires_grad_(True)

        for attr in ("get_input_embeddings",):
            if hasattr(module, attr):
                try:
                    emb = module.get_input_embeddings()
                    if emb is not None:
                        emb.register_forward_hook(_hook)
                except Exception:
                    pass
        visual = getattr(module, "visual", None)
        if visual is not None and hasattr(visual, "register_forward_hook"):
            visual.register_forward_hook(_hook)
        module._converseg_embed_grad_hook = True

    def prepare_gradient_checkpointing_training(self) -> None:
        """
        After load_lora / resume, re-hook embeddings so checkpointed Qwen blocks
        receive gradients (avoids 'None of the inputs have requires_grad=True').
        """
        if not self.peft_enabled:
            return
        bb = self.backbone
        enc = self._vl_encoder()
        if self._grad_ckpt_enabled and hasattr(bb, "gradient_checkpointing_enable"):
            try:
                if hasattr(bb, "config") and hasattr(bb.config, "use_cache"):
                    bb.config.use_cache = False
            except Exception:
                pass
            bb.gradient_checkpointing_enable()
            if hasattr(enc, "gradient_checkpointing_enable"):
                try:
                    enc.gradient_checkpointing_enable()
                except Exception:
                    pass
        for mod in (bb, enc):
            if hasattr(mod, "enable_input_require_grads"):
                try:
                    mod.enable_input_require_grads()
                except Exception:
                    pass
            self._install_embedding_grad_hooks(mod)
        for _name, p in bb.named_parameters():
            if "lora_" in _name:
                p.requires_grad = True

    def _encoder_forward(self, **kwargs):
        enc = self._vl_encoder()
        if self.training and self._grad_ckpt_enabled:
            self.prepare_gradient_checkpointing_training()
        if not getattr(LanguageAdapter, "_logged_encoder_only", False):
            print(
                f"[LanguageAdapter] encoder-only forward (skips lm_head): {enc.__class__.__name__}",
                flush=True,
            )
            LanguageAdapter._logged_encoder_only = True
        return enc(
            **kwargs,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
        )

    def _final_token_features(self, out):
        # Prefer hidden_states[-1] (decoder-only models usually don't return last_hidden_state)
        hs = getattr(out, "hidden_states", None)
        if hs is not None and len(hs) > 0:
            return hs[-1]
        lh = getattr(out, "last_hidden_state", None)
        if lh is not None:
            return lh
        raise RuntimeError(
            "Model output has neither last_hidden_state nor hidden_states. "
            "Pass output_hidden_states=True to the forward call."
        )

    
    # --- batched V+L (your Point 1 version) ---
    def encode_text_image(self, texts: list[str], image_paths: list[str]):
        assert self.processor is not None and len(texts) == len(image_paths) and len(texts) > 0
        device = self.backbone.device
        proj_dtype = self.to_sparse.weight.dtype

        def truncate_text(txt: str) -> str:
            toks = self.tok(txt or "", return_tensors="pt", padding=False, truncation=True,
                            max_length=getattr(self, "max_txt_len", 256), add_special_tokens=False)
            return self.tok.decode(toks["input_ids"][0], skip_special_tokens=True)

        conversations = [[{
            "role": "user",
            "content": [{"type": "image", "url": p}, {"type": "text", "text": truncate_text(t)}],
        }] for t, p in zip(texts, image_paths)]

        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=False,      # keep image tokens intact
            images_kwargs=self._qwen_images_kwargs(),
            # pad_to_multiple_of=8,  # uncomment if your processor supports it
        )

        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device, non_blocking=True)

        out = self._encoder_forward(**inputs)

        seq = self._final_token_features(out).to(proj_dtype)     # [B, T, D_t]
        attn = inputs["attention_mask"].to(torch.bool)           # [B, T]
        ids  = inputs["input_ids"].to(torch.long)                # [B, T]
        return seq, attn, ids



    def forward(self, texts: list[str], H: int, W: int, image_paths: list[str] | None = None):
        # start = time.time()
        # Route to V+L or text-only encoder
        if image_paths is not None and self.processor is not None:
            seq, attn, ids = self.encode_text_image(texts, image_paths)   # [B, T, D_t]
        else:
            seq, attn, ids = self.encode_text(texts)                      # [B, T, D_t]

        B, T, D_t = seq.shape
        device = seq.device

        # print("Shape of seq:", seq.shape)

        # match projection dtype
        proj_dtype = self.to_sparse.weight.dtype
        seq = seq.to(proj_dtype)

        # Normalize token embeddings
        seq = self.txt_norm(seq)

        # ---- find EOS per sequence ----
        eos_id = self.tok.eos_token_id
        if eos_id is None:
            eos_mask = torch.zeros_like(ids, dtype=torch.bool, device=device)
        else:
            eos_mask = (ids == eos_id).to(device)

        idxs = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        valid_counts = attn.long().sum(dim=1)
        fallback = (valid_counts - 1).clamp(min=0)
        eos_pos = torch.where(eos_mask, idxs, torch.full_like(idxs, -1)).amax(dim=1)
        eos_pos = torch.where(eos_pos >= 0, eos_pos, fallback)  # [B]

        # Dense = EOS vector
        eos_vec = seq[torch.arange(B, device=device), eos_pos]   # [B, D_t]

        # ---- sparse = TEXT token positions only (exclude image + all special + EOS) ----
        non_eos_mask = self._text_positions_mask(ids, attn, eos_pos)      # [B, T]
        if self.n_sparse_tokens > 0:
            N = self.n_sparse_tokens
        else:
            N = int(non_eos_mask.sum(dim=1).max().item())
            if N == 0:
                N = 1

        idx_mat = torch.full((B, N), -1, device=device, dtype=torch.long)
        for b in range(B):
            pos = torch.nonzero(non_eos_mask[b], as_tuple=False).squeeze(-1)
            take = pos[:N]
            idx_mat[b, :take.numel()] = take

        safe_idx = idx_mat.clamp(min=0)
        sparse_tok = seq[torch.arange(B, device=device).unsqueeze(-1), safe_idx]  # [B, N, D_t]
        valid_mask = (idx_mat >= 0).unsqueeze(-1).to(sparse_tok.dtype)
        sparse_tok = sparse_tok * valid_mask  # zero-pad

        # Project
        sparse = self.to_sparse(sparse_tok) * self.scale                           # [B, N, C]

        # Dense projection from EOS only
        if self.use_dense_bias:
            bias = self.to_dense(eos_vec) * self.temp.clamp(min=0.01)              # [B, C]
            dense = bias.unsqueeze(-1).unsqueeze(-1).expand(B, bias.shape[-1], H, W)
        else:
            C = self.to_sparse.out_features
            dense = torch.zeros(B, C, H, W, device=device, dtype=proj_dtype)

        return sparse, dense



    # -------- Save / Load LoRA adapters only --------
    def save_lora(self, out_dir: str):
        """
        Saves only the LoRA adapters (and PEFT config). Use with PeftModel.
        """
        if not self.peft_enabled:
            raise RuntimeError("LoRA is not enabled.")
        self.backbone.save_pretrained(out_dir)

    def load_lora(self, adapter_dir: str):
        """
        Loads adapters onto the *current* backbone weights.
        """
        if PeftModel is None:
            raise ImportError("peft is not installed. `pip install peft`")
        # If already a PeftModel, just load the new adapter weights.
        if isinstance(self.backbone, PeftModel):
            self.backbone.load_adapter(adapter_dir, adapter_name="default", is_trainable=True)
            self.backbone.set_adapter("default")
        else:
            self.backbone = PeftModel.from_pretrained(
                self.backbone, adapter_dir, is_trainable=True, local_files_only=True
            )
        self.peft_enabled = True
        self.prepare_gradient_checkpointing_training()
