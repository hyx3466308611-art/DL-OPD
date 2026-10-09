#!/usr/bin/env bash
# Use converseg without `conda activate` (base python broken on glibc 2.17 login nodes).
# Usage: source scripts/env_converseg.sh
CONDA_PREFIX="${HOME}/miniconda311/envs/converseg"
if [[ ! -x "${CONDA_PREFIX}/bin/python" ]]; then
  echo "Missing ${CONDA_PREFIX}/bin/python" >&2
  return 1 2>/dev/null || exit 1
fi
export CONDA_PREFIX
export PATH="${CONDA_PREFIX}/bin:${PATH}"
hash -r 2>/dev/null || true
echo "converseg: $(python --version) ($(which python))"
