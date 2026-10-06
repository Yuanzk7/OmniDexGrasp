#!/usr/bin/env bash
# Recommended on 16GB GPUs: bash scripts/run_reconstruction.sh sequential
set -eo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
stage="${1:-}"
if [ "$#" -gt 0 ]; then shift; fi
case "$stage" in
  hamer) stage_env=hamer; stage_module=recons.server.hamer ;;
  gsam) stage_env=gsam; stage_module=recons.server.gsam ;;
  client) stage_env=omnidexgrasp; stage_module=recons.client ;;
  sequential) stage_env=omnidexgrasp; stage_module=recons.sequential ;;
  pose) stage_env=megapose; stage_module=recons.pose_est ;;
  *) echo 'Usage: bash scripts/run_reconstruction.sh {sequential|hamer|gsam|client|pose} [Hydra overrides...]' >&2; exit 2 ;;
esac
source /home/unist/anaconda3/etc/profile.d/conda.sh
conda activate "$stage_env"
set -u
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$project_root/omnidexgrasp/thirdparty/hamer${PYTHONPATH:+:$PYTHONPATH}"
export MEGAPOSE_DATA_DIR="$project_root/omnidexgrasp/thirdparty/megapose6d/local_data"
mkdir -p "$project_root/omnidexgrasp/log"
cd "$project_root/omnidexgrasp"
if [ "$stage" = hamer ] && [ ! -f "$project_root/assets/mano/models/MANO_RIGHT.pkl" ]; then
  echo 'Missing licensed MANO model: assets/mano/models/MANO_RIGHT.pkl' >&2
  exit 1
fi
# Servers bind only to this machine. Caller-provided Hydra overrides take precedence.
case "$stage" in
  hamer|gsam) exec python -m "$stage_module" server.host=127.0.0.1 "$@" ;;
  pose) exec python -m "$stage_module" megapose6d.bsz_images=16 "$@" ;;
  *) exec python -m "$stage_module" "$@" ;;
esac
