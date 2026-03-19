#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${REPO_ROOT}/cracks_OOD_exps.py"
[ -f "${PY}" ] || { echo "[ERROR] Not found: ${PY}"; exit 1; }

csv_join() { local IFS=,; echo "$*"; }

check_dirs() {
  local trains=("$@")
  local test_dir="${trains[-1]}"; unset 'trains[-1]'
  [ "${#trains[@]}" -eq 5 ] || { echo "[ERROR] Need 5 train folders, got ${#trains[@]}"; exit 1; }
  for d in "${trains[@]}"; do [ -d "$d" ] || { echo "[ERROR] Not found: $d"; exit 1; }; done
  [ -d "$test_dir" ] || { echo "[ERROR] Not found: $test_dir"; exit 1; }
}

TRAIN_DIRS=(
  /home/data8t/sherlock/dataset/PDE/Irregular/Aletheia/single_layer
  /home/data8t/sherlock/dataset/PDE/Irregular/Aletheia/typeI_double-layer
  /home/data8t/sherlock/dataset/PDE/Irregular/Aletheia/typeII_double-layer
  /home/data8t/sherlock/dataset/PDE/Irregular/Aletheia/typeIII_double-layer
  /home/data8t/sherlock/dataset/PDE/Irregular/Aletheia/typeI_multi-layer
)
TEST_DIR=/home/data8t/sherlock/dataset/PDE/Irregular/Aletheia/typeII_multi-layer

check_dirs "${TRAIN_DIRS[@]}" "${TEST_DIR}"
TRAINS_CSV="$(csv_join "${TRAIN_DIRS[@]}")"

PER_FOLDER_TRAIN=100
PER_FOLDER_TEST=100

IN_CH=13
OUT_CH=1
USE_SURF=1

GPU=3
EPOCHS=200
BATCH_SIZE=4
LR=0.0001
TEST_CRACK_TYPE=6
DOWNSAMPLE_COUNT=8000
SURF_DOWNSAMPLE_COUNT=8000
TRAINING_NUM=3

MODELS_UNS="Transolver,MLP,FNO3d,GeoFNO,FFNO,FCNO,LNO,DeepONet,LaMO"
MODELS_STR="Transolver,MLP,FNO3d,GeoFNO,FFNO,FCNO,LNO,DeepONet,LaMO"

run_phase() {
  local TAG="$1" DTYPE="$2" MODELS_LIST="$3"
  echo "[RUN] S2Q  dtype=${DTYPE}  in=${IN_CH}  out=${OUT_CH}  gpu=${GPU}"
  CMD=(python "${PY}"
       --mode S2Q
       --ood_train_dirs "${TRAINS_CSV}"
       --ood_test_dir "${TEST_DIR}"
       --per_folder_train "${PER_FOLDER_TRAIN}"
       --per_folder_test "${PER_FOLDER_TEST}"
       --in_channels "${IN_CH}"
       --out_channels "${OUT_CH}"
       --epochs "${EPOCHS}"
       --batch_size "${BATCH_SIZE}"
       --lr "${LR}"
       --data_type "${DTYPE}"
       --downsample_count "${DOWNSAMPLE_COUNT}"
       --surf_downsample_count "${SURF_DOWNSAMPLE_COUNT}"
       --models "${MODELS_LIST}"
       --training_num "${TRAINING_NUM}"
       --test_crack_type "${TEST_CRACK_TYPE}"
       --gpu 0)
  if [ "${USE_SURF}" -eq 1 ]; then CMD+=(--use_surf); fi
  ( cd "${REPO_ROOT}" && CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" )
}

un_phase() { run_phase "$@"; }

run_phase "uns" "unstructured_data" "${MODELS_UNS}"

run_phase "s"   "structured_data"   "${MODELS_STR}"

echo "[MAIN] S2Q (unstructured and structured) finished."
