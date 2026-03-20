#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${REPO_ROOT}/run_all_models_exps.py"
[ -f "${PY}" ] || { echo "[ERROR] Not found: ${PY}"; exit 1; }


check_dir() { [ -d "$1" ] || { echo "[ERROR] Not found: $1"; exit 1; }; }


DATA_DIR_UNS="/data root/"
DATA_DIR_STR="/data root/"

CRACK_TYPE=6

check_dir "${DATA_DIR_UNS}"
check_dir "${DATA_DIR_STR}"


IN_CH=13
OUT_CH=1
USE_SURF=1

GPU=3
EPOCHS=200
BATCH_SIZE=4
LR=0.0001
DOWNSAMPLE_COUNT=8000
SURF_DOWNSAMPLE_COUNT=8000
TRAINING_NUM=1
TEST_SPLIT=0.2
DATA_NUM=600

SFO_TRAIN_NUM=480
SFO_TEST_NUM=120


MODELS_UNS="Transolver,MLP,FNO3d,GeoFNO,FFNO,FCNO,LNO,DeepONet"
MODELS_STR="Transolver,MLP,FNO3d,GeoFNO,FFNO,FCNO,LNO,DeepONet"

run_phase() {
  local TAG="$1" DTYPE="$2" ROOT="$3" FREQ_INDEX="$4" MODELS_LIST="$5"

  echo "[RUN] S2Q | tag=${TAG} | dtype=${DTYPE} | SFO_freq=${FREQ_INDEX} | root=${ROOT}"
  CMD=(python "${PY}"
       --mode S2Q
       --data_root "${ROOT}"
       --data_type "${DTYPE}"
       --crack_type "${CRACK_TYPE}"
       --in_channels "${IN_CH}"
       --out_channels "${OUT_CH}"
       --epochs "${EPOCHS}"
       --batch_size "${BATCH_SIZE}"
       --lr "${LR}"
       --downsample_count "${DOWNSAMPLE_COUNT}"
       --surf_downsample_count "${SURF_DOWNSAMPLE_COUNT}"
       --models "${MODELS_LIST}"
       --training_num "${TRAINING_NUM}"
       --test_split "${TEST_SPLIT}"
       --data_num "${DATA_NUM}"
       --gpu 0
       --OOD "sfo"
       --sfo_freq "${FREQ_INDEX}"
       --sfo_train_num "${SFO_TRAIN_NUM}"
       --sfo_test_num "${SFO_TEST_NUM}")
  if [ "${USE_SURF}" -eq 1 ]; then CMD+=(--use_surf); fi

  ( cd "${REPO_ROOT}" && CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" )
}


run_phase "SFO_9kHZ_uns"    "unstructured_data"  "${DATA_DIR_UNS}"  3   "${MODELS_UNS}"

run_phase "SFO_9kHZ_s"      "structured_data"    "${DATA_DIR_STR}"  3   "${MODELS_STR}"

run_phase "SFO_36kHZ_uns"   "unstructured_data"  "${DATA_DIR_UNS}"  6   "${MODELS_UNS}"

run_phase "SFO_36kHZ_uns"   "structured_data"    "${DATA_DIR_STR}"  6   "${MODELS_STR}"

run_phase "SFO_81kHZ_uns"   "unstructured_data"  "${DATA_DIR_UNS}"  9   "${MODELS_UNS}"

run_phase "SFO_81kHZ_uns"   "structured_data"    "${DATA_DIR_STR}"  9   "${MODELS_STR}"

echo "[MAIN] All S2Q SFO OOD experiments finished."
