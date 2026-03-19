#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${REPO_ROOT}/run_all_models_exps.py"
[ -f "${PY}" ] || { echo "[ERROR] Not found: ${PY}"; exit 1; }


check_dir() { [ -d "$1" ] || { echo "[ERROR] Not found: $1"; exit 1; }; }

#1
#DATA_DIR_UNS="/home/data8t/sherlock/dataset/PDE/Irregular/Aletheia/single_layer"
#DATA_DIR_STR="/home/data8t/sherlock/dataset/PDE/Irregular/Aletheia/single_layer"
#CRACK_TYPE=1
#2
#DATA_DIR_UNS="/home/data8t/sherlock/dataset/PDE/Irregular/Aletheia/typeI_double-layer"
#DATA_DIR_STR="/home/data8t/sherlock/dataset/PDE/Irregular/Aletheia/typeI_double-layer"
#CRACK_TYPE=2
#3
#DATA_DIR_UNS="/home/data8t/sherlock/dataset/PDE/Irregular/Aletheia/typeII_double-layer"
#DATA_DIR_STR="/home/data8t/sherlock/dataset/PDE/Irregular/Aletheia/typeII_double-layer"
#CRACK_TYPE=3
#4
#DATA_DIR_UNS="/home/data8t/sherlock/dataset/PDE/Irregular/Aletheia/typeIII_double-layer"
#DATA_DIR_STR="/home/data8t/sherlock/dataset/PDE/Irregular/Aletheia/typeIII_double-layer"
#CRACK_TYPE=4
#5
#DATA_DIR_UNS="/home/data8t/sherlock/dataset/PDE/Irregular/Aletheia/typeI_multi-layer"
#DATA_DIR_STR="/home/data8t/sherlock/dataset/PDE/Irregular/Aletheia/typeI_multi-layer"
#CRACK_TYPE=5
#6
DATA_DIR_UNS="/home/data8t/sherlock/dataset/PDE/Irregular/Aletheia/typeII_multi-layer"
DATA_DIR_STR="/home/data8t/sherlock/dataset/PDE/Irregular/Aletheia/typeII_multi-layer"
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
TRAINING_NUM=2
TEST_SPLIT=0.2
DATA_NUM=600


MODELS_UNS="Transolver,MLP,FNO3d,GeoFNO,FFNO,FCNO,LNO,DeepONet,LaMO"
MODELS_STR="Transolver,MLP,FNO3d,GeoFNO,FFNO,FCNO,LNO,DeepONet,LaMO"

run_phase() {
  local TAG="$1" DTYPE="$2" ROOT="$3" OOD_TYPE="$4" MODELS_LIST="$5"

  echo "[RUN] S2Q | tag=${TAG} | dtype=${DTYPE} | OOD=${OOD_TYPE} | root=${ROOT}"
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
       --OOD "${OOD_TYPE}")
  if [ "${USE_SURF}" -eq 1 ]; then CMD+=(--use_surf); fi

  ( cd "${REPO_ROOT}" && CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" )
}


run_phase "uns_normal"    "unstructured_data"  "${DATA_DIR_UNS}"  ""      "${MODELS_UNS}"

run_phase "str_normal"    "structured_data"    "${DATA_DIR_STR}"  ""      "${MODELS_STR}"

run_phase "uns_ood_high"  "unstructured_data"  "${DATA_DIR_UNS}"  "high"  "${MODELS_UNS}"

run_phase "str_ood_high"  "structured_data"    "${DATA_DIR_STR}"  "high"  "${MODELS_STR}"

run_phase "uns_ood_mid"   "unstructured_data"  "${DATA_DIR_UNS}"  "mid"   "${MODELS_UNS}"

run_phase "str_ood_mid"   "structured_data"    "${DATA_DIR_STR}"  "mid"   "${MODELS_STR}"

run_phase "uns_ood_low"   "unstructured_data"  "${DATA_DIR_UNS}"  "low"   "${MODELS_UNS}"

run_phase "str_ood_low"   "structured_data"    "${DATA_DIR_STR}"  "low"   "${MODELS_STR}"

echo "[MAIN] All S2Q experiments finished."