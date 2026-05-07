#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${REPO_ROOT}/run_all_models_exps.py"
[ -f "${PY}" ] || { echo "[ERROR] Not found: ${PY}"; exit 1; }

check_dir() { [ -d "$1" ] || { echo "[ERROR] Not found: $1"; exit 1; }; }

# Dataset base path (shared prefix for all crack types)
DATA_BASE="/data path/"

# Crack type -> folder mapping
# 1: single_layer
# 2: typeI_double-layer
# 3: typeII_double-layer
# 4: typeIII_double-layer
# 5: typeI_multi-layer
# 6: typeII_multi-layer
declare -A DATASET_SUBDIR=(
  [1]="single_layer"
  [2]="typeI_double-layer"
  [3]="typeII_double-layer"
  [4]="typeIII_double-layer"
  [5]="typeI_multi-layer"
  [6]="typeII_multi-layer"
)

check_dir "${DATA_BASE}"

IN_CH=1
OUT_CH=13
USE_SURF=0

GPU=2
EPOCHS=200
BATCH_SIZE=4
LR=0.0001
DOWNSAMPLE_COUNT=8000
SURF_DOWNSAMPLE_COUNT=8000
TRAINING_NUM=3
TEST_SPLIT=0.2
DATA_NUM=600

MODELS_UNS="Transolver,MLP,FNO3d,GeoFNO,FFNO,FCNO,LNO,LaMO"
MODELS_STR="Transolver,MLP,FNO3d,GeoFNO,FFNO,FCNO,LNO,LaMO"

run_phase() {
  local crack_type="$1" tag="$2" dtype="$3" root="$4" ood_type="$5" models_list="$6"

  echo "[RUN] Q2T | crack_type=${crack_type} | tag=${tag} | dtype=${dtype} | OOD=${ood_type:-normal} | root=${root}"

  CMD=(python "${PY}"
       --mode Q2T
       --data_root "${root}"
       --data_type "${dtype}"
       --crack_type "${crack_type}"
       --in_channels "${IN_CH}"
       --out_channels "${OUT_CH}"
       --epochs "${EPOCHS}"
       --batch_size "${BATCH_SIZE}"
       --lr "${LR}"
       --downsample_count "${DOWNSAMPLE_COUNT}"
       --surf_downsample_count "${SURF_DOWNSAMPLE_COUNT}"
       --models "${models_list}"
       --training_num "${TRAINING_NUM}"
       --test_split "${TEST_SPLIT}"
       --data_num "${DATA_NUM}"
       --gpu 0
       --OOD "${ood_type}")
  if [ "${USE_SURF}" -eq 1 ]; then CMD+=(--use_surf); fi

  ( cd "${REPO_ROOT}" && CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" )
}

run_all_phases_for_type() {
  local crack_type="$1" data_dir_uns="$2" data_dir_str="$3"

  run_phase "${crack_type}" "uns_normal"    "unstructured_data" "${data_dir_uns}" ""     "${MODELS_UNS}"
  run_phase "${crack_type}" "str_normal"    "structured_data"   "${data_dir_str}" ""     "${MODELS_STR}"

  run_phase "${crack_type}" "uns_ood_high"  "unstructured_data" "${data_dir_uns}" "high" "${MODELS_UNS}"
  run_phase "${crack_type}" "str_ood_high"  "structured_data"   "${data_dir_str}" "high" "${MODELS_STR}"

  run_phase "${crack_type}" "uns_ood_mid"   "unstructured_data" "${data_dir_uns}" "mid"  "${MODELS_UNS}"
  run_phase "${crack_type}" "str_ood_mid"   "structured_data"   "${data_dir_str}" "mid"  "${MODELS_STR}"

  run_phase "${crack_type}" "uns_ood_low"   "unstructured_data" "${data_dir_uns}" "low"  "${MODELS_UNS}"
  run_phase "${crack_type}" "str_ood_low"   "structured_data"   "${data_dir_str}" "low"  "${MODELS_STR}"
}

for CRACK_TYPE in 1 2 3 4 5 6; do
  SUBDIR="${DATASET_SUBDIR[${CRACK_TYPE}]:-}"
  if [ -z "${SUBDIR}" ]; then
    echo "[ERROR] No dataset mapping for crack type ${CRACK_TYPE}"
    exit 1
  fi
  DATA_DIR_UNS="${DATA_BASE}/${SUBDIR}"
  DATA_DIR_STR="${DATA_BASE}/${SUBDIR}"

  check_dir "${DATA_DIR_UNS}"
  check_dir "${DATA_DIR_STR}"

  echo "=================================================="
  echo "[MAIN] Start crack_type=${CRACK_TYPE}, subdir=${SUBDIR}"
  echo "=================================================="

  run_all_phases_for_type "${CRACK_TYPE}" "${DATA_DIR_UNS}" "${DATA_DIR_STR}"
done

echo "[MAIN] All Q2T experiments finished for crack_type 1~6."