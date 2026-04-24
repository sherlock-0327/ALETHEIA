#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${REPO_ROOT}/run_all_models_exps.py"
[ -f "${PY}" ] || { echo "[ERROR] Not found: ${PY}"; exit 1; }

check_dir() { [ -d "$1" ] || { echo "[ERROR] Not found: $1"; exit 1; }; }

DATA_BASE="/home/data8t/sherlock/dataset/PDE/Irregular/Aletheia"

declare -A DATASET_SUBDIR=(
  [1]="single_layer"
  [2]="typeI_double-layer"
  [3]="typeII_double-layer"
  [4]="typeIII_double-layer"
  [5]="typeI_multi-layer"
  [6]="typeII_multi-layer"
)

IN_CH=13
OUT_CH=1
USE_SURF=0

GPU=0
EPOCHS=200
BATCH_SIZE=4
LR=0.0001
DOWNSAMPLE_COUNT=8000
SURF_DOWNSAMPLE_COUNT=8000
TRAINING_NUM=3
TEST_SPLIT=0.2
DATA_NUM=600

SFO_TRAIN_NUM=480
SFO_TEST_NUM=120

MODELS_UNS="Transolver,MLP,FNO3d,GeoFNO,FFNO,FCNO,LNO,DeepONet"
MODELS_STR="Transolver,MLP,FNO3d,GeoFNO,FFNO,FCNO,LNO,DeepONet"

run_phase() {
  local CRACK_TYPE="$1" TAG="$2" DTYPE="$3" ROOT="$4" FREQ_INDEX="$5" MODELS_LIST="$6"

  echo "[RUN] T2Q | crack_type=${CRACK_TYPE} | tag=${TAG} | dtype=${DTYPE} | SFO_freq=${FREQ_INDEX} | root=${ROOT}"
  CMD=(python "${PY}"
       --mode T2Q
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

run_all_sfo_for_type() {
  local crack_type="$1" data_dir_uns="$2" data_dir_str="$3"

  local freq_label freq_index
  for freq_label in 9kHz 36kHz 81kHz; do
    case "${freq_label}" in
      9kHz)  freq_index=3 ;;
      36kHz) freq_index=6 ;;
      81kHz) freq_index=9 ;;
      *) echo "[ERROR] Unsupported freq label: ${freq_label}"; exit 1 ;;
    esac

    run_phase "${crack_type}" "SFO_${freq_label}_uns" "unstructured_data" "${data_dir_uns}" "${freq_index}" "${MODELS_UNS}"
    run_phase "${crack_type}" "SFO_${freq_label}_s"   "structured_data"   "${data_dir_str}" "${freq_index}" "${MODELS_STR}"
  done
}

for CRACK_TYPE in 6; do
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

  run_all_sfo_for_type "${CRACK_TYPE}" "${DATA_DIR_UNS}" "${DATA_DIR_STR}"
done

echo "[MAIN] All T2Q SFO OOD experiments finished for crack_type 2/4/6."
