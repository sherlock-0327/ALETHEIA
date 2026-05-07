#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="${REPO_ROOT}/multi_crack_exps.py"
[ -f "${PY}" ] || { echo "[ERROR] Not found: ${PY}"; exit 1; }

DATA_BASE="/data path/"

MODE="T2Q"
IN_CH=13
OUT_CH=1
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

SFO_TRAIN_NUM=480
SFO_TEST_NUM=120

MODELS="Transolver,MLP,FNO3d,GeoFNO,FFNO,FCNO,LNO,DeepONet,LaMO"

run_phase() {
  local multi_crack="$1" tag="$2" dtype="$3" ood_type="$4" models_list="$5"

  echo "[RUN] ${MODE} | multi_crack=${multi_crack} | tag=${tag} | dtype=${dtype} | OOD=${ood_type:-normal}"
  CMD=(python "${PY}"
       --mode "${MODE}"
       --data_root "${DATA_BASE}"
       --data_type "${dtype}"
       --multi_crack "${multi_crack}"
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

run_sfo_phase() {
  local multi_crack="$1" tag="$2" dtype="$3" freq_index="$4" models_list="$5"

  echo "[RUN] ${MODE} | multi_crack=${multi_crack} | tag=${tag} | dtype=${dtype} | SFO_freq=${freq_index}"
  CMD=(python "${PY}"
       --mode "${MODE}"
       --data_root "${DATA_BASE}"
       --data_type "${dtype}"
       --multi_crack "${multi_crack}"
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
       --OOD "sfo"
       --sfo_freq "${freq_index}"
       --sfo_train_num "${SFO_TRAIN_NUM}"
       --sfo_test_num "${SFO_TEST_NUM}")
  if [ "${USE_SURF}" -eq 1 ]; then CMD+=(--use_surf); fi

  ( cd "${REPO_ROOT}" && CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" )
}

# ============================================================
# Phase 1: Normal + OOD (crack types 1-6)
## ============================================================
#echo "=================================================="
#echo "[MAIN] Phase 1: Normal + OOD (multi_crack=123456)"
#echo "=================================================="
#
#run_phase "123456" "uns_normal"   "unstructured_data" ""     "${MODELS}"
#run_phase "123456" "str_normal"   "structured_data"   ""     "${MODELS}"
#
#run_phase "123456" "uns_ood_high" "unstructured_data" "high" "${MODELS}"
#run_phase "123456" "str_ood_high" "structured_data"   "high" "${MODELS}"
#
#run_phase "123456" "uns_ood_mid"  "unstructured_data" "mid"  "${MODELS}"
#run_phase "123456" "str_ood_mid"  "structured_data"   "mid"  "${MODELS}"
#
#run_phase "123456" "uns_ood_low"  "unstructured_data" "low"  "${MODELS}"
#run_phase "123456" "str_ood_low"  "structured_data"   "low"  "${MODELS}"

# ============================================================
# Phase 2: SFO (crack types 2/4/6)
# ============================================================

echo "=================================================="
echo "[MAIN] Phase 2: SFO (multi_crack=246)"
echo "=================================================="

run_phase "246" "uns_normal"   "unstructured_data" ""     "${MODELS}"
run_phase "246" "str_normal"   "structured_data"   ""     "${MODELS}"

for FREQ_LABEL in 9kHz 36kHz 81kHz; do
  case "${FREQ_LABEL}" in
    9kHz)  FREQ_INDEX=3 ;;
    36kHz) FREQ_INDEX=6 ;;
    81kHz) FREQ_INDEX=9 ;;
    *) echo "[ERROR] Unsupported freq label: ${FREQ_LABEL}"; exit 1 ;;
  esac

  run_sfo_phase "246" "SFO_${FREQ_LABEL}_uns" "unstructured_data" "${FREQ_INDEX}" "${MODELS}"
  run_sfo_phase "246" "SFO_${FREQ_LABEL}_s"   "structured_data"   "${FREQ_INDEX}" "${MODELS}"
done

echo "[MAIN] All ${MODE} multi-crack experiments finished."
