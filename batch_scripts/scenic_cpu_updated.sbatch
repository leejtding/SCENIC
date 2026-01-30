#!/bin/bash
#SBATCH --job-name=scenic_sim
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-99

set -euo pipefail

mkdir -p logs results

# ---- User-configurable design ----
METHOD=${METHOD:-scenic}         # scenic | spinet
SIM_MODEL=${SIM_MODEL:-PH}       # PH | PO | AFT
CENSOR=${CENSOR:-mixed}          # mixed | pure_interval
MONITOR=${MONITOR:-low}          # low | high
P_COV=${P_COV:-5}                # 5 | 100
P_AUX=${P_AUX:-5}                # 1 | 5 | 50

# SCENIC-only knobs (ignored by spinet)
VS=${VS:-F}                      # T | F
EPOCHS=${EPOCHS:-50}

# SPINet-only knobs (ignored by scenic)
SPINET_HIDDEN=${SPINET_HIDDEN:-10,10}   # e.g. "10,10"
SPINET_DENSE_ONLY=${SPINET_DENSE_ONLY:-0}  # 1 to set --spinet_dense_only

# Optional: activate your environment
# source ~/.bashrc
# conda activate <env>

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

EXTRA_SPINET_FLAGS=()
if [[ "$SPINET_DENSE_ONLY" == "1" ]]; then
  EXTRA_SPINET_FLAGS+=(--spinet_dense_only)
fi

python run_simulation_spinet.py   --method "$METHOD"   --sim_model "$SIM_MODEL"   --censor "$CENSOR"   --monitor "$MONITOR"   --p_cov "$P_COV"   --p_aux "$P_AUX"   --vs "$VS"   --epochs "$EPOCHS"   --spinet_hidden "$SPINET_HIDDEN"   --rep "$SLURM_ARRAY_TASK_ID"   --device cpu   --outdir results   "${EXTRA_SPINET_FLAGS[@]}"
