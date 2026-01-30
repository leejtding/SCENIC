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
SIM_MODEL=${SIM_MODEL:-PH}        # PH | PO | AFT
SCENARIO=${SCENARIO:-mixed}       # mixed | pure_interval | basic
MONITOR=${MONITOR:-low}           # low | high
P_COV=${P_COV:-5}                 # 5 | 100
P_AUX=${P_AUX:-5}                 # 1 | 5 | 50
VS=${VS:-F}                       # T | F
EPOCHS=${EPOCHS:-50}

# Optional: activate your environment
# source ~/.bashrc
# conda activate <env>

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python run_simulation.py \
  --sim_model "$SIM_MODEL" \
  --scenario "$SCENARIO" \
  --monitor "$MONITOR" \
  --p_cov "$P_COV" \
  --p_aux "$P_AUX" \
  --vs "$VS" \
  --epochs "$EPOCHS" \
  --rep "$SLURM_ARRAY_TASK_ID" \
  --device cpu \
  --outdir results
