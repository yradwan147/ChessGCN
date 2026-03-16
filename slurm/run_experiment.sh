#!/bin/bash --login
# SLURM runner for ChessGCN self-play experiments
#
# Usage: sbatch -J run1 slurm/run_experiment.sh <RUN_TAG> <HIDDEN> <HEADS> <BLOCKS> <SIMS> <GAMES> [EXTRA_FLAGS]
#   $1: RUN_TAG     — experiment identifier (e.g., "r1_baseline")
#   $2: HIDDEN      — hidden dimension (128, 192, 256)
#   $3: HEADS       — attention heads (1, 2, 4)
#   $4: BLOCKS      — ResGATv2 blocks (4, 5, 6)
#   $5: SIMS        — Gumbel MCTS simulations (16, 32, 64)
#   $6: GAMES       — games per self-play iteration (3, 5, 8)
#   $7+: EXTRA      — extra flags passed to selfplay.py (e.g., "" or "--freeze-value")

#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=batch
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH -o slurm/slurm_logs/%x_%J.out

set -e

# ── Conda ──
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate chessgcn

# ── Parse args ──
RUN_TAG=${1:?Must provide RUN_TAG}
HIDDEN=${2:-128}
HEADS=${3:-4}
BLOCKS=${4:-4}
SIMS=${5:-32}
GAMES=${6:-3}
shift 6
EXTRA_FLAGS="$*"

# ── Paths ──
PROJECT_DIR="${SLURM_SUBMIT_DIR:-.}"
WORKDIR="${PROJECT_DIR}/workdirs/${RUN_TAG}"
OUTDIR="${PROJECT_DIR}/results/${RUN_TAG}"
mkdir -p "$WORKDIR" "$OUTDIR"

# ── Set up isolated workdir (symlinks to save disk, copies only .py) ──
cp "${PROJECT_DIR}"/*.py "$WORKDIR/"
ln -sf "${PROJECT_DIR}/dataset_eval.csv" "$WORKDIR/dataset_eval.csv"
cd "$WORKDIR"

echo "============================================================"
echo "[CONFIG] Run: ${RUN_TAG}"
echo "  Model:  hidden=${HIDDEN}, heads=${HEADS}, blocks=${BLOCKS}"
echo "  MCTS:   Gumbel, sims=${SIMS}, games/iter=${GAMES}"
echo "  Extra:  ${EXTRA_FLAGS:-none}"
echo "  Device: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Workdir: ${WORKDIR}"
echo "  Output:  ${OUTDIR}"
echo "  Disk:    $(df -h . | tail -1 | awk '{print $4 " available"}')"
echo "============================================================"

# ── Phase 1: Supervised training ──
echo ""
echo "[PHASE 1] Supervised training..."
python train.py \
    --data dataset_eval.csv \
    --num-samples 999999 \
    --hidden "$HIDDEN" \
    --heads "$HEADS" \
    --blocks "$BLOCKS" \
    --epochs 100 \
    --patience 20 \
    --batch-size 256 \
    --lr 1e-3 \
    --no-cache 2>&1 | tee "${OUTDIR}/supervised.log"

# Copy supervised checkpoint
cp best_model.pt "${OUTDIR}/best_model_supervised.pt"
echo "[PHASE 1] Done. Checkpoint saved to ${OUTDIR}/best_model_supervised.pt"

# ── Phase 2: Self-play ──
echo ""
echo "[PHASE 2] Self-play training (Gumbel MCTS, ${SIMS} sims, ${GAMES} games/iter)..."
python selfplay.py \
    --hidden "$HIDDEN" \
    --heads "$HEADS" \
    --blocks "$BLOCKS" \
    --checkpoint best_model.pt \
    --pretrain-policy \
    --pretrain-data dataset_eval.csv \
    --pretrain-epochs 10 \
    --use-gumbel \
    --iterations 30 \
    --games-per-iter "$GAMES" \
    --simulations "$SIMS" \
    --max-moves 200 \
    --buffer-size 300000 \
    --lr 1e-4 \
    --batch-size 256 \
    --train-steps 200 \
    --eval-interval 10 \
    --eval-games 12 \
    --eval-sims 64 \
    --save-interval 5 \
    --games-file "${OUTDIR}/selfplay_games.jsonl" \
    $EXTRA_FLAGS 2>&1 | tee "${OUTDIR}/selfplay.log"

# Copy final checkpoints to results
for f in selfplay_final.pt selfplay_latest.pt selfplay_best_*.pt pretrained_dual_head.pt; do
    [ -f "$f" ] && cp "$f" "${OUTDIR}/"
done

echo ""
echo "[SUCCESS] ${RUN_TAG} complete"
echo "  Results in: ${OUTDIR}/"
ls -la "${OUTDIR}/"

# Clean up workdir (keep results, drop copies)
rm -rf "$WORKDIR"
