#!/bin/bash
# ============================================================
# Training Dynamics Sweep: break the policy loss plateau
# ============================================================
#
# Architecture locked: 192h, 4blk, 1head (winner of arch sweep)
# All runs: Gumbel MCTS, frozen value head, 32 sims
#
# Tests: iteration count, games/iter, eval frequency,
#        gating threshold, training steps
#
# Usage: bash slurm/submit_training_sweep.sh

set -e
mkdir -p slurm/slurm_logs results

COUNT=0
H=192; HEADS=1; BLK=4; SIMS=32

echo "============================================================"
echo "ChessGCN Training Dynamics Sweep"
echo "Architecture: ${H}h ${BLK}blk ${HEADS}head (locked)"
echo "============================================================"
echo ""

# ── Run A: Long run (100 iters, 5 games, eval every 5) ──
# Did we stop too early? r3 was still improving at iter 30.
echo "Run A: 100 iters, 5 games/iter, eval every 5"
sbatch -J td_long slurm/run_experiment.sh td_long $H $HEADS $BLK $SIMS 5 \
    --freeze-value --iterations 100 --eval-interval 5
COUNT=$((COUNT + 1))

# ── Run B: Aggressive gating (52% threshold, eval every 5) ──
# More model updates = faster learning?
echo "Run B: 60 iters, 3 games, gate at 52%, eval every 5"
sbatch -J td_aggr_gate slurm/run_experiment.sh td_aggr_gate $H $HEADS $BLK $SIMS 3 \
    --freeze-value --iterations 60 --eval-interval 5 --gate-threshold 0.52
COUNT=$((COUNT + 1))

# ── Run C: More training (400 steps, bigger buffer utilization) ──
# 200 steps on ~5K buffer may underfit. Double the gradient steps.
echo "Run C: 60 iters, 5 games, 400 train steps"
sbatch -J td_more_train slurm/run_experiment.sh td_more_train $H $HEADS $BLK $SIMS 5 \
    --freeze-value --iterations 60 --eval-interval 5 --train-steps 400
COUNT=$((COUNT + 1))

# ── Run D: Kitchen sink (best of A+B+C) ──
# Long run, more games, more training, aggressive gating.
echo "Run D: 100 iters, 5 games, 400 steps, gate at 52%, eval every 5"
sbatch -J td_kitchen_sink slurm/run_experiment.sh td_kitchen_sink $H $HEADS $BLK $SIMS 5 \
    --freeze-value --iterations 100 --eval-interval 5 --gate-threshold 0.52 --train-steps 400
COUNT=$((COUNT + 1))

echo ""
echo "============================================================"
echo "Submitted $COUNT jobs"
echo "============================================================"
echo ""
echo "  Run A: td_long          100 iters, 5g, eval@5, gate 55%"
echo "  Run B: td_aggr_gate      60 iters, 3g, eval@5, gate 52%"
echo "  Run C: td_more_train     60 iters, 5g, eval@5, 400 steps"
echo "  Run D: td_kitchen_sink  100 iters, 5g, eval@5, gate 52%, 400 steps"
echo ""
echo "Comparisons:"
echo "  A vs r3_wide:  Does 100 iters break the 2.6 plateau?"
echo "  B vs A:        Does lower gate threshold help?"
echo "  C vs A:        Does more training per iter help?"
echo "  D:             All improvements combined"
echo "============================================================"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  for d in results/td_*/; do printf '%-25s ' \"\$d\"; grep 'policy_loss' \"\${d}selfplay.log\" 2>/dev/null | tail -1 | grep -o 'pol_loss=[0-9.]*' || echo 'pending'; done"
echo ""
echo "Estimated times:"
echo "  Run A: ~10-12h (100 iters × 5 games)"
echo "  Run B: ~5-6h   (60 iters × 3 games)"
echo "  Run C: ~8-10h  (60 iters × 5 games + more training)"
echo "  Run D: ~14-16h (100 iters × 5 games + more training)"
