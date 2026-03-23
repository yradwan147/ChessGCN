#!/bin/bash
# ============================================================
# Final Best-Model Runs — 6 experiments, 80 iterations each
# ============================================================
#
# Based on 50+ prior experiments. Features selected for no interference:
#   - Is-in-check (p3: 7 gates, best single feature)
#   - Moves-left head (a3: 5 gates, 75% final eval)
#   - Self-edges (p2: 5 gates)
#   - Color flip (p7: 5 gates, stable)
#   - WDL_K=111 (NOT 200 — causes interference when combined)
#
# Excluded (proven harmful or marginal):
#   - Attention pooling (collapsed in a2/a5)
#   - 30 games/iter (no benefit in p6/p10)
#   - EMA (marginal in p5)
#   - WDL_K=200 (spiky when combined with other features)
#
# Usage: bash slurm/submit_final_runs.sh

set -e
mkdir -p slurm/slurm_logs results

COUNT=0
H=192; HEADS=1; BLK=4; SIMS=32

# Shared base config (all proven settings)
BASE="--freeze-value --iterations 80 --eval-interval 5 --gate-threshold 0.52 --buffer-window 10 --save-interval 5 --openings-file openings.txt --opening-ratio 0.7 --fen-pool-ratio 0.2"
# Force old features as baseline (no self-edges, no check, old WDL_K)
OLD="--no-self-edges --no-check-feature --wdl-k 111"

echo "============================================================"
echo "Final Best-Model Runs (6 runs, 80 iters)"
echo "============================================================"
echo ""

# ── f1: Control (prior best, no research features) ──
echo "  f1: Control — prior best config, no new features"
sbatch --time=36:00:00 -J f1_control slurm/run_experiment.sh f1_control $H $HEADS $BLK $SIMS 3 \
    $BASE $OLD
COUNT=$((COUNT + 1))

# ── f2: Top 2 winners (check + moves-left) ──
echo "  f2: Check + moves-left (top 2 winners)"
sbatch --time=36:00:00 -J f2_check_ml slurm/run_experiment.sh f2_check_ml $H $HEADS $BLK $SIMS 3 \
    $BASE --no-self-edges --wdl-k 111 --moves-left-head
COUNT=$((COUNT + 1))

# ── f3: Top 3 (check + self-edges + moves-left) ──
echo "  f3: Check + self-edges + moves-left"
sbatch --time=30:00:00 -J f3_safe slurm/run_experiment.sh f3_safe $H $HEADS $BLK $SIMS 3 \
    $BASE --wdl-k 111 --moves-left-head
COUNT=$((COUNT + 1))

# ── f4: Top 4 (check + self-edges + moves-left + color flip) ──
echo "  f4: Full feature set (check + self-edges + moves-left + flip)"
sbatch --time=30:00:00 -J f4_full slurm/run_experiment.sh f4_full $H $HEADS $BLK $SIMS 3 \
    $BASE --wdl-k 111 --moves-left-head --color-flip
COUNT=$((COUNT + 1))

# ── f5: Big dataset + top 2 ──
echo "  f5: Big dataset (1M) + check + moves-left"
sbatch --time=48:00:00 -J f5_bigdata slurm/run_experiment.sh f5_bigdata $H $HEADS $BLK $SIMS 3 \
    $BASE --no-self-edges --wdl-k 111 --moves-left-head --pretrain-data big_dataset.csv
COUNT=$((COUNT + 1))

# ── f6: Big dataset + full features ──
echo "  f6: Big dataset (1M) + all features"
sbatch --time=48:00:00 -J f6_bigdata_full slurm/run_experiment.sh f6_bigdata_full $H $HEADS $BLK $SIMS 3 \
    $BASE --wdl-k 111 --moves-left-head --color-flip --pretrain-data big_dataset.csv
COUNT=$((COUNT + 1))

echo ""
echo "============================================================"
echo "Submitted $COUNT final runs"
echo "============================================================"
echo ""
echo "  f1_control      — Prior best, no research features (baseline)"
echo "  f2_check_ml     — Check + moves-left (top 2 winners)"
echo "  f3_safe         — Check + self-edges + moves-left"
echo "  f4_full         — Check + self-edges + moves-left + color flip"
echo "  f5_bigdata      — 1M dataset + check + moves-left"
echo "  f6_bigdata_full — 1M dataset + all features"
echo ""
echo "Comparisons:"
echo "  f1 vs f2: Do top 2 winners stack?"
echo "  f2 vs f3: Does adding self-edges help or interfere?"
echo "  f3 vs f4: Does color flip add value on top?"
echo "  f2 vs f5: Does 20x more supervised data improve foundation?"
echo "  f4 vs f6: Does bigger data + all features = best model?"
echo ""
echo "Expected: f5 or f6 should be the best model."
echo ""
echo "Estimated times (based on actual GTX 1080 Ti measurements):"
echo "  f1: ~13h  (no self-edges, fast iters)"
echo "  f2: ~14h  (check+ml, no self-edges)"
echo "  f3: ~19h  (self-edges = 4x graph → slower iters)"
echo "  f4: ~19h  (same as f3 + color flip overhead)"
echo "  f5: ~16h  (1M dataset adds ~2.5h supervised)"
echo "  f6: ~21h  (1M + self-edges, slowest run)"
echo ""
echo "Monitor:"
echo "  for d in results/f[1-6]_*/; do"
echo "    name=\$(basename \"\$d\")"
echo "    gates=\$(grep -c 'New best' \"\${d}selfplay.log\" 2>/dev/null || echo 0)"
echo "    eval=\$(grep 'Challenger' \"\${d}selfplay.log\" 2>/dev/null | tail -1 | grep -o '[0-9.]*%')"
echo "    printf '%-20s gates=%-2s latest: %s\n' \"\$name\" \"\$gates\" \"\${eval:-pending}\""
echo "  done"
