#!/bin/bash
# ============================================================
# Resubmit failed Phase 1 + Phase 2 runs
# ============================================================
# These failed due to node_dim mismatch (monkey-patching bug).
# Fixed in commit 74f511d: module-level defaults instead of patching.
#
# Usage: cd ~/ChessGCN && git pull && bash slurm/submit_rerun_failed.sh

set -e
mkdir -p slurm/slurm_logs results

COUNT=0
H=192; HEADS=1; BLK=4; SIMS=32

# Shared base for Phase 1
P1_BASE="--freeze-value --iterations 40 --eval-interval 5 --gate-threshold 0.52 --buffer-window 10 --save-interval 5 --openings-file openings.txt --opening-ratio 0.7 --fen-pool-ratio 0.2"

# Shared base for Phase 2 (old features to isolate arch changes)
P2_BASE="--freeze-value --iterations 40 --eval-interval 5 --gate-threshold 0.52 --buffer-window 10 --save-interval 5 --openings-file openings.txt --opening-ratio 0.7 --fen-pool-ratio 0.2 --no-self-edges --no-check-feature --wdl-k 111"

echo "============================================================"
echo "Resubmit 14 failed runs (node_dim bug fixed)"
echo "============================================================"
echo ""

# ── Phase 1 failed runs (8) ──
echo "── Phase 1 (8 reruns) ──"

echo "  p2: Self-edges only"
sbatch --time=36:00:00 -J p2_selfedge slurm/run_experiment.sh p2_selfedge $H $HEADS $BLK $SIMS 3 $P1_BASE --no-check-feature --wdl-k 111
COUNT=$((COUNT + 1))

echo "  p4: WDL_K=200 only"
sbatch --time=36:00:00 -J p4_wdlk slurm/run_experiment.sh p4_wdlk $H $HEADS $BLK $SIMS 3 $P1_BASE --no-self-edges --no-check-feature
COUNT=$((COUNT + 1))

echo "  p5: EMA only"
sbatch --time=36:00:00 -J p5_ema slurm/run_experiment.sh p5_ema $H $HEADS $BLK $SIMS 3 $P1_BASE --no-self-edges --no-check-feature --wdl-k 111 --use-ema
COUNT=$((COUNT + 1))

echo "  p6: 30 games/iter"
sbatch --time=48:00:00 -J p6_30games slurm/run_experiment.sh p6_30games $H $HEADS $BLK $SIMS 30 $P1_BASE --no-self-edges --no-check-feature --wdl-k 111
COUNT=$((COUNT + 1))

echo "  p7: Color flip"
sbatch --time=36:00:00 -J p7_colorflip slurm/run_experiment.sh p7_colorflip $H $HEADS $BLK $SIMS 3 $P1_BASE --no-self-edges --no-check-feature --wdl-k 111 --color-flip
COUNT=$((COUNT + 1))

echo "  p8: Playout cap"
sbatch --time=36:00:00 -J p8_playoutcap slurm/run_experiment.sh p8_playoutcap $H $HEADS $BLK $SIMS 3 $P1_BASE --no-self-edges --no-check-feature --wdl-k 111 --playout-cap
COUNT=$((COUNT + 1))

echo "  p10: Data volume combined (30g + color flip + playout cap)"
sbatch --time=48:00:00 -J p10_data slurm/run_experiment.sh p10_data $H $HEADS $BLK $SIMS 30 $P1_BASE --no-self-edges --no-check-feature --wdl-k 111 --color-flip --playout-cap
COUNT=$((COUNT + 1))

echo "  p11: Training process combined (EMA + playout cap)"
sbatch --time=36:00:00 -J p11_training slurm/run_experiment.sh p11_training $H $HEADS $BLK $SIMS 3 $P1_BASE --no-self-edges --no-check-feature --wdl-k 111 --use-ema --playout-cap
COUNT=$((COUNT + 1))

echo ""

# ── Phase 2 failed runs (6) ──
echo "── Phase 2 (6 reruns) ──"

echo "  a1: Baseline (no arch changes)"
sbatch --time=36:00:00 -J a1_baseline slurm/run_experiment.sh a1_baseline $H $HEADS $BLK $SIMS 3 $P2_BASE
COUNT=$((COUNT + 1))

echo "  a2: Attention pooling"
sbatch --time=36:00:00 -J a2_attnpool slurm/run_experiment.sh a2_attnpool $H $HEADS $BLK $SIMS 3 $P2_BASE --attn-pool
COUNT=$((COUNT + 1))

echo "  a3: Moves-left head"
sbatch --time=36:00:00 -J a3_movesleft slurm/run_experiment.sh a3_movesleft $H $HEADS $BLK $SIMS 3 $P2_BASE --moves-left-head
COUNT=$((COUNT + 1))

echo "  a4: Score distribution (128 bins)"
sbatch --time=36:00:00 -J a4_scoredist slurm/run_experiment.sh a4_scoredist $H $HEADS $BLK $SIMS 3 $P2_BASE --score-dist-bins 128
COUNT=$((COUNT + 1))

echo "  a5: Attn pool + moves-left"
sbatch --time=36:00:00 -J a5_arch_combo slurm/run_experiment.sh a5_arch_combo $H $HEADS $BLK $SIMS 3 $P2_BASE --attn-pool --moves-left-head
COUNT=$((COUNT + 1))

echo "  a6: All Phase 2 changes"
sbatch --time=36:00:00 -J a6_all_p2 slurm/run_experiment.sh a6_all_p2 $H $HEADS $BLK $SIMS 3 $P2_BASE --attn-pool --moves-left-head --score-dist-bins 128
COUNT=$((COUNT + 1))

echo ""
echo "============================================================"
echo "Submitted $COUNT jobs"
echo "============================================================"
echo ""
echo "Note: FEN pool ratio updated to 0.2 (from opening sweep winner o7_fen20)"
echo ""
echo "Quick status:"
echo "  for d in results/p*/ results/a*/; do printf '%-20s ' \"\$d\"; grep 'Challenger' \"\${d}selfplay.log\" 2>/dev/null | tail -1 | grep -o '[0-9.]*%' || echo 'pending'; done"
