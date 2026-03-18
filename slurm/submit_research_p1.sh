#!/bin/bash
# ============================================================
# Phase 1 Research Sweep: Data + Training (no arch changes)
# ============================================================
# 12 runs, 40 iters each, testing: self-edges, is-in-check, WDL_K,
# EMA, games/iter, color flip, playout cap
#
# Usage: bash slurm/submit_research_p1.sh

set -e
mkdir -p slurm/slurm_logs results

COUNT=0
H=192; HEADS=1; BLK=4; SIMS=32
# Base flags shared by all runs
BASE="--freeze-value --iterations 40 --eval-interval 5 --gate-threshold 0.52 --buffer-window 10 --save-interval 5 --openings-file openings.txt --opening-ratio 0.7 --fen-pool-ratio 0.1"

echo "============================================================"
echo "Phase 1 Research Sweep (12 runs, 40 iters)"
echo "============================================================"
echo ""

# ── Isolated tests (each change alone) ──
echo "── Isolated tests ──"

# p1: Baseline (new defaults: self-edges ON, check ON, WDL_K=200)
echo "  p1: Baseline (new data.py defaults)"
sbatch --time=30:00:00 -J p1_baseline slurm/run_experiment.sh p1_baseline $H $HEADS $BLK $SIMS 3 $BASE

# p2: Self-edges only (disable check + old WDL_K)
echo "  p2: Self-edges only"
sbatch --time=30:00:00 -J p2_selfedge slurm/run_experiment.sh p2_selfedge $H $HEADS $BLK $SIMS 3 $BASE --no-check-feature --wdl-k 111

# p3: Is-in-check only (disable self-edges + old WDL_K)
echo "  p3: Check feature only"
sbatch --time=30:00:00 -J p3_check slurm/run_experiment.sh p3_check $H $HEADS $BLK $SIMS 3 $BASE --no-self-edges --wdl-k 111

# p4: WDL_K=200 only (disable self-edges + check)
echo "  p4: WDL_K=200 only"
sbatch --time=30:00:00 -J p4_wdlk slurm/run_experiment.sh p4_wdlk $H $HEADS $BLK $SIMS 3 $BASE --no-self-edges --no-check-feature

# p5: EMA only (old features)
echo "  p5: EMA only"
sbatch --time=30:00:00 -J p5_ema slurm/run_experiment.sh p5_ema $H $HEADS $BLK $SIMS 3 $BASE --no-self-edges --no-check-feature --wdl-k 111 --use-ema

# p6: 30 games/iter (old features, tests data volume)
echo "  p6: 30 games/iter"
sbatch --time=48:00:00 -J p6_30games slurm/run_experiment.sh p6_30games $H $HEADS $BLK $SIMS 30 $BASE --no-self-edges --no-check-feature --wdl-k 111

# p7: Color flip only (old features)
echo "  p7: Color flip"
sbatch --time=30:00:00 -J p7_colorflip slurm/run_experiment.sh p7_colorflip $H $HEADS $BLK $SIMS 3 $BASE --no-self-edges --no-check-feature --wdl-k 111 --color-flip

# p8: Playout cap only (old features)
echo "  p8: Playout cap"
sbatch --time=30:00:00 -J p8_playoutcap slurm/run_experiment.sh p8_playoutcap $H $HEADS $BLK $SIMS 3 $BASE --no-self-edges --no-check-feature --wdl-k 111 --playout-cap

COUNT=$((COUNT + 8))

echo ""
echo "── Combined tests ──"

# p9: All feature changes (self-edges + check + WDL_K)
echo "  p9: All features combined"
sbatch --time=30:00:00 -J p9_features slurm/run_experiment.sh p9_features $H $HEADS $BLK $SIMS 3 $BASE

# p10: All data volume (30 games + color flip + playout cap, old features)
echo "  p10: Data volume combined"
sbatch --time=48:00:00 -J p10_data slurm/run_experiment.sh p10_data $H $HEADS $BLK $SIMS 30 $BASE --no-self-edges --no-check-feature --wdl-k 111 --color-flip --playout-cap

# p11: Training process (EMA + playout cap, old features)
echo "  p11: Training process combined"
sbatch --time=30:00:00 -J p11_training slurm/run_experiment.sh p11_training $H $HEADS $BLK $SIMS 3 $BASE --no-self-edges --no-check-feature --wdl-k 111 --use-ema --playout-cap

# p12: EVERYTHING
echo "  p12: All Phase 1 changes"
sbatch --time=48:00:00 -J p12_all_p1 slurm/run_experiment.sh p12_all_p1 $H $HEADS $BLK $SIMS 30 $BASE --use-ema --color-flip --playout-cap

COUNT=$((COUNT + 4))

echo ""
echo "============================================================"
echo "Submitted $COUNT Phase 1 jobs"
echo "============================================================"
echo ""
echo "  p1  baseline      — new defaults (self-edges+check+WDL200)"
echo "  p2  selfedge      — self-edges only"
echo "  p3  check         — is-in-check only"
echo "  p4  wdlk          — WDL_K=200 only"
echo "  p5  ema           — EMA only"
echo "  p6  30games       — 30 games/iter (48h limit)"
echo "  p7  colorflip     — color flip only"
echo "  p8  playoutcap    — playout cap only"
echo "  p9  features      — self-edges+check+WDL200"
echo "  p10 data          — 30g+colorflip+playoutcap (48h)"
echo "  p11 training      — EMA+playoutcap"
echo "  p12 all_p1        — EVERYTHING (48h)"
echo ""
echo "Quick status:"
echo "  for d in results/p*/; do printf '%-20s ' \"\$d\"; grep 'Challenger' \"\${d}selfplay.log\" 2>/dev/null | tail -1 | grep -o '[0-9.]*%' || echo 'pending'; done"
