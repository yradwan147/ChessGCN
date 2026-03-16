#!/bin/bash
# ============================================================
# Architecture + MCTS Sweep: 12 parallel experiments
# ============================================================
#
# Settled decisions (all runs):
#   - Gumbel MCTS (10x faster than PUCT)
#   - Smooth temp annealing, 12 eval games, 55% gate
#   - 30 iterations, buffer 300K, 200 train steps
#
# Variables under test:
#   - Architecture: hidden dim (128 vs 192), blocks (4 vs 5), heads (1 vs 4)
#   - MCTS: sims (16 vs 32 vs 64), games/iter (3 vs 5)
#   - Value head: frozen vs unfrozen
#
# Args to run_experiment.sh: RUN_TAG HIDDEN HEADS BLOCKS SIMS GAMES [EXTRA_FLAGS]
#
# Usage: bash slurm/submit_arch_sweep.sh

set -e
mkdir -p slurm/slurm_logs results

COUNT=0

echo "============================================================"
echo "ChessGCN Architecture + MCTS Sweep (12 runs)"
echo "============================================================"
echo ""
echo "All runs use: Gumbel MCTS, smooth temp, 30 iters, 12 eval games"
echo ""

# ═══════════════════════════════════════════════════════════════
# GROUP A: Architecture scaling (frozen value head, 32 sims, 3 games)
# ═══════════════════════════════════════════════════════════════

echo "── Group A: Architecture scaling (all frozen, 32 sims, 3 games) ──"
echo ""

# Run 1: Current arch baseline
echo "  Run 1:  128h 4blk 4head  (baseline)"
sbatch -J r1_baseline slurm/run_experiment.sh r1_baseline 128 4 4 32 3 --freeze-value
COUNT=$((COUNT + 1))

# Run 2: Single-head (AlphaGateau finding)
echo "  Run 2:  128h 4blk 1head  (single head)"
sbatch -J r2_1head slurm/run_experiment.sh r2_1head 128 1 4 32 3 --freeze-value
COUNT=$((COUNT + 1))

# Run 3: Wider
echo "  Run 3:  192h 4blk 1head  (wider)"
sbatch -J r3_wide slurm/run_experiment.sh r3_wide 192 1 4 32 3 --freeze-value
COUNT=$((COUNT + 1))

# Run 4: Deeper
echo "  Run 4:  128h 5blk 1head  (deeper)"
sbatch -J r4_deep slurm/run_experiment.sh r4_deep 128 1 5 32 3 --freeze-value
COUNT=$((COUNT + 1))

# Run 5: AlphaGateau scale (wide + deep)
echo "  Run 5:  192h 5blk 1head  (AlphaGateau scale)"
sbatch -J r5_agscale slurm/run_experiment.sh r5_agscale 192 1 5 32 3 --freeze-value
COUNT=$((COUNT + 1))

# Run 6: 2-head compromise
echo "  Run 6:  128h 4blk 2head  (2-head compromise)"
sbatch -J r6_2head slurm/run_experiment.sh r6_2head 128 2 4 32 3 --freeze-value
COUNT=$((COUNT + 1))

echo ""

# ═══════════════════════════════════════════════════════════════
# GROUP B: MCTS sim count scaling (best small + best large arch)
# ═══════════════════════════════════════════════════════════════

echo "── Group B: Simulation count (frozen, 3 games) ──"
echo ""

# Run 7: 16 sims on small arch (was fast in prior runs)
echo "  Run 7:  128h 4blk 1head 16sim  (low sims)"
sbatch -J r7_16sim slurm/run_experiment.sh r7_16sim 128 1 4 16 3 --freeze-value
COUNT=$((COUNT + 1))

# Run 8: 64 sims on large arch
echo "  Run 8:  192h 5blk 1head 64sim  (high sims + big arch)"
sbatch -J r8_64sim_big slurm/run_experiment.sh r8_64sim_big 192 1 5 64 3 --freeze-value
COUNT=$((COUNT + 1))

echo ""

# ═══════════════════════════════════════════════════════════════
# GROUP C: Data volume (more games per iteration)
# ═══════════════════════════════════════════════════════════════

echo "── Group C: Data volume (frozen, 32 sims) ──"
echo ""

# Run 9: More games on small arch
echo "  Run 9:  128h 4blk 1head 32sim 5g  (more games, small arch)"
sbatch -J r9_5games slurm/run_experiment.sh r9_5games 128 1 4 32 5 --freeze-value
COUNT=$((COUNT + 1))

# Run 10: More games on large arch
echo "  Run 10: 192h 5blk 1head 32sim 5g  (more games, big arch)"
sbatch -J r10_5games_big slurm/run_experiment.sh r10_5games_big 192 1 5 32 5 --freeze-value
COUNT=$((COUNT + 1))

echo ""

# ═══════════════════════════════════════════════════════════════
# GROUP D: Value head unfrozen (confirm frozen is better)
# ═══════════════════════════════════════════════════════════════

echo "── Group D: Value head unfrozen (32 sims, 3 games) ──"
echo ""

# Run 11: Small arch, unfrozen — direct comparison with Run 2
echo "  Run 11: 128h 4blk 1head UNFROZEN  (vs Run 2)"
sbatch -J r11_unfrozen slurm/run_experiment.sh r11_unfrozen 128 1 4 32 3
COUNT=$((COUNT + 1))

# Run 12: Large arch, unfrozen — direct comparison with Run 5
echo "  Run 12: 192h 5blk 1head UNFROZEN  (vs Run 5)"
sbatch -J r12_unfrozen_big slurm/run_experiment.sh r12_unfrozen_big 192 1 5 32 3
COUNT=$((COUNT + 1))

echo ""
echo "============================================================"
echo "Submitted $COUNT jobs"
echo "============================================================"
echo ""
echo "  GROUP A — Architecture (frozen, 32sim, 3g):"
echo "    Run 1:  r1_baseline     128h 4blk 4head"
echo "    Run 2:  r2_1head        128h 4blk 1head"
echo "    Run 3:  r3_wide         192h 4blk 1head"
echo "    Run 4:  r4_deep         128h 5blk 1head"
echo "    Run 5:  r5_agscale      192h 5blk 1head"
echo "    Run 6:  r6_2head        128h 4blk 2head"
echo ""
echo "  GROUP B — Sims (frozen, 3g):"
echo "    Run 7:  r7_16sim        128h 4blk 1head 16sim"
echo "    Run 8:  r8_64sim_big    192h 5blk 1head 64sim"
echo ""
echo "  GROUP C — Data volume (frozen, 32sim):"
echo "    Run 9:  r9_5games       128h 4blk 1head 5g"
echo "    Run 10: r10_5games_big  192h 5blk 1head 5g"
echo ""
echo "  GROUP D — Value head unfrozen (32sim, 3g):"
echo "    Run 11: r11_unfrozen    128h 4blk 1head"
echo "    Run 12: r12_unfrozen_big 192h 5blk 1head"
echo ""
echo "Key comparisons:"
echo "  1 vs 2:         4-head vs 1-head (same arch)"
echo "  2 vs 6:         1-head vs 2-head"
echo "  2 vs 3:         width scaling (128 vs 192)"
echo "  2 vs 4:         depth scaling (4blk vs 5blk)"
echo "  3 vs 4:         width vs depth (~same param count)"
echo "  5 vs 3, 5 vs 4: width+depth vs each alone"
echo "  7 vs 2 vs 8:    sim count (16 vs 32 vs 64)"
echo "  2 vs 9:         3 vs 5 games/iter (small arch)"
echo "  5 vs 10:        3 vs 5 games/iter (big arch)"
echo "  2 vs 11:        frozen vs unfrozen (small arch)"
echo "  5 vs 12:        frozen vs unfrozen (big arch)"
echo "============================================================"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  for f in slurm/slurm_logs/r*_*.out; do echo \"--- \$f ---\"; tail -3 \"\$f\"; done"
echo ""
echo "Quick policy loss comparison:"
echo "  for d in results/r*/; do printf '%-25s ' \"\$d\"; grep 'policy_loss' \"\${d}selfplay.log\" 2>/dev/null | tail -1 | grep -o 'pol_loss=[0-9.]*' || echo 'pending'; done"
echo ""
echo "Eval win rates:"
echo "  for d in results/r*/; do printf '%-25s ' \"\$d\"; grep 'Eval:' \"\${d}selfplay.log\" 2>/dev/null | tail -1 || echo 'pending'; done"
