#!/bin/bash
# Run red team agent with each misleading bias description sequentially.
# The agent thinks it's attacking a nationality/age/politics probe,
# but the actual auditor is the gender probe.
#
# Usage:
#   bash run_misleading_experiments.sh [--coherence] [--max-evaluations N]
#
# Requires: eval server running with user_gender_probe

set -e

BASELINES="results/baseline_probe.json"
MAX_EVALS="${MAX_EVALUATIONS:-50}"
EXTRA_ARGS=""

# Parse optional flags
for arg in "$@"; do
    case $arg in
        --coherence) EXTRA_ARGS="$EXTRA_ARGS --coherence" ;;
        --max-evaluations=*) MAX_EVALS="${arg#*=}" ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
    esac
done

echo "=========================================="
echo "  Probe Coherence Experiments"
echo "  Max evaluations per run: $MAX_EVALS"
echo "  Extra args: $EXTRA_ARGS"
echo "=========================================="

echo ""
echo ">>> [1/4] Running CONTROL (standard gender probe instructions)..."
echo ""
python -m red_team_agent.agent \
    --env user_gender_probe \
    --baselines-file "$BASELINES" \
    --max-evaluations "$MAX_EVALS" \
    --coherence \
    --session "control_gender_$(date +%Y%m%d_%H%M%S)" \
    $EXTRA_ARGS

echo ""
echo ">>> [2/4] Running NATIONALITY (Japan vs US) mislead..."
echo ""
python -m red_team_agent.agent \
    --env user_gender_probe \
    --baselines-file "$BASELINES" \
    --max-evaluations "$MAX_EVALS" \
    --coherence \
    --session "mislead_nationality_$(date +%Y%m%d_%H%M%S)" \
    --method-description-file red_team_agent/method_descriptions/probe_user_gender_misleading.md \
    --target-model-description-file red_team_agent/target_model_descriptions/user_gender_misleading.md \
    --goal-description-file red_team_agent/goal_descriptions/user_gender_probe_misleading.md \
    $EXTRA_ARGS

echo ""
echo ">>> [3/4] Running AGE (under/over 25) mislead..."
echo ""
python -m red_team_agent.agent \
    --env user_gender_probe \
    --baselines-file "$BASELINES" \
    --max-evaluations "$MAX_EVALS" \
    --coherence \
    --session "mislead_age_$(date +%Y%m%d_%H%M%S)" \
    --method-description-file red_team_agent/method_descriptions/probe_user_gender_misleading_age.md \
    --target-model-description-file red_team_agent/target_model_descriptions/user_gender_misleading_age.md \
    --goal-description-file red_team_agent/goal_descriptions/user_gender_probe_misleading_age.md \
    $EXTRA_ARGS

echo ""
echo ">>> [4/4] Running POLITICS (Democrat vs Republican) mislead..."
echo ""
python -m red_team_agent.agent \
    --env user_gender_probe \
    --baselines-file "$BASELINES" \
    --max-evaluations "$MAX_EVALS" \
    --coherence \
    --session "mislead_politics_$(date +%Y%m%d_%H%M%S)" \
    --method-description-file red_team_agent/method_descriptions/probe_user_gender_misleading_politics.md \
    --target-model-description-file red_team_agent/target_model_descriptions/user_gender_misleading_politics.md \
    --goal-description-file red_team_agent/goal_descriptions/user_gender_probe_misleading_politics.md \
    $EXTRA_ARGS

echo ""
echo "=========================================="
echo "  All 4 experiments complete (1 control + 3 misleading)."
echo "  Results in: results/red_team_sessions/"
echo "=========================================="
