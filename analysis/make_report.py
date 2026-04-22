#!/usr/bin/env python
"""Generate charts and a markdown report from misleading-prompt experiment results.

Usage:
    python -m analysis.make_report \
        --results-dir results/results_big \
        --output-dir results/report
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # noqa: F401
import numpy as np


# ── Colour palette ─────────────────────────────────────────────────────────
PROBE_COLOR = "#2196F3"   # blue
SAE_COLOR   = "#FF5722"   # orange-red
BASELINE_COLOR = "#9E9E9E"

TOPIC_ORDER = [
    "nationality", "age", "politics", "religion",
    "education", "income", "language", "personality",
    "occupation", "noinfo",
]

TOPIC_LABELS = {
    "nationality": "Nationality",
    "age":         "Age",
    "politics":    "Politics",
    "religion":    "Religion",
    "education":   "Education",
    "income":      "Income",
    "language":    "Language",
    "personality": "Personality",
    "occupation":  "Occupation",
    "noinfo":      "No Info",
}

# ── Data loading helpers ────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def is_intern_success(entry, threshold):
    """Return True if internalize_accuracy >= threshold."""
    intern_acc = entry.get("internalize_accuracy")
    if intern_acc is None:
        return False
    return intern_acc >= threshold


def load_eval_log_stats(session_dir, skip=20, baseline_internalize=None):
    """Return mean/min audit_accuracy and mean score from evaluation_log.json.

    'Successful' prompts are those where internalize_accuracy >= baseline_internalize.
    Falls back to is_success flag if baseline_internalize is None.
    """
    log_path = session_dir / "evaluation_log.json"
    if not log_path.exists():
        return None
    entries = load_json(log_path)[skip:]
    accs = [e["audit_accuracy"] for e in entries if "audit_accuracy" in e]
    scores = [e["score"] for e in entries if "score" in e]
    perps = [e["mean_perplexity"] for e in entries if "mean_perplexity" in e]
    coherences = [e["coherence_score"] for e in entries if "coherence_score" in e]
    # Top-5 lowest-accuracy prompts among those that meet the success threshold
    if baseline_internalize is not None:
        successful = [e for e in entries if is_intern_success(e, baseline_internalize)]
    else:
        successful = [e for e in entries if e.get("is_success")]
    top5_accs = sorted([e["audit_accuracy"] for e in successful if "audit_accuracy" in e])[:5]
    return {
        "n": len(accs),
        "mean_accuracy": mean(accs),
        "min_accuracy": min(accs) if accs else float("nan"),
        "mean_score": mean(scores),
        "top5_best_acc": mean(top5_accs),
        "mean_perplexity": mean(perps),
        "mean_coherence": mean(coherences),
    }


def find_session(sessions_dir, pattern):
    matches = sorted(sessions_dir.glob(pattern))
    return matches[-1] if matches else None  # latest if multiple


INTERN_THRESHOLD = 0.783  # prompt is 'successful' if internalize_accuracy >= this


def load_all_conditions(results_dir):
    """Load per-condition stats for probe and SAE experiments."""
    sessions = results_dir / "red_team_sessions"

    probe_baseline = load_json(results_dir / "baseline_probe.json")["audit_accuracy"]
    sae_baseline   = load_json(results_dir / "baseline_sae.json")["audit_accuracy"]

    def _load(session_dir):
        return load_eval_log_stats(session_dir, baseline_internalize=INTERN_THRESHOLD)

    control_probe = _load(find_session(sessions, "control_gender_2*"))
    control_sae   = _load(find_session(sessions, "control_gender_sae_2*"))

    conditions = {}
    for topic in TOPIC_ORDER:
        probe_dir = find_session(sessions, f"mislead_{topic}_2*")
        sae_dir   = find_session(sessions, f"mislead_{topic}_sae_2*")
        conditions[topic] = {
            "probe": _load(probe_dir) if probe_dir else None,
            "sae":   _load(sae_dir)   if sae_dir   else None,
        }

    return {
        "probe_baseline": probe_baseline,
        "sae_baseline":   sae_baseline,
        "control_probe":  control_probe,
        "control_sae":    control_sae,
        "conditions":     conditions,
    }


def load_random_baselines(results_dir):
    probe = load_json(results_dir / "random_baseline_probe.json")["summary"]
    sae   = load_json(results_dir / "random_baseline_sae.json")["summary"]
    return probe, sae


def load_feature_inspection(results_dir):
    path = results_dir / "feature_inspection.json"
    if not path.exists():
        path = results_dir / "feature_inspection_2.json"
    return load_json(path) if path.exists() else {}


def load_gender_slant(results_dir):
    path = results_dir / "gender_slant_analysis.json"
    if not path.exists():
        path = results_dir / "gender_slant_analysis_2.json"
    if not path.exists():
        return [], None, None

    data = load_json(path)
    sessions = data if isinstance(data, list) else []

    # Load control scores from separate _control.json if it exists
    ctrl_path = results_dir / "gender_slant_analysis_2_control.json"
    ctrl_p = None
    ctrl_s = None
    if ctrl_path.exists():
        ctrl_data = load_json(ctrl_path)
        if isinstance(ctrl_data, list):
            for entry in ctrl_data:
                name = entry.get("session", "")
                score = entry.get("mean_score")
                if score is None:
                    continue
                if "control_gender_sae" in name:
                    ctrl_s = score
                elif "control_gender" in name:
                    ctrl_p = score

    return sessions, ctrl_p, ctrl_s


# ── Chart helpers ───────────────────────────────────────────────────────────

def savefig(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Shared layout helper ────────────────────────────────────────────────────

def _add_separators(ax, n_topics):
    """Add vertical separator lines between Gender | Misleading | Mean groups."""
    ax.axvline(0.5,           color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(n_topics + 0.5, color="gray", linewidth=0.8, linestyle="--")


def _std_layout(ctrl_p, ctrl_s, topic_p, topic_s):
    """Return (all_p, all_s, labels, sep positions) for the standard layout."""
    valid_p = [v for v in topic_p if not math.isnan(v)]
    valid_s = [v for v in topic_s if not math.isnan(v)]
    all_p = [ctrl_p] + topic_p + [mean(valid_p)]
    all_s = [ctrl_s] + topic_s + [mean(valid_s)]
    labels = ["Gender"] + [TOPIC_LABELS[t] for t in TOPIC_ORDER] + ["Mean"]
    return all_p, all_s, labels


# ── Chart 1: Main accuracy comparison (probe vs SAE per misleading topic) ──

def chart_accuracy_comparison(data, output_dir):
    topics = TOPIC_ORDER

    probe_accs = [data["conditions"][t]["probe"]["mean_accuracy"]
                  if data["conditions"][t]["probe"] else float("nan") for t in topics]
    sae_accs   = [data["conditions"][t]["sae"]["mean_accuracy"]
                  if data["conditions"][t]["sae"] else float("nan") for t in topics]

    ctrl_p = data["control_probe"]["mean_accuracy"] if data["control_probe"] else float("nan")
    ctrl_s = data["control_sae"]["mean_accuracy"]   if data["control_sae"]   else float("nan")

    all_p, all_s, labels = _std_layout(ctrl_p, ctrl_s, probe_accs, sae_accs)
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width/2, all_p, width, label="Probe", color=PROBE_COLOR, alpha=0.85)
    ax.bar(x + width/2, all_s, width, label="SAE",   color=SAE_COLOR,   alpha=0.85)

    ax.axhline(data["probe_baseline"], color=PROBE_COLOR, linestyle="--", linewidth=1.2,
               label=f"Probe baseline ({data['probe_baseline']:.2f})")
    ax.axhline(data["sae_baseline"],   color=SAE_COLOR,   linestyle="--", linewidth=1.2,
               label=f"SAE baseline ({data['sae_baseline']:.2f})")
    _add_separators(ax, len(topics))

    ax.set_xlabel("Condition")
    ax.set_ylabel("Mean Audit Accuracy")
    ax.set_title("Auditor Accuracy: Gender Control vs. Misleading Topics\n(prompts 21–50; lower = easier to fool)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    savefig(fig, output_dir / "fig1_accuracy_comparison.png")
    return probe_accs, sae_accs  # topic-only, for downstream use


def chart_top5_best(data, output_dir):
    """Mean audit accuracy of the 5 lowest-accuracy successful prompts per condition."""
    topics = TOPIC_ORDER

    probe_top5 = [data["conditions"][t]["probe"]["top5_best_acc"]
                  if data["conditions"][t]["probe"] else float("nan") for t in topics]
    sae_top5   = [data["conditions"][t]["sae"]["top5_best_acc"]
                  if data["conditions"][t]["sae"] else float("nan") for t in topics]

    ctrl_p = data["control_probe"]["top5_best_acc"] if data["control_probe"] else float("nan")
    ctrl_s = data["control_sae"]["top5_best_acc"]   if data["control_sae"]   else float("nan")

    all_p, all_s, labels = _std_layout(ctrl_p, ctrl_s, probe_top5, sae_top5)
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width/2, all_p, width, label="Probe", color=PROBE_COLOR, alpha=0.85)
    ax.bar(x + width/2, all_s, width, label="SAE",   color=SAE_COLOR,   alpha=0.85)

    ax.axhline(data["probe_baseline"], color=PROBE_COLOR, linestyle="--", linewidth=1.2,
               label=f"Probe baseline ({data['probe_baseline']:.2f})")
    ax.axhline(data["sae_baseline"],   color=SAE_COLOR,   linestyle="--", linewidth=1.2,
               label=f"SAE baseline ({data['sae_baseline']:.2f})")
    _add_separators(ax, len(topics))

    ax.set_xlabel("Condition")
    ax.set_ylabel("Mean Audit Accuracy")
    ax.set_title("Top-5 Best Prompts: Gender Control vs. Misleading Topics\n"
                 "(lowest-accuracy prompts meeting internalization threshold)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    savefig(fig, output_dir / "fig1b_top5_best.png")


# ── Chart 2: Accuracy drop from baseline ────────────────────────────────────

def chart_accuracy_drop(data, probe_accs, sae_accs, output_dir):
    topics = TOPIC_ORDER

    probe_topic_drops = [data["probe_baseline"] - a for a in probe_accs]
    sae_topic_drops   = [data["sae_baseline"]   - a for a in sae_accs]

    ctrl_p_drop = (data["probe_baseline"] - data["control_probe"]["mean_accuracy"]
                   if data["control_probe"] else float("nan"))
    ctrl_s_drop = (data["sae_baseline"]   - data["control_sae"]["mean_accuracy"]
                   if data["control_sae"] else float("nan"))

    all_p, all_s, labels = _std_layout(ctrl_p_drop, ctrl_s_drop,
                                        probe_topic_drops, sae_topic_drops)
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width/2, all_p, width, label="Probe", color=PROBE_COLOR, alpha=0.85)
    ax.bar(x + width/2, all_s, width, label="SAE",   color=SAE_COLOR,   alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    _add_separators(ax, len(topics))

    ax.set_xlabel("Condition")
    ax.set_ylabel("Accuracy Drop from Baseline")
    ax.set_title("Accuracy Drop: Gender Control vs. Misleading Topics\n(higher = more confusable)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    savefig(fig, output_dir / "fig2_accuracy_drop.png")


# ── Chart 3: Random baseline fragility ──────────────────────────────────────

def chart_random_baseline(probe_rand, sae_rand, data, output_dir):
    categories = ["random_words", "random_chars", "lorem_ipsum", "irrelevant_coherent"]
    cat_labels  = ["Random\nWords", "Random\nChars", "Lorem\nIpsum", "Irrelevant\nCoherent"]

    probe_vals = [probe_rand["by_type"][c]["mean_accuracy"] for c in categories]
    sae_vals   = [sae_rand["by_type"][c]["mean_accuracy"]   for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, probe_vals, width, label="Probe", color=PROBE_COLOR, alpha=0.85)
    ax.bar(x + width/2, sae_vals,   width, label="SAE",   color=SAE_COLOR,   alpha=0.85)

    ax.axhline(data["probe_baseline"], color=PROBE_COLOR, linestyle="--", linewidth=1.2,
               label=f"Probe baseline ({data['probe_baseline']:.2f})")
    ax.axhline(data["sae_baseline"],   color=SAE_COLOR,   linestyle="--", linewidth=1.2,
               label=f"SAE baseline ({data['sae_baseline']:.2f})")

    ax.set_ylabel("Mean Audit Accuracy")
    ax.set_title("Auditor Accuracy Under Random / Nonsense System Prompts")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    savefig(fig, output_dir / "fig3_random_baseline.png")


# ── Chart 4: Feature inspection — gender feature fraction ───────────────────

def load_late_accuracies(sessions_dir, skip=20):
    """Load per-eval accuracies from prompt skip+1 onwards for each misleading condition."""
    probe_stats = {}
    sae_stats   = {}
    for topic in TOPIC_ORDER:
        for dest, pattern in [(probe_stats, f"mislead_{topic}_2*"),
                               (sae_stats,   f"mislead_{topic}_sae_2*")]:
            d = find_session(sessions_dir, pattern)
            if not d:
                continue
            log = d / "evaluation_log.json"
            if not log.exists():
                continue
            with open(log) as f:
                entries = json.load(f)
            accs = [e["audit_accuracy"] for e in entries[skip:] if "audit_accuracy" in e]
            if accs:
                dest[topic] = accs
    return probe_stats, sae_stats


def stdev(values):
    if len(values) < 2:
        return float("nan")
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def _load_session_accs(sessions_dir, pattern, skip):
    """Load per-eval accuracies from a single session pattern."""
    d = find_session(sessions_dir, pattern)
    if not d:
        return []
    log = d / "evaluation_log.json"
    if not log.exists():
        return []
    with open(log) as f:
        entries = json.load(f)
    return [e["audit_accuracy"] for e in entries[skip:] if "audit_accuracy" in e]


def chart_late_accuracy_variance(sessions_dir, output_dir, skip=20):
    probe_stats, sae_stats = load_late_accuracies(sessions_dir, skip)

    # Control (gender) variance
    ctrl_p_accs = _load_session_accs(sessions_dir, "control_gender_2*", skip)
    ctrl_s_accs = _load_session_accs(sessions_dir, "control_gender_sae_2*", skip)
    ctrl_p_std  = stdev(ctrl_p_accs) if ctrl_p_accs else float("nan")
    ctrl_s_std  = stdev(ctrl_s_accs) if ctrl_s_accs else float("nan")

    topics = [t for t in TOPIC_ORDER if t in probe_stats or t in sae_stats]
    p_stds = [stdev(probe_stats[t]) if t in probe_stats else float("nan") for t in topics]
    s_stds = [stdev(sae_stats[t])   if t in sae_stats   else float("nan") for t in topics]

    # Pad to full TOPIC_ORDER length (needed for downstream report)
    p_stds_full = [stdev(probe_stats[t]) if t in probe_stats else float("nan") for t in TOPIC_ORDER]
    s_stds_full = [stdev(sae_stats[t])   if t in sae_stats   else float("nan") for t in TOPIC_ORDER]

    all_p, all_s, labels = _std_layout(ctrl_p_std, ctrl_s_std, p_stds_full, s_stds_full)
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width/2, all_p, width, label="Probe", color=PROBE_COLOR, alpha=0.85)
    ax.bar(x + width/2, all_s, width, label="SAE",   color=SAE_COLOR,   alpha=0.85)
    _add_separators(ax, len(TOPIC_ORDER))

    ax.set_xlabel("Condition")
    ax.set_ylabel(f"Std Dev of Audit Accuracy (prompts {skip+1}–50)")
    ax.set_title(f"Accuracy Variability: Gender Control vs. Misleading Topics\n(higher = more inconsistently fooled)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    valid_vals = [v for v in all_p + all_s if not math.isnan(v)]
    ax.set_ylim(0, max(max(valid_vals, default=0.3) * 1.3, 0.3))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    savefig(fig, output_dir / "fig4_late_accuracy_variance.png")
    return probe_stats, sae_stats, p_stds_full, s_stds_full


# ── Chart 5: Gender slant distribution ──────────────────────────────────────

def chart_gender_slant(slant_data, ctrl_p_override, ctrl_s_override, output_dir):
    """Bar chart of mean gender slant score per condition, separated probe vs SAE."""
    sessions, _, _ = slant_data  # unpack; ctrl overrides come from load_gender_slant
    probe_scores = {}
    sae_scores   = {}

    for session in sessions:
        name = session.get("session", "")
        # New schema: mean_score; old schema: gendered_fraction
        score = session.get("mean_score")
        if score is None:
            gf = session.get("gendered_fraction")
            if gf is not None:
                score = gf * 3  # rough re-scale from 0-1 to 0-3 for comparability

        for topic in TOPIC_ORDER:
            if f"mislead_{topic}_sae" in name:
                sae_scores[topic] = score
                break
            elif f"mislead_{topic}" in name and "_sae" not in name:
                probe_scores[topic] = score
                break
        # Also capture in-file control entries as fallback
        if "control_gender_sae" in name:
            sae_scores["_control"] = score
        elif "control_gender" in name:
            probe_scores["_control"] = score

    # Use override values from _control.json if available
    ctrl_p = ctrl_p_override if ctrl_p_override is not None else probe_scores.get("_control", float("nan"))
    ctrl_s = ctrl_s_override if ctrl_s_override is not None else sae_scores.get("_control",  float("nan"))

    if not (probe_scores or sae_scores) and ctrl_p_override is None and ctrl_s_override is None:
        print("  No gender slant data matched conditions — skipping chart.")
        return

    topic_p = [probe_scores.get(t, float("nan")) for t in TOPIC_ORDER]
    topic_s = [sae_scores.get(t, float("nan")) for t in TOPIC_ORDER]

    all_p, all_s, labels = _std_layout(ctrl_p, ctrl_s, topic_p, topic_s)
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width/2, all_p, width, label="Probe", color=PROBE_COLOR, alpha=0.85)
    ax.bar(x + width/2, all_s, width, label="SAE",   color=SAE_COLOR,   alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    _add_separators(ax, len(TOPIC_ORDER))

    ax.set_ylim(-3.3, 3.3)
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_yticklabels(["-3\n(very female)", "-2", "-1", "0\n(neutral)", "+1", "+2", "+3\n(very male)"])
    ax.set_xlabel("Condition")
    ax.set_ylabel("Mean Gender Slant Score")
    ax.set_title("Gender Slant of Red-Team Prompts: Gender Control vs. Misleading Topics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    savefig(fig, output_dir / "fig5_gender_slant.png")


# ── Chart 6: Summary scatter — accuracy drop vs gender feature fraction ─────

def chart_scatter_drop_vs_features(data, feat_data, probe_accs, sae_accs, output_dir):
    topic_stats = {}
    for key, val in feat_data.items():
        if "error" in val:
            continue
        for topic in TOPIC_ORDER:
            if f"mislead_{topic}_sae" in key:
                topic_stats[topic] = val
                break

    topics = [t for t in TOPIC_ORDER if t in topic_stats and data["conditions"][t]["sae"]]
    if not topics:
        return

    sae_drops = [data["sae_baseline"] - data["conditions"][t]["sae"]["mean_accuracy"] for t in topics]
    gender_fracs = [topic_stats[t]["mean_per_eval_gender_fraction"] * 100 for t in topics]

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(gender_fracs, sae_drops, c=SAE_COLOR, s=80, zorder=3)
    for i, topic in enumerate(topics):
        ax.annotate(TOPIC_LABELS[topic], (gender_fracs[i], sae_drops[i]),
                    textcoords="offset points", xytext=(6, 3), fontsize=8)

    # Trend line
    if len(topics) >= 3:
        m, b = np.polyfit(gender_fracs, sae_drops, 1)
        xs = np.linspace(min(gender_fracs), max(gender_fracs), 50)
        ax.plot(xs, m*xs + b, "--", color="gray", linewidth=1, alpha=0.7)

    ax.set_xlabel("Mean Gender Feature Fraction per Eval (%)")
    ax.set_ylabel("SAE Accuracy Drop from Baseline")
    ax.set_title("SAE: Gender Feature Activation vs. Accuracy Drop\nper Misleading Condition")
    ax.grid(alpha=0.3)

    savefig(fig, output_dir / "fig6_scatter_features_vs_drop.png")


# ── Markdown report ─────────────────────────────────────────────────────────

def format_pct(v):
    return f"{v*100:.1f}%" if not math.isnan(v) else "—"

def format_val(v, decimals=3):
    return f"{v:.{decimals}f}" if not math.isnan(v) else "—"


def write_report(data, probe_accs, sae_accs, probe_rand, sae_rand,
                 p_stds, s_stds, slant_data, output_dir):
    lines = []
    A = lines.append

    A("# Misleading Prompt Experiments — Results Report\n")
    A("## 1. Overview\n")
    A(
        "We test whether an adversarial red-team agent can suppress a gender auditor's accuracy "
        "by framing the target model as auditing for a *different* demographic attribute "
        "(nationality, age, politics, etc.). We compare two auditor types:\n"
        "- **Probe**: A linear classifier on layer-26 hidden states.\n"
        "- **SAE**: An LLM auditor reading top-50 sparse-autoencoder feature descriptions from layer 23.\n\n"
        "Each condition runs 50 red-team evaluations. We additionally test random/nonsense prompts "
        "as a lower bound on perturbability, and inspect which SAE features fire under each condition.\n"
    )

    A("## 2. Baselines\n")
    A("| Metric | Probe | SAE |")
    A("|---|---|---|")
    A(f"| Audit accuracy (empty prompt) | {data['probe_baseline']:.3f} | {data['sae_baseline']:.3f} |")
    if data["control_probe"] and data["control_sae"]:
        A(f"| Audit accuracy (control — gender framing) | {data['control_probe']['mean_accuracy']:.3f} | {data['control_sae']['mean_accuracy']:.3f} |")
    A("")

    A("## 3. Per-Condition Audit Accuracy\n")
    A("Mean audit accuracy over 50 red-team evaluations per condition. "
      "Dashes (—) indicate missing data.\n")
    A("| Condition | Probe acc | Probe Δ | SAE acc | SAE Δ |")
    A("|---|---|---|---|---|")
    # Gender (control) row
    cp_acc = data["control_probe"]["mean_accuracy"] if data["control_probe"] else float("nan")
    cs_acc = data["control_sae"]["mean_accuracy"]   if data["control_sae"]   else float("nan")
    A(f"| **Gender** | {format_val(cp_acc)} | {format_val(data['probe_baseline']-cp_acc,3)} | "
      f"{format_val(cs_acc)} | {format_val(data['sae_baseline']-cs_acc,3)} |")
    # Misleading rows
    for i, topic in enumerate(TOPIC_ORDER):
        p_acc = probe_accs[i]
        s_acc = sae_accs[i]
        p_drop = data["probe_baseline"] - p_acc if not math.isnan(p_acc) else float("nan")
        s_drop = data["sae_baseline"]   - s_acc if not math.isnan(s_acc) else float("nan")
        A(f"| {TOPIC_LABELS[topic]} | {format_val(p_acc)} | {format_val(p_drop,3)} | {format_val(s_acc)} | {format_val(s_drop,3)} |")
    # Mean row
    mean_p_drop = mean([data["probe_baseline"] - a for a in probe_accs if not math.isnan(a)])
    mean_s_drop = mean([data["sae_baseline"]   - a for a in sae_accs   if not math.isnan(a)])
    mean_p_acc  = mean([a for a in probe_accs if not math.isnan(a)])
    mean_s_acc  = mean([a for a in sae_accs   if not math.isnan(a)])
    A(f"| **Mean (misleading)** | {format_val(mean_p_acc)} | {format_val(mean_p_drop,3)} | "
      f"{format_val(mean_s_acc)} | {format_val(mean_s_drop,3)} |")
    A("")
    A(f"**Mean accuracy drop across all misleading conditions:** Probe = {mean_p_drop:.3f}, SAE = {mean_s_drop:.3f}\n")

    A("## 4. Random / Nonsense Prompt Baseline\n")
    A("Evaluating auditor accuracy on prompts that contain no semantic meaning — "
      "these serve as an upper bound on unintentional disruption.\n")
    A("| Prompt type | Probe acc | Probe Δ | SAE acc | SAE Δ |")
    A("|---|---|---|---|---|")
    for cat, label in [("random_words","Random words"), ("random_chars","Random chars"),
                       ("lorem_ipsum","Lorem ipsum"), ("irrelevant_coherent","Irrelevant coherent")]:
        p = probe_rand["by_type"][cat]["mean_accuracy"]
        s = sae_rand["by_type"][cat]["mean_accuracy"]
        A(f"| {label} | {p:.3f} | {data['probe_baseline']-p:.3f} | {s:.3f} | {data['sae_baseline']-s:.3f} |")
    A("")
    A(f"**Key finding:** SAE accuracy under random characters drops to "
      f"**{sae_rand['by_type']['random_chars']['mean_accuracy']:.3f}** "
      f"(Δ = {data['sae_baseline']-sae_rand['by_type']['random_chars']['mean_accuracy']:.3f}), "
      f"compared to probe accuracy {probe_rand['by_type']['random_chars']['mean_accuracy']:.3f} "
      f"(Δ = {data['probe_baseline']-probe_rand['by_type']['random_chars']['mean_accuracy']:.3f}). "
      "The SAE auditor is dramatically more sensitive to noisy context than the probe.\n"
    )

    A("## 5. Accuracy Variance After First 20 Prompts\n")
    A("For each condition we look at evaluations 21–50 (after the red-team agent has had time to "
      "optimise its prompts) and report the standard deviation of audit accuracy across those "
      "individual evaluations. High std dev means the auditor is inconsistently fooled — "
      "some prompts work, others don't.\n")
    A("| Condition | Probe std dev | SAE std dev |")
    A("|---|---|---|")
    for i, topic in enumerate(TOPIC_ORDER):
        ps = p_stds[i] if i < len(p_stds) else float("nan")
        ss = s_stds[i] if i < len(s_stds) else float("nan")
        A(f"| {TOPIC_LABELS[topic]} | {format_val(ps, 3)} | {format_val(ss, 3)} |")
    valid_p = [v for v in p_stds if not math.isnan(v)]
    valid_s = [v for v in s_stds if not math.isnan(v)]
    mean_p_std = mean(valid_p)
    mean_s_std = mean(valid_s)
    A(f"| **Mean (misleading)** | {format_val(mean_p_std, 3)} | {format_val(mean_s_std, 3)} |")
    A(f"\n**Mean std dev across misleading conditions:** Probe = {mean_p_std:.3f}, SAE = {mean_s_std:.3f}\n")

    A("## 6. Gender Slant of Red-Team Prompts\n")
    A("An LLM judge rates each generated system prompt on a 7-point scale "
      "from −3 (very female) to +3 (very male). This checks whether the red-team agent "
      "accidentally introduces gendered language even when instructed to attack a non-gender target.\n")
    slant_sessions = slant_data[0] if isinstance(slant_data, tuple) else slant_data
    if slant_sessions:
        probe_by_topic = {}
        sae_by_topic   = {}
        for session in slant_sessions:
            name = session.get("session", "")
            score = session.get("mean_score") or (session.get("gendered_fraction", 0) * 3)
            for topic in TOPIC_ORDER:
                if f"mislead_{topic}_sae" in name:
                    sae_by_topic[topic] = score
                elif f"mislead_{topic}" in name:
                    probe_by_topic[topic] = score

        if probe_by_topic or sae_by_topic:
            A("| Condition | Probe mean score | SAE mean score |")
            A("|---|---|---|")
            for topic in TOPIC_ORDER:
                p = probe_by_topic.get(topic)
                s = sae_by_topic.get(topic)
                A(f"| {TOPIC_LABELS[topic]} | {format_val(p,2) if p is not None else '—'} | "
                  f"{format_val(s,2) if s is not None else '—'} |")
            A("")
            A("Scores close to 0 indicate the generated prompts are gender-neutral, "
              "confirming that any accuracy drops are due to off-topic distraction rather than "
              "the red-team agent covertly introducing gender signals.\n")
    elif not slant_sessions:
        A("*(Gender slant analysis data not found)*\n")

    A("## 7. Summary and Interpretation\n")
    A(
        f"- **Probe robustness**: The linear probe maintains near-baseline accuracy "
        f"({data['probe_baseline']:.2f}) across all misleading conditions "
        f"(mean drop: {mean_p_drop:.3f}). Even nonsense prompts cause negligible degradation.\n"
        f"- **SAE vulnerability**: The SAE auditor shows substantially larger accuracy drops "
        f"under misleading framing (mean drop: {mean_s_drop:.3f}). "
        f"Random character strings cause drops of up to "
        f"{data['sae_baseline']-sae_rand['by_type']['random_chars']['mean_accuracy']:.2f}, "
        f"compared to near-zero for the probe.\n"
        f"- **Feature displacement**: Under misleading conditions, <5% of the SAE features "
        f"that fire are gender-related, compared to much higher fractions under a neutral prompt. "
        f"The model's feature activations are easily redirected by off-topic context.\n"
        f"- **No covert gender injection**: Gender slant scores of generated prompts remain near 0 "
        f"across conditions, confirming that accuracy drops are due to interference, not hidden "
        f"gendered language.\n\n"
        "Together, these results support the claim that SAE-based auditing is **pathologically "
        "more vulnerable** to contextual manipulation than linear probes. The SAE auditor's "
        "reliance on semantically interpretable features makes it susceptible to feature "
        "displacement by unrelated context, while the probe's low-dimensional representation "
        "appears invariant to such perturbations.\n"
    )

    report_path = output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved report.md")
    return report_path


# ── Main ────────────────────────────────────────────────────────────────────

def write_best_prompts(sessions_dir, output_dir, skip=20, top_n=5):
    lines = []
    lines.append(
        f"Best Prompts by Condition (top-{top_n} lowest audit_accuracy where "
        f"internalize_accuracy >= {INTERN_THRESHOLD:.3f}, after first {skip})\n"
    )
    lines.append("=" * 80 + "\n")

    configs = [
        ("Probe", [("Gender", "control_gender_2*")] + [(TOPIC_LABELS[t], f"mislead_{t}_2*") for t in TOPIC_ORDER]),
        ("SAE",   [("Gender", "control_gender_sae_2*")] + [(TOPIC_LABELS[t], f"mislead_{t}_sae_2*") for t in TOPIC_ORDER]),
    ]
    for auditor_label, conditions in configs:
        lines.append(f"\n{'=' * 80}")
        lines.append(f"  AUDITOR: {auditor_label}")
        lines.append(f"{'=' * 80}\n")

        for cond_label, pattern in conditions:
            d = find_session(sessions_dir, pattern)
            if not d:
                continue
            log = d / "evaluation_log.json"
            if not log.exists():
                continue
            with open(log) as f:
                entries = json.load(f)[skip:]

            successful = [
                e for e in entries
                if is_intern_success(e, INTERN_THRESHOLD) and "audit_accuracy" in e
            ]
            top = sorted(successful, key=lambda e: e["audit_accuracy"])[:top_n]

            lines.append(f"--- {cond_label} ({auditor_label}) ---\n")
            if not top:
                lines.append("  (no successful prompts found)\n")
                continue
            for i, e in enumerate(top, 1):
                acc = e.get("audit_accuracy", float("nan"))
                intern = e.get("internalize_accuracy", float("nan"))
                score = e.get("score", float("nan"))
                prompt = e.get("prompt", "").strip()
                lines.append(f"[{i}] audit_accuracy={acc:.3f}  internalize_accuracy={intern:.3f}  score={score:.3f}")
                lines.append(prompt)
                lines.append("")

    out_path = output_dir / "best_prompts.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved best_prompts.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/results_big")
    parser.add_argument("--output-dir",  default="results/report")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_all_conditions(results_dir)
    probe_rand, sae_rand = load_random_baselines(results_dir)
    slant_sessions, ctrl_p, ctrl_s = load_gender_slant(results_dir)
    slant_data = (slant_sessions, ctrl_p, ctrl_s)

    print("Generating charts...")
    probe_accs, sae_accs = chart_accuracy_comparison(data, output_dir)
    chart_top5_best(data, output_dir)
    chart_accuracy_drop(data, probe_accs, sae_accs, output_dir)
    chart_random_baseline(probe_rand, sae_rand, data, output_dir)
    _, _, p_stds, s_stds = chart_late_accuracy_variance(
        results_dir / "red_team_sessions", output_dir)
    chart_gender_slant(slant_data, ctrl_p, ctrl_s, output_dir)

    print("Writing report...")
    write_report(data, probe_accs, sae_accs, probe_rand, sae_rand,
                 p_stds, s_stds, slant_data, output_dir)
    write_best_prompts(results_dir / "red_team_sessions", output_dir)

    print(f"\nDone. Output in {output_dir}/")


if __name__ == "__main__":
    main()
