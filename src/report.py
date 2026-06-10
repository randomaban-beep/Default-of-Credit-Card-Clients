"""Auto-generate README results section from metrics.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import METRICS_FILE, ROOT

START_MARKER = "<!-- RESULTS:START -->"
END_MARKER = "<!-- RESULTS:END -->"


def generate_results_block(metrics: dict) -> str:
    """Build the markdown results block from metrics data."""
    bl = metrics.get("baseline", {})
    lg = metrics.get("lgbm", {})
    cal = metrics.get("calibration", {})
    biz = metrics.get("business", {})
    top = metrics.get("top_drivers", [])

    lines = [START_MARKER, ""]

    # Model comparison table
    lines.append("## Results")
    lines.append("")
    lines.append("### Model Comparison")
    lines.append("")
    lines.append("| Metric | Baseline (LR) | LightGBM |")
    lines.append("|--------|:-------------:|:--------:|")
    for metric in ["roc_auc", "pr_auc", "ks", "gini", "brier"]:
        bl_val = f"{bl.get(metric, 'N/A'):.4f}" if isinstance(bl.get(metric), float) else bl.get(metric, 'N/A')
        lg_val = f"{lg.get(metric, 'N/A'):.4f}" if isinstance(lg.get(metric), float) else lg.get(metric, 'N/A')
        lines.append(f"| {metric.upper()} | {bl_val} | {lg_val} |")
    lines.append("")

    # Calibration
    lines.append("### Probability Calibration")
    lines.append("")
    lines.append(f"| | Brier Score |")
    lines.append("|---|:---:|")
    lines.append(f"| Before calibration | {cal.get('brier_before', 'N/A'):.4f} |")
    lines.append(f"| After calibration ({cal.get('method', 'N/A')}) | {cal.get('brier_after', 'N/A'):.4f} |")
    if not cal.get("improved", True):
        lines.append("")
        lines.append("> Calibration did not improve Brier score; original model probabilities retained.")
    lines.append("")

    # Business metrics
    lines.append("### Cost-Based Decision Threshold")
    lines.append("")
    lines.append(f"- **Optimal threshold:** {biz.get('optimal_threshold', 'N/A')} (selected on out-of-fold training predictions)")
    lines.append(f"- **Expected loss @ 0.5:** {biz.get('loss_at_0.5', 'N/A'):.4f}")
    lines.append(f"- **Expected loss @ optimal:** {biz.get('loss_at_optimal', 'N/A'):.4f}")
    lines.append(f"- **Loss reduction vs 0.5:** {biz.get('loss_reduction_pct', 'N/A'):.1f}%")
    lines.append(f"- **Top-decile capture:** {biz.get('top_decile_capture', 0)*100:.1f}% of all defaults")
    lines.append(f"- **Top-decile lift:** {biz.get('top_decile_lift', 'N/A'):.2f}x")
    lines.append("")

    # Top SHAP drivers
    lines.append("### Top 5 SHAP Drivers")
    lines.append("")
    lines.append("| Rank | Feature | Mean Abs SHAP |")
    lines.append("|:----:|---------|:-------------:|")
    for i, d in enumerate(top, 1):
        lines.append(f"| {i} | {d.get('feature', 'N/A')} | {d.get('mean_abs_shap', 0):.4f} |")
    lines.append("")

    lines.append(END_MARKER)
    return "\n".join(lines)


def update_readme() -> None:
    """Rewrite the RESULTS block in README.md from metrics.json.

    Running this twice leaves README.md unchanged.
    """
    readme_path = ROOT / "README.md"
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    new_block = generate_results_block(metrics)

    content = readme_path.read_text(encoding="utf-8")
    start_idx = content.find(START_MARKER)
    end_idx = content.find(END_MARKER)

    if start_idx == -1 or end_idx == -1:
        raise ValueError(f"Could not find {START_MARKER} or {END_MARKER} in README.md")

    new_content = content[:start_idx] + new_block + content[end_idx + len(END_MARKER):]
    readme_path.write_text(new_content, encoding="utf-8")
    print("README.md updated.")


if __name__ == "__main__":
    update_readme()
