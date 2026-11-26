"""Stage the committed demo figure and summarize its associated run archive."""

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG = "active-inference/logs/active_inference_demo_20250129_165711.json"
PLOT = (
    "active-inference/visualizations/run_20250129_165717/belief_evolution.png"
)
FIGURE = "archived-belief-evolution.png"
ALT = "Line plots of Low, Medium, and High belief probabilities over the archived run."


def main() -> None:
    rows = [
        json.loads(line) for line in (ROOT / LOG).read_text(encoding="utf-8").splitlines()
    ]
    output = ROOT / "output"
    figures = output / "figures"
    reports = output / "reports"
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / PLOT, figures / FIGURE)

    registry = {
        "schema_version": 1,
        "figures": [
            {
                "label": "fig:archived-belief-evolution",
                "filename": FIGURE,
                "alt_text": ALT,
                "caption": "Belief evolution in the archived January 2025 simulation.",
                "generated_by": "scripts/stage_manuscript_evidence.py",
            }
        ],
    }
    (figures / "figure_registry.json").write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )

    evidence = {
        "log_source": LOG,
        "plot_source": PLOT,
        "plot_sha256": hashlib.sha256((ROOT / PLOT).read_bytes()).hexdigest(),
        "agent_steps": sum(row.get("event") == "agent_step" for row in rows),
        "environment_steps": sum(row.get("event") == "environment_step" for row in rows),
    }
    (reports / "archived_demo.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
