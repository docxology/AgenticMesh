# Local verification and manuscript build

From the repository root, use the [installed environment](active-inference.md#setup-and-usage).
The following checks use a fixed arithmetic literal, one known observation in
the synthetic model, and the committed archive.

```bash
.venv/bin/python -m compileall -q models.py tool.py tool_component.py active-inference scripts
.venv/bin/python scripts/stage_manuscript_evidence.py
PYTHONPATH=active-inference .venv/bin/python - <<'PY'
import asyncio

import numpy as np

from tool import MathTool
from tool_active_inference import ActiveInferenceTool
from tool_environment import EnvironmentTool


async def check():
    assert await MathTool().execute("smoke", {"expression": "3 * (2 + 4)"}) == {
        "status": "COMPLETE",
        "value": 18,
    }

    env = EnvironmentTool()
    env.observation_probs[env.current_state] = [0, 1, 0]
    observation = await env.execute("fixture", {})
    assert observation["observation_name"] == "Medium"

    result = await ActiveInferenceTool().execute(
        "fixture", {"observation": observation["observation"]}
    )
    np.testing.assert_allclose(result["belief_posterior"], [1 / 6, 2 / 3, 1 / 6])
    np.testing.assert_allclose(sum(result["policy_posterior"]), 1)
    assert result["selected_action_name"] in observation["valid_actions"]


asyncio.run(check())
PY
```

The controlled fixture changes the environment's observation row for that call
and allows any sampled action from the declared set. It checks the computed
posterior and policy normalization. The evidence staging script copies the
historical plot byte for byte and writes its source path, hash, and log event
counts to `output/reports/archived_demo.json`.

## Pinned Template render

Set up [Template's pinned runtime](https://github.com/docxology/template/blob/87704feae0a4490ef7dad9c931d268e8c30273c2/docs/RUN_GUIDE.md).
After staging the evidence above, run from the separate Template root:

```bash
git switch --detach 87704feae0a4490ef7dad9c931d268e8c30273c2
mkdir -p projects/working
ln -s /absolute/path/to/AgenticMesh projects/working/AgenticMesh
uv run --locked python scripts/pipeline/stage_03_render.py --project working/AgenticMesh
uv run --locked python scripts/pipeline/stage_04_validate.py --project working/AgenticMesh
```

The project slot may also hold a detached AgenticMesh worktree. Inspect
`output/pdf/AgenticMesh_combined.pdf`, `output/web/index.html`, and the
validation and provenance reports under `output/reports/` in that checkout.
Verify the ordered sections, citations, and the archived belief plot.
