# Active Inference model

The example pairs an agent with a simulated environment. Both use Low, Medium,
and High states and observations. The environment accepts the action names
Decrease, Stay, and Increase.

## Setup and usage

From the repository root, create an environment and install the requirements.
The verification used Python 3.14. The [requirements file](../active-inference/requirements.txt)
declares lower bounds.

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r active-inference/requirements.txt
cd active-inference
../.venv/bin/python tool_simulation_active_inference.py
../.venv/bin/python tool_actinf_visualization.py
```

Run the driver from `active-inference/` so its working-directory-relative
`logs/` destination matches the visualizer's script-relative search. The driver
writes a timestamped newline-delimited JSON log. The visualizer selects the
most recently modified demo log and writes PNGs under
`visualizations/run_<TIMESTAMP>/`. These include belief evolution and phase
space, policies, energy components, transition counts, exploration and action
diagnostics, and a summary dashboard. Keep the committed January 2025 outputs
when generating fresh examples.

## Tool calls and configuration

| Module | Behavior |
| --- | --- |
| [Agent](../active-inference/tool_active_inference.py) | Accepts an observation index, returns the belief update, sampled action, energy arrays, and history |
| [Environment](../active-inference/tool_environment.py) | With no action, samples an observation from the current state. With an action, samples a transition and then an observation |
| [Simulation driver](../active-inference/tool_simulation_active_inference.py) | Alternates calls and records structured step events |
| [Visualizer](../active-inference/tool_actinf_visualization.py) | Reads recorded events and creates diagnostic plots |
| [Utilities](../active-inference/active_inference_utils.py) | Matrix validation, normalization, and a standalone YAML loader |

The default constructors supply the model used by the driver. The
[example YAML](../active-inference/agent-active-inference.yaml) documents model
and plotting settings, and the utilities can parse it. The driver and
visualizer have their own defaults and do not read that file automatically.

The agent uses likelihood matrix A, transition matrix B, observation preference
scores C, initial belief vector D, and habit prior E. The preference cost
computed by the code is cross-entropy between predicted observations and
the preference distribution. Histories record beliefs, actions, policy priors,
and diagnostic energy values. These describe behavior in the toy environment.

Use the [local verification](verification.md) for one controlled observation
and the [manuscript](manuscript/README.md) for the interpretation and references.
