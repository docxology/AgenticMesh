# docs/ — AgenticMesh

Human entry point for repository documentation. Facts below come from the repo
root `README.md`, `active-inference/ActiveInference_README.md`, and source
inspection (2026-08-29).

## What this repo is

AgenticMesh has two layers: (1) a small, explicit contract for tools an agent
can call — `tool_component.py` (`ToolComponent` abstract base) and `models.py`
(`ToolInfoModel`, `ToolParameterModel`, `ToolReturnValueModel`) — and (2) a
worked Active Inference agent built on top of that contract in
`active-inference/`.

## Directory map (top level)

| Path | Contents |
|---|---|
| `tool_component.py` | `ToolComponent` abstract base class (the tool contract) |
| `models.py` | Tool-description dataclasses |
| `tool.py` | Worked example tool (`MathTool`) |
| `active-inference/` | Active Inference agent implementation + visualization tools |
| `LICENSE` | License file |
| `docs/` | This documentation directory |

## How to run / test

Not documented in repo — needs owner input. No pyproject.toml, Makefile,
package.json, or test suite was found. The agent subproject's dependencies are
listed in `active-inference/requirements.txt` (numpy, scipy, pandas,
python-dateutil). The subproject README (`active-inference/ActiveInference_README.md`)
describes the agent components (`tool_active_inference.py`,
`tool_actinf_visualization.py`, `tool_environment.py`,
`tool_simulation_active_inference.py`) but no entry-point command.

## More documentation

- `active-inference/ActiveInference_README.md` — the Active Inference agent layer
- Root `README.md` — the tool-contract layer, with method tables
