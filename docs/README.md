# docs/ — AgenticMesh

Human entry point for repository documentation. Facts below come from the repo
root `README.md`, `active-inference/ActiveInference_README.md`, and source
inspection.

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

```bash
pip install -r active-inference/requirements.txt
cd active-inference
python3 tool_simulation_active_inference.py
python3 tool_actinf_visualization.py
```

No dedicated test suite, packaging metadata, or CI configuration is committed. Use `python3 -m compileall . active-inference` as the repository syntax smoke check.

## More documentation

- `active-inference/ActiveInference_README.md` — the Active Inference agent layer
- Root `README.md` — the tool-contract layer, with method tables
