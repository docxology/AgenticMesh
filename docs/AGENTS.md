# docs/ — Agent Notes (AgenticMesh)

## Layout

- `docs/` contains entry-point docs, maintainer notes, and manuscript status.
- Root files `tool_component.py`, `models.py`, `tool.py` are the tool-contract
  layer; `active-inference/` is the agent layer.

## Key modules

- `tool_component.py` — `ToolComponent`: `info()`, `add_parameter()`,
  `parameters()`, `return_value()`, `async execute(iid, parameters)`.
- `models.py` — `ToolParameterModel(name, type, description, default=None)`,
  plus `ToolInfoModel` and `ToolReturnValueModel` (see root `README.md`).
- `active-inference/tool_active_inference.py` — belief updating, action
  selection, habits (per the subproject README).
- `active-inference/tool_actinf_visualization.py` — belief/policy visualization suite.

## Conventions observed

- Tools self-describe via the `ToolComponent` contract; new tools should
  implement it and register `ToolParameterModel`s, following `tool.py::MathTool`.
- Imports in `tool.py` are flat (`from tool_component import ToolComponent`),
  not package-relative; there is no packaging metadata.

## How docs here are maintained

Plain Markdown. Keep the contract tables in the root `README.md` in sync with
`tool_component.py`; this directory holds entry-point docs and status notes. No
docs CI gate is configured.
