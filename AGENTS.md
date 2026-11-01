# AGENTS.md — AgenticMesh

Guidance for agents working in this repository.

## What this repository is

Two layers, deliberately small:

1. **A tool contract.** `tool_component.py` defines `ToolComponent`, the abstract
   base every tool implements (`info()`, `add_parameter()`, `parameters()`,
   `return_value()`, `async execute(iid, parameters)`). `models.py` holds the
   three dataclasses that describe a tool to a caller: `ToolInfoModel`,
   `ToolParameterModel`, `ToolReturnValueModel`.
2. **A worked agent on top of it.** `tool.py` implements `MathTool`, a sandboxed
   expression evaluator. `active-inference/` holds a discrete-POMDP Active
   Inference agent (A/B/C/D/E matrices, variational free energy, habit learning)
   exposed as tools, with matplotlib diagnostics.

There is no packaging metadata and no installer. Modules are imported from the
repository root, so run and import from here, not from a parent directory.

## Layout

| Path | Role |
|---|---|
| `tool_component.py` | `ToolComponent` abstract base — the contract |
| `models.py` | `ToolInfoModel`, `ToolParameterModel`, `ToolReturnValueModel` |
| `tool.py` | `MathTool` reference implementation |
| `active-inference/tool_active_inference.py` | Active Inference tool |
| `active-inference/tool_environment.py` | Environment tool |
| `active-inference/tool_simulation_active_inference.py` | End-to-end simulation driver |
| `active-inference/tool_actinf_visualization.py` | Figure generation |
| `active-inference/active_inference_utils.py` | Shared math and matrix helpers |
| `active-inference/agent-active-inference.yaml` | Agent configuration |
| `active-inference/logs/`, `active-inference/visualizations/` | Committed sample output from a January 2025 run |
| `docs/` | Supporting documentation |

## Running

```bash
pip install -r active-inference/requirements.txt
python3 active-inference/tool_simulation_active_inference.py
python3 active-inference/tool_actinf_visualization.py
```

Each run writes a JSON log and a `visualizations/run_<TIMESTAMP>/` directory of
PNG figures.

## Conventions

- New tools implement `ToolComponent` and nothing else. Do not invent a second
  dispatch mechanism; `async execute(iid, parameters)` is the entry point.
- `MathTool` evaluates expressions. If you extend it, keep the evaluation
  sandboxed — do not widen it to arbitrary builtins or attribute access.
- The committed `logs/` and `visualizations/` trees are sample output, not
  fixtures. Regenerating them is expected; deleting them removes the only
  example of what a run produces.
- `active-inference/requirements.txt` pins lower bounds only (`numpy>=`,
  `scipy>=`, …). Leave the floors alone unless you are testing a specific
  incompatibility.

## Verification

No CI and no tests. Verify by running the simulation driver and confirming that
a new `visualizations/run_<TIMESTAMP>/` directory appears with the expected
figures, then report which command you ran. Do not describe the POMDP agent as
validated beyond what you actually executed.
