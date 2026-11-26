# AGENTS.md — AgenticMesh

Guidance for agents working in this repository.

## What this repository is

Two layers, deliberately small:

1. **A tool contract.** `tool_component.py` defines `ToolComponent`, the abstract
   base every tool implements (`info()`, `add_parameter()`, `parameters()`,
   `return_value()`, `async execute(iid, parameters)`). `models.py` holds the
   three dataclasses that describe a tool to a caller: `ToolInfoModel`,
   `ToolParameterModel`, `ToolReturnValueModel`.
2. **A worked agent on top of it.** `tool.py` implements `MathTool`, a trusted-input
   arithmetic example. `active-inference/` holds a discrete-POMDP Active
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

See the [documentation index](docs/README.md) for the two example workflows
and [local verification](docs/verification.md) for reproducible checks.

## Conventions

- New tools implement `ToolComponent` and nothing else. Do not invent a second
  dispatch mechanism; `async execute(iid, parameters)` is the entry point.
- The existing `MathTool` evaluates Python expressions. Only pass trusted
  arithmetic to it. The documented [backlog](docs/backlog.md) tracks a
  narrower evaluator.
- The committed `logs/` and `visualizations/` trees are sample output, not
  fixtures. Regenerating them is expected; deleting them removes the only
  example of what a run produces.
- `active-inference/requirements.txt` pins lower bounds only (`numpy>=`,
  `scipy>=`, …). Leave the floors alone unless you are testing a specific
  incompatibility.

## Verification

The [verification guide](docs/verification.md) runs syntax checks, fixed
arithmetic, one synthetic agent observation, and a pinned manuscript render.
The archived traces describe a single example run.
