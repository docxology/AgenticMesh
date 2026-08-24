# AgenticMesh

Two layers: a small, explicit contract for tools that an agent can call, and a worked Active Inference agent built on top of it.

## The tool contract

`tool_component.py` defines `ToolComponent`, the abstract base every tool implements:

| Method | Returns | Purpose |
|---|---|---|
| `info()` | `ToolInfoModel` | Name, purpose, description, module |
| `add_parameter(parameter)` | — | Register a `ToolParameterModel` |
| `parameters()` | `Dict[str, ToolParameterModel]` | The tool's declared parameters |
| `return_value()` | `ToolReturnValueModel` | The shape of what the tool returns |
| `async execute(iid, parameters)` | `Any` | Run the tool for interaction `iid` |

`models.py` holds the three dataclasses that describe a tool to its caller:

```python
@dataclass
class ToolParameterModel:
    name: str
    type: str
    description: str
    default: Optional[Any] = None
    required: bool = True

@dataclass
class ToolInfoModel:
    name: str
    purpose: str
    description: str
    module: str

@dataclass
class ToolReturnValueModel:
    type: str
    description: str
    properties: Optional[Dict[str, Any]] = None
```

Because parameters and return values are declared rather than implied, an agent can inspect a tool before deciding to call it.

## MathTool

`tool.py` is the smallest complete implementation of that contract: evaluate a mathematical expression.

It takes one required parameter, `expression` (for example `3 * (2 + 4) / 5`), and evaluates it under deliberately constrained globals — `__builtins__` set to `None`, only the `math` module whitelisted, no locals passed in. Evaluation failures are caught and logged rather than raised. The result is returned as a float inside a `COMPLETE` status dictionary.

## Active Inference agent

`active-inference/` is the substantial component: a discrete-POMDP Active Inference agent following the free energy principle.

The generative model uses the standard matrices — `A` likelihood, `B` transition dynamics, `C` preference priors, `D` initial beliefs, `E` action habits — over discrete states, observations, and actions. The shipped configuration uses Low/Medium/High states with Decrease/Stay/Increase actions.

| File | Role |
|---|---|
| `tool_active_inference.py` | The agent: belief updating by variational free energy, action selection, habit learning, state management |
| `tool_actinf_visualization.py` | Six diagnostics — belief evolution, policy development, free energy analysis, state-action transitions, belief phase space, and a summary dashboard |
| `tool_environment.py` | The environment the agent acts in |
| `tool_simulation_active_inference.py` | Drives a simulation run end to end |
| `agent-active-inference.yaml` | Model dimensions and matrices |
| `active_inference_utils.py` | Shared helpers |

Each run writes a JSON log and a `visualizations/run_<TIMESTAMP>/` directory of PNG figures; example output from a January 2025 run is committed.

### Running it

```bash
pip install -r active-inference/requirements.txt
```

Requires NumPy, SciPy, Pandas, Matplotlib, Seaborn, and NetworkX. See `active-inference/ActiveInference_README.md` for the agent's parameters and how to invoke a run.

## Status

Early-stage. The tool contract is deliberately small and stable; the Active Inference component is the most developed part of the repository.
