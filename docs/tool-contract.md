# Tool contract

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

## Arithmetic example

`MathTool` accepts a required `expression` parameter and returns its evaluated
value with a `COMPLETE` status. Errors raise `ValueError`. Python determines
the result type, so integer inputs can produce an integer. The implementation
uses Python evaluation and is suitable only for fixed or otherwise trusted
arithmetic. Tool metadata does not provide an execution security boundary.

Run this example from the repository root. It uses the standard library.

```python
import asyncio
from tool import MathTool

tool = MathTool()
assert tool.parameters()["expression"].required
assert asyncio.run(tool.execute("example", {"expression": "3 * (2 + 4)"})) == {
    "status": "COMPLETE",
    "value": 18,
}
```
