# Tool contract

`ToolComponent` requires methods for tool information, parameter registration,
parameter lookup, return-value metadata, and the asynchronous entry point
`execute(iid, parameters)`. The interaction identifier travels into the tool
and its logs. `ToolInfoModel`, `ToolParameterModel`, and
`ToolReturnValueModel` expose declarations to the caller. The declarations
describe expected values. Tools apply their own validation and behavior.

`MathTool` takes an expression and returns the result in a dictionary with a
completion status. The arithmetic check supplies a fixed literal and asserts
the returned value. The example uses Python evaluation and accepts trusted
arithmetic only. The interface itself provides neither input isolation nor an
authorization mechanism.
