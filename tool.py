import logging
from typing import Dict, Any


from tool_component import ToolComponent
import models


logger = logging.getLogger(__name__)


class MathTool(ToolComponent):
    """Tool to evaluate mathematical expressions safely."""


    def __init__(self):
        self.tool_parameters: Dict[str, models.ToolParameterModel] = {}
        self._load_parameters()

        self.tool_info = models.ToolInfoModel(
            name="tool.math",
            purpose="Calculates results of a mathematical expression.",
            description="Calculates results of a mathematical expression.  Any type of valid python math expression works.",
            module="tools.brodagroupsoftware.math.tool"
        )


    def info(self) -> models.ToolInfoModel:
        return self.tool_info


    def add_parameter(self, parameter: models.ToolParameterModel):
        logger.debug(f"Adding parameter:{parameter}")
        self.tool_parameters[parameter.name] = parameter


    def parameters(self) -> Dict[str, models.ToolParameterModel]:
        return self.tool_parameters


    def _load_parameters(self):
        logger.debug("Loading parameters")
        expression = models.ToolParameterModel(
            name="expression",
            type="string",
            description="The mathematical expression to evaluate, e.g., '3 * (2 + 4) / 5'."
        )
        self.tool_parameters[expression.name] = expression


    def return_value(self) -> models.ToolReturnValueModel:
        tool_return_value = models.ToolReturnValueModel(
            type="float",
            description="The result of evaluating the mathematical expression."
        )
        return tool_return_value


    async def execute(self, iid: str, parameters: Dict[str, Any]) -> Any:

        logger.info(f"Math tool iid:{iid} math tool with parameters: {parameters}")

        # Extract the expression from parameters
        expression = parameters.get("expression")
        if not expression:
            raise ValueError("No expression provided for evaluation")

        # Safely evaluate the expression with limited built-in functions
        try:
            # Define a safe evaluation environment
            import math
            safe_globals = {"__builtins__": None, "math": math}
            # Evaluate the expression within the safe environment
            result = eval(expression, safe_globals, {})
        except Exception as e:
            logger.error(f"Failed to evaluate expression '{expression}': {e}", exc_info=True)
            raise ValueError(f"Error evaluating expression: {e}")
        logger.debug(f"Result of the expression '{expression}': {result}")

        output = {
            "status": "COMPLETE",
            "value": result
        }
        msg = (
            "\n *********"
            "\n *"
            "\n * " + f"Tool: {self.tool_info.name}"
            "\n * " + f"Math tool iid:{iid} math tool with parameters: {parameters}"
            "\n * " + f"Output: {str(output)[:100] + '...' + f' (total length: {len(str(output))})' if len(str(output)) > 100 else str(output)}"
            "\n *"
            "\n *********"
        )
        logger.info(msg)
        return output

