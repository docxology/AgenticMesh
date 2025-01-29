from abc import ABC, abstractmethod
from typing import Dict, Any


import models

class ToolComponent(ABC):
    """Base class for all tools, providing a consistent interface for execution."""

    @abstractmethod
    def info(self) -> models.ToolInfoModel:
        """
        Return tool information

        Returns:
            models.ToolInfoModel: The tool information.
        """
        pass

    @abstractmethod
    def add_parameter(self, parameter: models.ToolParameterModel):
        """
        Return tool parameters

        Returns:
            models.ToolParameterModel: The tool parameters.
        """
        pass

    @abstractmethod
    def parameters(self) -> Dict[str, models.ToolParameterModel]:
        """
        Return tool parameters

        Returns:
            models.ToolParameterModel: The tool parameters.
        """
        pass


    @abstractmethod
    def return_value(self) -> models.ToolReturnValueModel:
        """
        Return tool return value

        Returns:
            models.ToolReturnValueModel: The tool return value.
        """
        pass


    @abstractmethod
    async def execute(self, iid: str, parameters: Dict[str, models.ToolParameterModel]) -> Any:
        """
        Execute the tool's functionality with the provided parameters.

        Args:
            iid (str): Interaction ID for this request
            parameters (Dict[str, Any]): Dictionary of parameters required for execution.

        Returns:
            Any: The result of the tool's execution.
        """
        pass

