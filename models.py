from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class ToolParameterModel:
    """Model for tool parameters."""
    name: str
    type: str
    description: str
    default: Optional[Any] = None
    required: bool = True

@dataclass
class ToolInfoModel:
    """Model for tool information."""
    name: str
    purpose: str
    description: str
    module: str

@dataclass
class ToolReturnValueModel:
    """Model for tool return values."""
    type: str
    description: str
    properties: Optional[Dict[str, Any]] = None 