import logging
import sys
import os
import numpy as np
import json
from typing import Dict, Any, Optional
from datetime import datetime

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tool_component import ToolComponent
import models

logger = logging.getLogger(__name__)

class EnvironmentTool(ToolComponent):
    """
    Simple environment tool that generates observations from three states (Low/Medium/High)
    and responds to three possible actions (Decrease/Stay/Increase).
    
    The environment follows a simple transition dynamics where:
    - Decrease action tends to lower the state
    - Stay action tends to maintain the state
    - Increase action tends to raise the state
    
    States are partially observable with some noise in the observations.
    """
    
    def __init__(self):
        # Initialize environment
        self.states = ["Low", "Medium", "High"]
        self.actions = ["Decrease", "Stay", "Increase"]
        self.current_state = 1  # Start in Medium state
        
        self._tool_info = models.ToolInfoModel(
            name="tool.environment",
            purpose="Generates observations from a three-state environment",
            description="Simulates a simple environment with Low/Medium/High states and responds to Decrease/Stay/Increase actions",
            module="tools.brodagroupsoftware.active_inference.tool_environment"
        )
        
        self._tool_parameters = {}
        self._load_parameters()
        
        # Define transition probabilities for each action
        # Format: [P(next_state | current_state, action)]
        self.transition_probs = {
            "Decrease": np.array([
                [0.8, 0.2, 0.0],  # From Low
                [0.7, 0.3, 0.0],  # From Medium
                [0.1, 0.8, 0.1]   # From High
            ]),
            "Stay": np.array([
                [0.7, 0.3, 0.0],  # From Low
                [0.2, 0.6, 0.2],  # From Medium
                [0.0, 0.3, 0.7]   # From High
            ]),
            "Increase": np.array([
                [0.1, 0.8, 0.1],  # From Low
                [0.0, 0.3, 0.7],  # From Medium
                [0.0, 0.2, 0.8]   # From High
            ])
        }
        
        # Define observation probabilities
        # Format: P(observation | state)
        self.observation_probs = np.array([
            [0.8, 0.2, 0.0],  # Low state
            [0.1, 0.8, 0.1],  # Medium state
            [0.0, 0.2, 0.8]   # High state
        ])

    def info(self) -> models.ToolInfoModel:
        """Return tool information."""
        return self._tool_info

    def add_parameter(self, parameter: models.ToolParameterModel):
        """Add a parameter to the tool."""
        self._tool_parameters[parameter.name] = parameter

    def parameters(self) -> Dict[str, models.ToolParameterModel]:
        """Return tool parameters."""
        return self._tool_parameters

    def _load_parameters(self):
        """Load tool parameters."""
        action = models.ToolParameterModel(
            name="action",
            type="string",
            description="Action to take in environment (Decrease/Stay/Increase)",
            required=False  # Not required for initial observation
        )
        self.add_parameter(action)

    def _update_state(self, action: Optional[str] = None):
        """Update environment state based on action."""
        if action is not None:
            if action not in self.actions:
                raise ValueError(f"Invalid action: {action}. Must be one of {self.actions}")
            
            # Get transition probabilities for current state and action
            trans_probs = self.transition_probs[action][self.current_state]
            
            # Sample next state
            self.current_state = np.random.choice(len(self.states), p=trans_probs)
            
        # Generate observation
        obs_probs = self.observation_probs[self.current_state]
        observation = np.random.choice(len(self.states), p=obs_probs)
        
        return observation

    def return_value(self) -> models.ToolReturnValueModel:
        """Define return value structure."""
        return models.ToolReturnValueModel(
            type="dict",
            description="Returns current observation and environment info",
            properties={
                "observation": {"type": "integer", "description": "Observation index"},
                "observation_name": {"type": "string", "description": "Name of observation"},
                "valid_actions": {"type": "array", "description": "List of valid actions"}
            }
        )

    async def execute(self, iid: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one step in the environment."""
        try:
            # Get action if provided
            action = parameters.get("action", None)
            
            # Update state and get observation
            observation = self._update_state(action)
            
            output = {
                "observation": int(observation),
                "observation_name": self.states[observation],
                "valid_actions": self.actions
            }
            
            # Log in JSON format
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "component": "environment",
                "iid": iid,
                "action": action,
                "observation": self.states[observation],
                "current_state": self.states[self.current_state]
            }
            logger.info(json.dumps(log_data))
            
            return output
            
        except Exception as e:
            error_data = {
                "timestamp": datetime.now().isoformat(),
                "level": "ERROR",
                "component": "environment",
                "iid": iid,
                "error": str(e),
                "current_state": self.states[self.current_state] if hasattr(self, 'current_state') else None
            }
            logger.error(json.dumps(error_data))
            raise

if __name__ == "__main__":
    # Simple test of environment
    env = EnvironmentTool()
    import asyncio
    
    async def test_env():
        # Get initial observation
        result = await env.execute("test", {})
        print(f"Initial observation: {result['observation_name']}")
        
        # Test each action
        for action in env.actions:
            result = await env.execute("test", {"action": action})
            print(f"After {action}: {result['observation_name']}")
    
    asyncio.run(test_env())
