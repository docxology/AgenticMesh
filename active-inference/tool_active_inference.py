import logging
import sys
import os
import json
from datetime import datetime

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy.special import softmax
import torch
import torch.nn.functional as F
import yaml

from tool_component import ToolComponent
import models
from active_inference_utils import ActiveInferenceUtils

logger = logging.getLogger(__name__)

class ActiveInferenceTool(ToolComponent):
    """
    Active Inference POMDP tool for action-perception cycles in discrete time and state spaces.
    Implements free energy minimization for belief updating and action selection.
    
    The implementation follows the mathematical framework from:
    Friston, K., et al. (2017). Active Inference: A Process Theory. Neural Computation, 29(1), 1-49.
    
    Attributes:
        n_states (int): Number of discrete states in the POMDP
        n_observations (int): Number of possible observations
        n_actions (int): Number of possible actions
        A (np.ndarray): Likelihood matrix mapping states to observations
        B (np.ndarray): Transition matrix defining state dynamics under actions
        C (np.ndarray): Prior preferences over observations
        D (np.ndarray): Initial state beliefs
        beliefs (np.ndarray): Current belief distribution over states
        prev_action (int): Previously selected action
    """

    def __init__(self):
        # Initialize model parameters for three-state environment
        self.n_states = 3  # Low, Medium, High
        self.n_observations = 3  # Low, Medium, High observations
        self.n_actions = 3  # Decrease, Stay, Increase
        
        self._tool_info = models.ToolInfoModel(
            name="tool.active_inference",
            purpose="Performs Active Inference for three-state environment",
            description="Implements Active Inference framework for belief updating and action selection in a three-state environment",
            module="tools.brodagroupsoftware.active_inference.tool"
        )
        
        self._tool_parameters = {}
        self._load_parameters()
        
        # Initialize model matrices
        # A matrix: Likelihood P(o|s) - probability of observations given states
        self.A = np.array([
            [0.8, 0.2, 0.0],  # P(o=Low|s)
            [0.1, 0.8, 0.1],  # P(o=Medium|s)
            [0.0, 0.2, 0.8]   # P(o=High|s)
        ]).T  # Transpose to match expected shape
        
        # B matrix: Transition P(s'|s,a) - state transitions under actions
        self.B = np.zeros((self.n_states, self.n_states, self.n_actions))
        
        # Decrease action
        self.B[:,:,0] = np.array([
            [0.8, 0.2, 0.0],  # From Low
            [0.7, 0.3, 0.0],  # From Medium
            [0.1, 0.8, 0.1]   # From High
        ]).T
        
        # Stay action
        self.B[:,:,1] = np.array([
            [0.7, 0.3, 0.0],  # From Low
            [0.2, 0.6, 0.2],  # From Medium
            [0.0, 0.3, 0.7]   # From High
        ]).T
        
        # Increase action
        self.B[:,:,2] = np.array([
            [0.1, 0.8, 0.1],  # From Low
            [0.0, 0.3, 0.7],  # From Medium
            [0.0, 0.2, 0.8]   # From High
        ]).T
        
        # C matrix: Preferences over observations (slight preference for medium state)
        self.C = np.array([-0.1, 1.0, -0.1])
        
        # D matrix: Initial beliefs (start with uniform distribution)
        self.D = np.ones(self.n_states) / self.n_states
        
        # E matrix: Habits/policy prior (initially uniform)
        self.E = np.ones(self.n_actions) / self.n_actions
        
        # Initialize beliefs and history
        self.beliefs = self.D.copy()
        self.prev_action = None
        self.policy_prior = self.E.copy()
        
        self.belief_history = [self.beliefs.copy()]
        self.action_history = []
        self.free_energy_history = []
        self.policy_prior_history = [self.policy_prior.copy()]
        
        # Action mapping
        self.action_mapping = ["Decrease", "Stay", "Increase"]
        
        # Initialize monitoring
        self.metrics = ActiveInferenceUtils.setup_monitoring()
        self.monitoring_enabled = True

    def info(self) -> models.ToolInfoModel:
        """Return tool information."""
        return self._tool_info

    def add_parameter(self, parameter: models.ToolParameterModel):
        """Add a parameter to the tool."""
        self._tool_parameters[parameter.name] = parameter

    def parameters(self) -> Dict[str, models.ToolParameterModel]:
        """Return tool parameters."""
        return self._tool_parameters

    def _validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate input parameters for consistency and correctness."""
        required_params = ["n_states", "n_observations", "n_actions",
                         "likelihood_matrix", "transition_matrix",
                         "preferences", "initial_beliefs"]
        
        # Check for required parameters
        for param in required_params:
            if param not in parameters:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Validate probability distributions
        matrices = {
            "likelihood_matrix": (parameters["likelihood_matrix"], (parameters["n_observations"], parameters["n_states"])),
            "transition_matrix": (parameters["transition_matrix"], (parameters["n_states"], parameters["n_states"], parameters["n_actions"])),
            "preferences": (parameters["preferences"], (parameters["n_observations"],)),
            "initial_beliefs": (parameters["initial_beliefs"], (parameters["n_states"],)),
            "habit_matrix": (parameters["habit_matrix"], (parameters["n_actions"],))
        }
        
        for name, (matrix, shape) in matrices.items():
            matrix = np.array(matrix)
            if matrix.shape != shape:
                raise ValueError(f"Invalid shape for {name}: expected {shape}, got {matrix.shape}")
            
            # Check for valid probability distributions
            if name not in ["preferences", "habit_matrix"]:
                if not np.allclose(matrix.sum(axis=0), 1.0, rtol=1e-5, atol=1e-8):
                    axis_name = "observation" if name == "likelihood_matrix" else "first state"
                    raise ValueError(f"{name} must contain valid probability distributions (sum to 1 along {axis_name} axis)")
            
            # Check for non-negative values
            if name != "preferences":
                if np.any(matrix < 0):
                    raise ValueError(f"{name} contains negative values")

    def _load_parameters(self):
        """Load required parameters."""
        observation = models.ToolParameterModel(
            name="observation",
            type="integer",
            description="Current observation index (0=Low, 1=Medium, 2=High)",
            required=True
        )
        self.add_parameter(observation)

    def _initialize_model(self, parameters: Dict[str, Any]):
        """Initialize the Active Inference model with given parameters."""
        # Validate parameters first
        self._validate_parameters(parameters)
        
        self.n_states = parameters["n_states"]
        self.n_observations = parameters["n_observations"]
        self.n_actions = parameters["n_actions"]
        
        # Convert matrices to numpy arrays and ensure proper shapes
        self.A = np.array(parameters["likelihood_matrix"])
        self.B = np.array(parameters["transition_matrix"])
        self.C = np.array(parameters["preferences"])
        self.D = np.array(parameters["initial_beliefs"])
        
        # Initialize habit matrix (E) with uniform distribution if not provided
        if "habit_matrix" in parameters:
            self.E = np.array(parameters["habit_matrix"])
            # Normalize to ensure it sums to 1
            self.E = self.E / (self.E.sum() + 1e-16)
        else:
            self.E = np.ones(self.n_actions) / self.n_actions
        
        # Initialize beliefs and policy prior
        self.beliefs = self.D.copy()
        self.policy_prior = self.E.copy()
        self.prev_action = None
        
        # Reset history
        self.belief_history = [self.beliefs.copy()]
        self.action_history = []
        self.free_energy_history = []
        self.policy_prior_history = [self.policy_prior.copy()]

    def _compute_free_energy(self, observation: int) -> np.ndarray:
        """
        Compute variational free energy (F) for belief updating.
        
        The variational free energy F is defined as:
        F = -ln P(o|s) - ln P(s)
        where:
        - F is the variational free energy to be minimized
        - P(o|s) is the likelihood from matrix A
        - P(s) is the prior from previous beliefs or initial beliefs D
        
        Args:
            observation: Current observation index
            
        Returns:
            F: Variational free energy for each possible state
        """
        # Compute likelihood term: -ln P(o|s)
        log_likelihood = np.log(self.A[observation, :] + 1e-16)
        F_likelihood = -log_likelihood  # Negative log likelihood
        
        # Compute prior term: -ln P(s)
        if self.prev_action is not None:
            prior = self.B[:, :, self.prev_action] @ self.beliefs
            log_prior = np.log(prior + 1e-16)
        else:
            log_prior = np.log(self.D + 1e-16)
        F_prior = -log_prior  # Negative log prior
            
        # Total variational free energy
        F = F_likelihood + F_prior
        
        # Log for monitoring
        logger.debug(f"F_likelihood: {F_likelihood.mean():.4f}, F_prior: {F_prior.mean():.4f}, Total F: {F.mean():.4f}")
        
        return F

    def _update_beliefs(self, observation: int):
        """
        Update beliefs by minimizing variational free energy F.
        
        Uses softmax to convert free energy F to posterior probabilities:
        P(s|o) ∝ exp(-F)
        """
        # Store prior beliefs before update
        belief_prior = self.beliefs.copy()
        
        # Compute free energy and update beliefs
        F = self._compute_free_energy(observation)
        self.beliefs = softmax(-F)  # Minimize F through softmax (posterior)
        
        # Store history
        self.belief_history.append(self.beliefs.copy())
        self.free_energy_history.append(float(F.mean()))
        
        return belief_prior, F

    def _compute_expected_free_energy(self) -> np.ndarray:
        """
        Compute expected free energy for action selection.
        
        The expected free energy (G) combines:
        G = Epistemic Value + Pragmatic Value
        where:
        1. Epistemic Value: Expected information gain about hidden states
           - Measures how much uncertainty about states would be resolved
           - Drives exploration and information-seeking behavior
        2. Pragmatic Value: Expected alignment with preferences
           - Measures how well expected observations match preferences
           - Drives goal-directed behavior
        
        Returns:
            Expected free energy for each possible action
        """
        G = np.zeros(self.n_actions)
        epsilon = 1e-16  # Numerical stability constant
        
        for a in range(self.n_actions):
            # Predicted states under action
            predicted_states = self.B[:, :, a] @ self.beliefs
            
            # Expected observations
            expected_obs = self.A @ predicted_states
            
            # 1. Epistemic Value (Expected Information Gain)
            # Compute state entropy difference
            predicted_state_entropy = -(predicted_states * np.log(predicted_states + epsilon)).sum()
            current_entropy = -(self.beliefs * np.log(self.beliefs + epsilon)).sum()
            epistemic_value = predicted_state_entropy - current_entropy
            
            # 2. Pragmatic Value (Preference Alignment)
            # KL divergence between expected observations and preferences
            normalized_pref = np.exp(self.C) / (np.exp(self.C).sum() + epsilon)
            pragmatic_value = -(expected_obs * np.log(normalized_pref + epsilon)).sum()
            
            # Combine values (negative because we want to maximize epistemic value and minimize pragmatic cost)
            G[a] = -epistemic_value + pragmatic_value
            
        return G

    def _update_policy_prior(self, G: np.ndarray):
        """
        Update policy prior (habits) based on expected free energy.
        Uses Bayesian updating in log space for numerical stability.
        
        Args:
            G: Expected free energy for each action
        """
        # Convert expected free energy to probability distribution (lower G = higher probability)
        # Temperature parameter could be made configurable
        temperature = 1.0
        policy_likelihood = softmax(-G / temperature)
        
        # Update policy prior using Bayes rule (in log space for numerical stability)
        log_posterior = np.log(policy_likelihood + 1e-16) + np.log(self.policy_prior + 1e-16)
        
        # Softmax for numerical stability and proper normalization
        self.policy_prior = softmax(log_posterior)
        
        # Add small noise to prevent complete determinism (optional)
        noise = np.random.dirichlet(np.ones(self.n_actions) * 100)  # High concentration = small noise
        self.policy_prior = 0.99 * self.policy_prior + 0.01 * noise
        
        # Store in history
        self.policy_prior_history.append(self.policy_prior.copy())

    def _select_action(self) -> Tuple[int, np.ndarray]:
        """
        Select action by minimizing expected free energy and updating habits.
        Samples action from policy posterior distribution.
        
        Returns:
            Tuple containing:
            - Selected action index
            - Policy posterior probabilities
            - Expected free energy
            - Policy prior before update
        """
        # Compute expected free energy
        G = self._compute_expected_free_energy()
        
        # Store policy prior before update
        policy_prior = self.policy_prior.copy()
        
        # Update policy prior (habits)
        self._update_policy_prior(G)
        
        # Compute policy posterior (action probabilities)
        policy_posterior = softmax(-G) * self.policy_prior
        policy_posterior = policy_posterior / (policy_posterior.sum() + 1e-16)
        
        # Sample action from policy posterior
        action = int(np.random.choice(self.n_actions, p=policy_posterior))
        self.action_history.append(action)
        
        return action, policy_posterior, G, policy_prior

    def get_history(self) -> Dict[str, Any]:
        """
        Get the history of beliefs, actions, free energies, and policy priors.
        
        Returns:
            Dictionary containing historical data
        """
        return {
            "belief_history": [b.tolist() for b in self.belief_history],
            "action_history": self.action_history,
            "free_energy_history": self.free_energy_history,
            "policy_prior_history": [p.tolist() for p in self.policy_prior_history]
        }

    def return_value(self) -> models.ToolReturnValueModel:
        """Define return value structure."""
        return models.ToolReturnValueModel(
            type="dict",
            description="Returns beliefs, selected action, and history",
            properties={
                "beliefs": {"type": "array", "description": "Current belief distribution over states"},
                "selected_action": {"type": "integer", "description": "Selected action index"},
                "expected_free_energy": {"type": "array", "description": "Expected free energy for each action"},
                "policy_prior": {"type": "array", "description": "Current policy prior distribution"},
                "history": {
                    "type": "object",
                    "description": "Historical data of beliefs, actions, free energies, and policy priors"
                }
            }
        )

    async def execute(self, iid: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one step of active inference."""
        try:
            observation = parameters["observation"]
            if not 0 <= observation < self.n_observations:
                raise ValueError(f"Invalid observation index: {observation}")
            
            # Update beliefs and get intermediate values
            belief_prior, F = self._update_beliefs(observation)
            belief_posterior = self.beliefs.copy()
            
            # Select action and get intermediate values
            action, policy_posterior, G, policy_prior = self._select_action()
            self.prev_action = action
            
            # Prepare detailed output
            output = {
                "observation": observation,
                "belief_prior": belief_prior.tolist(),
                "belief_posterior": belief_posterior.tolist(),
                "variational_free_energy": F.tolist(),
                "policy_prior": policy_prior.tolist(),
                "policy_posterior": policy_posterior.tolist(),
                "expected_free_energy": G.tolist(),
                "selected_action": int(action),
                "selected_action_name": self.action_mapping[action],
                "history": self.get_history()
            }
            
            # Log in JSON format
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "component": "active_inference",
                "iid": iid,
                "observation": observation,
                "belief_prior": [f"{p:.3f}" for p in belief_prior],
                "belief_posterior": [f"{p:.3f}" for p in belief_posterior],
                "variational_free_energy": [f"{f:.3f}" for f in F],
                "policy_prior": [f"{p:.3f}" for p in policy_prior],
                "policy_posterior": [f"{p:.3f}" for p in policy_posterior],
                "expected_free_energy": [f"{g:.3f}" for g in G],
                "selected_action": self.action_mapping[action]
            }
            logger.info(json.dumps(log_data))
            
            return output
            
        except Exception as e:
            error_data = {
                "timestamp": datetime.now().isoformat(),
                "level": "ERROR",
                "component": "active_inference",
                "iid": iid,
                "error": str(e),
                "beliefs": [f"{p:.3f}" for p in self.beliefs] if hasattr(self, 'beliefs') else None
            }
            logger.error(json.dumps(error_data))
            raise

