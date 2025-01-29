"""
Active Inference Utilities Module

Provides shared functionality and helper methods for Active Inference components.
"""

import logging
import yaml
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class ActiveInferenceUtils:
    """Utility class for Active Inference operations"""
    
    @staticmethod
    def load_configuration(config_path: str) -> Dict:
        """Load and validate configuration from YAML file"""
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                ActiveInferenceUtils.validate_configuration(config)
                return config
        except Exception as e:
            logger.error(f"Configuration loading failed: {e}")
            raise

    @staticmethod
    def validate_configuration(config: Dict) -> None:
        """Validate configuration structure and values"""
        required_sections = ['model', 'visualization', 'logging']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing configuration section: {section}")
        
        # Validate model dimensions
        model = config.get('model', {})
        dimensions = model.get('dimensions', {})
        for dim in ['n_states', 'n_observations', 'n_actions']:
            if dim not in dimensions or not isinstance(dimensions[dim], int) or dimensions[dim] <= 0:
                raise ValueError(f"Invalid {dim} in configuration")

    @staticmethod
    def validate_matrices(matrices: Dict[str, np.ndarray], dimensions: Dict[str, int]) -> None:
        """Validate matrix shapes and probability distributions"""
        n_states = dimensions['n_states']
        n_observations = dimensions['n_observations']
        n_actions = dimensions['n_actions']
        
        # Check likelihood matrix
        if matrices['likelihood'].shape != (n_observations, n_states):
            raise ValueError(f"Invalid likelihood matrix shape: {matrices['likelihood'].shape}")
        if not np.allclose(matrices['likelihood'].sum(axis=0), 1):
            raise ValueError("Likelihood matrix columns must sum to 1")
        
        # Check transition matrix
        if matrices['transition'].shape != (n_states, n_states, n_actions):
            raise ValueError(f"Invalid transition matrix shape: {matrices['transition'].shape}")
        if not np.allclose(matrices['transition'].sum(axis=1), 1):
            raise ValueError("Transition matrix must sum to 1 along second dimension")
        
        # Check preferences
        if matrices['preferences'].shape != (n_observations,):
            raise ValueError(f"Invalid preferences shape: {matrices['preferences'].shape}")
        
        # Check initial beliefs
        if matrices['initial_beliefs'].shape != (n_states,):
            raise ValueError(f"Invalid initial beliefs shape: {matrices['initial_beliefs'].shape}")
        if not np.isclose(matrices['initial_beliefs'].sum(), 1):
            raise ValueError("Initial beliefs must sum to 1")

    @staticmethod
    def initialize_logging(config: Dict) -> None:
        """Initialize logging based on configuration"""
        log_config = config.get('logging', {})
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    @staticmethod
    def ensure_valid_distribution(dist: np.ndarray, axis: int = -1) -> np.ndarray:
        """Ensure array is a valid probability distribution"""
        # Add small epsilon to avoid log(0)
        eps = 1e-12
        dist = np.clip(dist, eps, None)
        # Normalize
        return dist / dist.sum(axis=axis, keepdims=True)

    @staticmethod
    def compute_entropy(dist: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute entropy of probability distribution"""
        return -np.sum(dist * np.log(dist + 1e-12), axis=axis)

    @staticmethod
    def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Compute softmax with temperature"""
        x = x / temperature
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / exp_x.sum() 