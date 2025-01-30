"""
Active Inference Visualization Tool

This module provides comprehensive visualization capabilities for analyzing
active inference agent behavior and performance using the logged JSON output.
"""

# Set matplotlib backend to non-GUI backend before other imports
import matplotlib
matplotlib.use('Agg')

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import os
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import networkx as nx
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ActiveInferenceVisualizer:
    """Visualizer for Active Inference agent data."""
    
    def __init__(self, log_file: str):
        """Initialize visualizer with log file path."""
        self.log_file = Path(log_file)
        self.data = self._load_data()
        self.setup_style()
        
    def setup_style(self):
        """Configure enhanced plot styling."""
        plt.style.use('default')
        sns.set_theme(style="whitegrid", font_scale=1.2)
        
        # Define color palettes
        self.state_colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
        self.action_colors = ['#f1c40f', '#9b59b6', '#1abc9c']  # Yellow, Purple, Turquoise
        self.analysis_colors = {
            'epistemic': '#3498db',    # Blue
            'pragmatic': '#e74c3c',    # Red
            'combined': '#2ecc71',     # Green
            'confidence': '#f1c40f',   # Yellow
            'threshold': '#95a5a6',    # Gray
            'highlight': '#e67e22'     # Orange
        }
        
        # Set default color cycles
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=self.action_colors)
        
        # Enhanced formatting with better readability
        plt.rcParams.update({
            # Figure settings
            'figure.figsize': [12, 8],
            'figure.dpi': 300,
            'figure.constrained_layout.use': True,
            'figure.facecolor': 'white',
            'figure.edgecolor': 'white',
            
            # Font settings
            'font.size': 12,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
            
            # Axes settings
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'axes.titleweight': 'bold',
            'axes.labelweight': 'bold',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'axes.grid.which': 'major',
            'axes.axisbelow': True,
            
            # Grid settings
            'grid.color': '#E0E0E0',
            'grid.linestyle': '-',
            'grid.linewidth': 0.5,
            'grid.alpha': 0.5,
            
            # Tick settings
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'xtick.major.size': 5,
            'ytick.major.size': 5,
            'xtick.minor.size': 3,
            'ytick.minor.size': 3,
            
            # Legend settings
            'legend.fontsize': 12,
            'legend.title_fontsize': 14,
            'legend.frameon': True,
            'legend.framealpha': 0.8,
            'legend.edgecolor': '#CCCCCC',
            
            # Line settings
            'lines.linewidth': 2,
            'lines.markersize': 8,
            'lines.markeredgewidth': 2,
            
            # Scatter plot settings
            'scatter.marker': 'o',
            'scatter.edgecolors': 'white',
            
            # Saving settings
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.facecolor': 'white',
            'savefig.edgecolor': 'none'
        })
        
        # Set seaborn specific settings
        sns.set_style("whitegrid", {
            'axes.grid': True,
            'axes.edgecolor': '#CCCCCC',
            'grid.color': '#E0E0E0',
            'grid.linestyle': '-',
            'grid.linewidth': 0.5
        })
        
        # Set default colormaps
        plt.set_cmap('viridis')
        
        # Configure color normalization for consistent color scaling
        self.norm = plt.Normalize(0, 1)

    def _validate_entry(self, entry: Dict) -> bool:
        """Validate the structure of a log entry.
        
        Args:
            entry: Dictionary containing the log entry data
            
        Returns:
            bool: True if entry is valid, False otherwise
        """
        try:
            # Check basic structure
            if not isinstance(entry, dict):
                logger.debug("Entry is not a dictionary")
                return False
                
            # Check component type - allow both active_inference and active-inference
            component = entry.get("component", "").replace("-", "_").lower()
            if component != "active_inference":
                logger.debug(f"Invalid component type: {component}")
                return False
                
            # Required fields with their expected types
            required_fields = {
                "belief_prior": (list, np.ndarray, str),
                "belief_posterior": (list, np.ndarray, str),
                "policy_prior": (list, np.ndarray, str),
                "policy_posterior": (list, np.ndarray, str),
                "variational_free_energy": (list, np.ndarray, str),
                "expected_free_energy": (list, np.ndarray, str)
            }
            
            # Optional fields with their expected types
            optional_fields = {
                "selected_action": (int, np.integer, float, np.floating, str),
                "observation": (int, np.integer, float, np.floating, str),
                "epistemic_values": (list, np.ndarray, str),
                "pragmatic_values": (list, np.ndarray, str),
                "timestamp": (str, float, int)
            }
            
            # Validate required fields
            for field, types in required_fields.items():
                if field not in entry:
                    logger.debug(f"Missing required field: {field}")
                    return False
                    
                value = entry[field]
                if not isinstance(value, types):
                    # If it's a string representation of a list, try to parse it
                    if isinstance(value, str):
                        try:
                            value = [float(x.strip('"')) for x in value.strip('[]').split(',')]
                            entry[field] = value
                        except (ValueError, TypeError):
                            logger.debug(f"Field {field} has invalid string format: {value}")
                            return False
                    else:
                        logger.debug(f"Field {field} has wrong type: {type(value)}, expected {types}")
                        return False
                    
                # For array-like fields, check they're not empty
                if isinstance(value, (list, np.ndarray)) and len(value) == 0:
                    logger.debug(f"Field {field} is empty")
                    return False
            
            # Validate optional fields if present
            for field, types in optional_fields.items():
                if field in entry:
                    value = entry[field]
                    if not isinstance(value, types):
                        # Handle string action names
                        if field == "selected_action" and isinstance(value, str):
                            if value in ["Decrease", "Stay", "Increase"]:
                                continue
                        logger.debug(f"Optional field {field} has wrong type: {type(value)}, expected {types}")
                        return False
                        
                    # For array-like fields, check they're not empty
                    if isinstance(value, (list, np.ndarray)) and len(value) == 0:
                        logger.debug(f"Optional field {field} is empty")
                        return False
            
            # Validate array lengths match expected dimensions
            belief_len = len(entry["belief_prior"])
            if not all(len(entry[k]) == belief_len for k in ["belief_posterior", "variational_free_energy"]):
                logger.debug("Inconsistent belief/VFE array lengths")
                return False
                
            policy_len = len(entry["policy_prior"])
            if not all(len(entry[k]) == policy_len for k in ["policy_posterior", "expected_free_energy"]):
                logger.debug("Inconsistent policy/EFE array lengths")
                return False
                
            # If value components exist, check their lengths
            if "epistemic_values" in entry and "pragmatic_values" in entry:
                if len(entry["epistemic_values"]) != policy_len or len(entry["pragmatic_values"]) != policy_len:
                    logger.debug("Inconsistent value component array lengths")
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error validating entry: {str(e)}")
            return False
            
    def _load_data(self) -> List[Dict]:
        """Load and validate data from log file."""
        try:
            # Ensure path is resolved
            log_path = Path(self.log_file).resolve()
            logger.info(f"Loading data from: {log_path}")
            
            if not log_path.exists():
                raise FileNotFoundError(f"Log file not found: {log_path}")
                
            if not log_path.is_file():
                raise ValueError(f"Path is not a file: {log_path}")
                
            if log_path.stat().st_size == 0:
                raise ValueError(f"Log file is empty: {log_path}")
            
            # Read the file line by line since it's JSONL format
            data = []
            with open(log_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                            
                        # Parse outer JSON structure
                        outer_entry = json.loads(line)
                        
                        # Check if this is a tool_active_inference log entry
                        if outer_entry.get("logger") == "tool_active_inference":
                            try:
                                # Parse the nested JSON message
                                inner_entry = json.loads(outer_entry["message"])
                                
                                if self._validate_entry(inner_entry):
                                    # Convert string values to numeric
                                    for key in ['belief_prior', 'belief_posterior', 
                                              'policy_prior', 'policy_posterior',
                                              'variational_free_energy', 'expected_free_energy',
                                              'epistemic_values', 'pragmatic_values']:
                                        if key in inner_entry:
                                            inner_entry[key] = [float(x) for x in inner_entry[key]]
                                    
                                    # Handle action and observation
                                    if 'selected_action' in inner_entry:
                                        if isinstance(inner_entry['selected_action'], str):
                                            action_map = {'Decrease': 0, 'Stay': 1, 'Increase': 2}
                                            inner_entry['selected_action'] = action_map.get(inner_entry['selected_action'], 0)
                                    
                                    if 'observation' in inner_entry:
                                        inner_entry['observation'] = int(inner_entry['observation'])
                                    
                                    # Add timestamp from outer entry
                                    inner_entry['timestamp'] = outer_entry.get('timestamp')
                                    
                                    data.append(inner_entry)
                                    
                            except json.JSONDecodeError as e:
                                logger.debug(f"Invalid nested JSON on line {line_num}: {str(e)}")
                                continue
                                
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {str(e)}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {str(e)}")
                        continue
            
            if not data:
                raise ValueError("No valid active inference entries found in log file")
                
            # Sort entries by timestamp if available
            if all('timestamp' in entry for entry in data):
                data.sort(key=lambda x: x['timestamp'])
                
            logger.info(f"Successfully loaded {len(data)} valid entries")
            
            # Log some sample data for debugging
            if data:
                logger.debug("Sample entry keys: " + ", ".join(data[0].keys()))
                logger.debug(f"First entry belief shape: {len(data[0].get('belief_prior', []))}")
                logger.debug(f"First entry beliefs: {data[0].get('belief_prior', [])}")
                
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return []
    
    def _extract_demo_data(self):
        """Extract relevant data from log entries with validation."""
        try:
            # Initialize data structures
            steps = []
            beliefs = []
            policy_priors = []
            policy_posteriors = []
            free_energies = []
            actions = []
            observations = []
            epistemic_values = []
            pragmatic_values = []
            
            # Track valid entries for consistent data
            valid_entries = 0
            
            # Define state and action mappings
            state_map = {'Low': 0, 'Medium': 1, 'High': 2}
            action_map = {'Decrease': 0, 'Stay': 1, 'Increase': 2}
            
            for entry in self.data:
                try:
                    # Initialize step data
                    valid_entries += 1
                    steps.append(valid_entries - 1)  # 0-based indexing
                    
                    # Extract and validate beliefs
                    belief_prior = np.array(entry.get("belief_prior", [0.0, 0.0, 0.0]), dtype=float)
                    belief_posterior = np.array(entry.get("belief_posterior", [0.0, 0.0, 0.0]), dtype=float)
                    
                    beliefs.append({
                        "prior": belief_prior,
                        "posterior": belief_posterior
                    })
                    
                    # Extract and validate policies
                    policy_prior = np.array(entry.get("policy_prior", [0.0, 0.0, 0.0]), dtype=float)
                    policy_posterior = np.array(entry.get("policy_posterior", [0.0, 0.0, 0.0]), dtype=float)
                    
                    policy_priors.append(policy_prior)
                    policy_posteriors.append(policy_posterior)
                    
                    # Extract and validate free energies
                    vfe = np.array(entry.get("variational_free_energy", [0.0, 0.0, 0.0]), dtype=float)
                    efe = np.array(entry.get("expected_free_energy", [0.0, 0.0, 0.0]), dtype=float)
                    
                    free_energies.append({
                        "variational": vfe,
                        "expected": efe
                    })
                    
                    # Extract and validate value components
                    ep_vals = np.array(entry.get("epistemic_values", [0.0, 0.0, 0.0]), dtype=float)
                    pr_vals = np.array(entry.get("pragmatic_values", [0.0, 0.0, 0.0]), dtype=float)
                    
                    epistemic_values.append(ep_vals)
                    pragmatic_values.append(pr_vals)
                    
                    # Extract and validate action/observation
                    action = entry.get("selected_action", 0)
                    if isinstance(action, str):
                        action = action_map.get(action, 0)
                    actions.append(int(action))
                    
                    observation = entry.get("observation", 0)
                    if isinstance(observation, str):
                        observation = state_map.get(observation, 0)
                    observations.append(int(observation))
                        
                except Exception as e:
                    logger.warning(f"Error processing entry: {str(e)}")
                    continue
            
            # Ensure we have at least one valid entry
            if not steps:
                logger.warning("No valid entries found in log file")
                return {
                    "steps": np.array([]),
                    "beliefs": [],
                    "policy_priors": np.array([]),
                    "policy_posteriors": np.array([]),
                    "free_energies": [],
                    "actions": np.array([]),
                    "observations": np.array([]),
                    "epistemic_values": np.array([]),
                    "pragmatic_values": np.array([])
                }
            
            # Convert lists to numpy arrays for consistent shapes
            result = {
                "steps": np.array(steps),
                "beliefs": beliefs,
                "policy_priors": np.array(policy_priors),
                "policy_posteriors": np.array(policy_posteriors),
                "free_energies": free_energies,
                "actions": np.array(actions),
                "observations": np.array(observations),
                "epistemic_values": np.array(epistemic_values),
                "pragmatic_values": np.array(pragmatic_values)
            }
            
            # Log data shapes for debugging
            logger.info(f"Extracted {len(steps)} valid entries")
            logger.info(f"Data shapes - Policy priors: {result['policy_priors'].shape}, " 
                       f"Policy posteriors: {result['policy_posteriors'].shape}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in data extraction: {str(e)}")
            return {
                "steps": np.array([]),
                "beliefs": [],
                "policy_priors": np.array([]),
                "policy_posteriors": np.array([]),
                "free_energies": [],
                "actions": np.array([]),
                "observations": np.array([]),
                "epistemic_values": np.array([]),
                "pragmatic_values": np.array([])
            }
    
    def _validate_array_shape(self, array: np.ndarray, expected_dims: int, name: str) -> bool:
        """Validate array shape and dimensions.
        
        Args:
            array: Array to validate
            expected_dims: Expected number of dimensions
            name: Name of the array for logging
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            if array is None or array.size == 0:
                logger.warning(f"Empty {name} array")
                return False
                
            if len(array.shape) != expected_dims:
                logger.warning(f"Invalid {name} shape: expected {expected_dims}D, got {len(array.shape)}D")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating {name}: {str(e)}")
            return False

    def plot_belief_evolution(self, save_dir: Path = None):
        """Plot the evolution of beliefs over time with enhanced formatting."""
        try:
            data = self._extract_demo_data()
            steps = data["steps"]
            beliefs = data["beliefs"]
            
            if len(steps) == 0:
                logger.warning("No time steps found in data")
                return None
                
            if len(beliefs) == 0:
                logger.warning("No belief data found")
                return None
                
            # Convert to numpy arrays with validation
            try:
                prior_beliefs = np.array([b["prior"] for b in beliefs])
                posterior_beliefs = np.array([b["posterior"] for b in beliefs])
                
                if prior_beliefs.size == 0 or posterior_beliefs.size == 0:
                    logger.warning("Empty belief arrays found")
                    return None
                    
                if len(prior_beliefs.shape) != 2 or len(posterior_beliefs.shape) != 2:
                    logger.warning("Invalid belief array shapes")
                    return None
                    
            except Exception as e:
                logger.error(f"Error converting belief data to arrays: {str(e)}")
                return None
            
            # Create figure
            fig = plt.figure(figsize=(15, 12))
            gs = GridSpec(3, 1, height_ratios=[2, 2, 1], figure=fig)
            
            # Define state labels and colors
            states = ['Low', 'Medium', 'High']
            colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
            
            try:
                # Plot prior beliefs
                ax1 = fig.add_subplot(gs[0])
                for i in range(prior_beliefs.shape[1]):
                    ax1.plot(steps, prior_beliefs[:, i], 
                            label=states[i], 
                            color=colors[i],
                            marker='o', markersize=4, linewidth=2)
                ax1.set_title('Prior Beliefs Evolution', pad=20)
                ax1.set_xlabel('Time Step')
                ax1.set_ylabel('Prior Belief Probability')
                ax1.legend(title='States', bbox_to_anchor=(1.02, 1), loc='upper left')
                ax1.set_ylim(-0.05, 1.05)  # Add padding to y-axis
                
                # Plot posterior beliefs
                ax2 = fig.add_subplot(gs[1])
                for i in range(posterior_beliefs.shape[1]):
                    ax2.plot(steps, posterior_beliefs[:, i], 
                            label=states[i], 
                            color=colors[i],
                            marker='o', markersize=4, linewidth=2)
                ax2.set_title('Posterior Beliefs Evolution', pad=20)
                ax2.set_xlabel('Time Step')
                ax2.set_ylabel('Posterior Belief Probability')
                ax2.legend(title='States', bbox_to_anchor=(1.02, 1), loc='upper left')
                ax2.set_ylim(-0.05, 1.05)  # Add padding to y-axis
                
                # Plot belief updates magnitude
                ax3 = fig.add_subplot(gs[2])
                updates = np.abs(posterior_beliefs - prior_beliefs).mean(axis=1)
                ax3.bar(steps, updates, alpha=0.6, color='darkblue')
                ax3.set_title('Magnitude of Belief Updates', pad=20)
                ax3.set_xlabel('Time Step')
                ax3.set_ylabel('Average Update')
                
                fig.set_constrained_layout(True)
                
                if save_dir:
                    plt.savefig(save_dir / 'belief_evolution.png', bbox_inches='tight', dpi=300)
                
                return fig
                
            except Exception as e:
                logger.error(f"Error plotting belief evolution: {str(e)}")
                plt.close(fig)
                return None
                
        except Exception as e:
            logger.error(f"Error generating belief evolution plot: {str(e)}")
            plt.close('all')
            return None
    
    def plot_policy_evolution(self, save_dir: Path = None):
        """Plot the evolution of policy priors and posteriors with enhanced formatting."""
        try:
            data = self._extract_demo_data()
            steps = data["steps"]
            
            if len(steps) == 0:
                logger.warning("No time steps found in data")
                return None
            
            # Convert and validate policy data
            try:
                policy_priors = data["policy_priors"]
                policy_posteriors = data["policy_posteriors"]
                
                if policy_priors.size == 0 or policy_posteriors.size == 0:
                    logger.warning("Empty policy arrays found")
                    return None
                    
                if len(policy_priors.shape) != 2 or len(policy_posteriors.shape) != 2:
                    logger.warning("Invalid policy array shapes")
                    return None
                    
            except Exception as e:
                logger.error(f"Error accessing policy data: {str(e)}")
                return None
            
            # Create figure
            fig = plt.figure(figsize=(15, 12))
            gs = GridSpec(3, 1, height_ratios=[2, 2, 1], figure=fig)
            
            # Define action labels and colors
            actions = ['Decrease', 'Stay', 'Increase']
            colors = ['#f1c40f', '#9b59b6', '#1abc9c']  # Yellow, Purple, Turquoise
            
            try:
                # Plot policy priors
                ax1 = fig.add_subplot(gs[0])
                for i in range(policy_priors.shape[1]):
                    ax1.plot(steps, policy_priors[:, i], 
                            label=actions[i], color=colors[i],
                            marker='o', markersize=4, linewidth=2)
                ax1.set_title('Policy Prior Evolution', pad=20)
                ax1.set_xlabel('Time Step')
                ax1.set_ylabel('Prior Probability')
                ax1.legend(title='Actions', bbox_to_anchor=(1.02, 1), loc='upper left')
                ax1.set_ylim(-0.05, 1.05)  # Add padding to y-axis
                
                # Plot policy posteriors
                ax2 = fig.add_subplot(gs[1])
                for i in range(policy_posteriors.shape[1]):
                    ax2.plot(steps, policy_posteriors[:, i], 
                            label=actions[i], color=colors[i],
                            marker='o', markersize=4, linewidth=2)
                ax2.set_title('Policy Posterior Evolution', pad=20)
                ax2.set_xlabel('Time Step')
                ax2.set_ylabel('Posterior Probability')
                ax2.legend(title='Actions', bbox_to_anchor=(1.02, 1), loc='upper left')
                ax2.set_ylim(-0.05, 1.05)  # Add padding to y-axis
                
                # Plot policy updates magnitude
                ax3 = fig.add_subplot(gs[2])
                updates = np.abs(policy_posteriors - policy_priors).mean(axis=1)
                ax3.bar(steps, updates, alpha=0.6, color='darkred')
                ax3.set_title('Magnitude of Policy Updates', pad=20)
                ax3.set_xlabel('Time Step')
                ax3.set_ylabel('Average Update')
                
                fig.set_constrained_layout(True)
                
                if save_dir:
                    plt.savefig(save_dir / 'policy_evolution.png', bbox_inches='tight', dpi=300)
                
                return fig
                
            except Exception as e:
                logger.error(f"Error plotting policy evolution: {str(e)}")
                plt.close(fig)
                return None
                
        except Exception as e:
            logger.error(f"Error generating policy evolution plot: {str(e)}")
            plt.close('all')
            return None

    def plot_free_energy_components(self, save_dir: Path = None):
        """Plot detailed free energy analysis with enhanced visualizations."""
        try:
            data = self._extract_demo_data()
            steps = data["steps"]
            free_energies = data["free_energies"]
            
            if len(steps) == 0:
                logger.warning("No time steps found in data")
                return None
            
            fig = plt.figure(figsize=(20, 20))
            gs = GridSpec(4, 2, height_ratios=[2, 2, 2, 1], figure=fig)
            
            # Define consistent colors
            state_colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
            action_colors = ['#f1c40f', '#9b59b6', '#1abc9c']  # Yellow, Purple, Turquoise
            
            # 1. Variational Free Energy by State
            ax1 = fig.add_subplot(gs[0, 0])
            vfe = np.array([fe["variational"] for fe in free_energies])
            for i in range(vfe.shape[1]):
                ax1.plot(steps, vfe[:, i], 
                        label=f'State {["Low", "Medium", "High"][i]}', 
                        color=state_colors[i],
                        marker='o', markersize=4, linewidth=2)
            ax1.set_title('Variational Free Energy by State', pad=20, fontweight='bold')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Free Energy')
            ax1.legend(title='States', title_fontsize=12)
            
            # 2. Expected Free Energy by Action
            ax2 = fig.add_subplot(gs[0, 1])
            efe = np.array([fe["expected"] for fe in free_energies])
            for i in range(efe.shape[1]):
                ax2.plot(steps, efe[:, i], 
                        label=f'Action {["Decrease", "Stay", "Increase"][i]}', 
                        color=action_colors[i],
                        marker='o', markersize=4, linewidth=2)
            ax2.set_title('Expected Free Energy by Action', pad=20, fontweight='bold')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Free Energy')
            ax2.legend(title='Actions', title_fontsize=12)
            
            # 3. Cumulative Free Energy Components
            ax3 = fig.add_subplot(gs[1, 0])
            cum_vfe = np.cumsum(vfe.mean(axis=1))
            cum_efe = np.cumsum(efe.mean(axis=1))
            ax3.plot(steps, cum_vfe, '-', color='blue', linewidth=2, label='Cumulative VFE')
            ax3.plot(steps, cum_efe, '-', color='red', linewidth=2, label='Cumulative EFE')
            ax3.set_title('Cumulative Free Energy Components', pad=20, fontweight='bold')
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('Cumulative Free Energy')
            ax3.legend(title='Components', title_fontsize=12)
            
            # 4. Free Energy Differences
            ax4 = fig.add_subplot(gs[1, 1])
            vfe_diff = np.diff(vfe.mean(axis=1))
            efe_diff = np.diff(efe.mean(axis=1))
            ax4.plot(steps[1:], vfe_diff, '-', color='blue', label='VFE Change')
            ax4.plot(steps[1:], efe_diff, '-', color='red', label='EFE Change')
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax4.set_title('Free Energy Changes Between Steps', pad=20, fontweight='bold')
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Free Energy Difference')
            ax4.legend(title='Components', title_fontsize=12)
            
            # 5. Free Energy Distribution
            ax5 = fig.add_subplot(gs[2, 0])
            # Create violin plot with fixed x-axis
            positions = [0, 1]  # Fixed positions for VFE and EFE
            violins = ax5.violinplot([vfe.flatten(), efe.flatten()], positions=positions)
            
            # Set colors for violin plots
            for pc in violins['bodies']:
                pc.set_facecolor('blue')
                pc.set_alpha(0.3)
            
            # Customize violin plot appearance
            ax5.set_xticks(positions)
            ax5.set_xticklabels(['VFE', 'EFE'])
            ax5.set_title('Free Energy Distribution', pad=20, fontweight='bold')
            ax5.set_ylabel('Free Energy Value')
            
            # 6. Free Energy Correlation
            ax6 = fig.add_subplot(gs[2, 1])
            vfe_mean = vfe.mean(axis=1)
            efe_mean = efe.mean(axis=1)
            scatter = ax6.scatter(vfe_mean, efe_mean, alpha=0.6, c=steps, cmap='viridis')
            ax6.set_xlabel('Mean VFE')
            ax6.set_ylabel('Mean EFE')
            ax6.set_title('VFE vs EFE Correlation', pad=20, fontweight='bold')
            plt.colorbar(scatter, ax=ax6, label='Time Step')
            
            # 7. Average Free Energy Components
            ax7 = fig.add_subplot(gs[3, :])
            x = np.arange(3)
            width = 0.35
            
            # Compute means and standard errors
            vfe_mean = vfe.mean(axis=0)
            vfe_sem = vfe.std(axis=0) / np.sqrt(len(steps))
            efe_mean = efe.mean(axis=0)
            efe_sem = efe.std(axis=0) / np.sqrt(len(steps))
            
            # Plot bars with error bars
            ax7.bar(x - width/2, vfe_mean, width, yerr=vfe_sem, 
                   label='Avg VFE', color='blue', alpha=0.6, capsize=5)
            ax7.bar(x + width/2, efe_mean, width, yerr=efe_sem,
                   label='Avg EFE', color='red', alpha=0.6, capsize=5)
            ax7.set_title('Average Free Energy Components', pad=20, fontweight='bold')
            ax7.set_xticks(x)
            ax7.set_xticklabels(['Low/Decrease', 'Medium/Stay', 'High/Increase'])
            ax7.legend(title='Components', title_fontsize=12)
            
            plt.suptitle('Free Energy Analysis Dashboard', 
                        fontsize=24, fontweight='bold', y=1.02)
            fig.set_constrained_layout(True)
            
            if save_dir:
                plt.savefig(save_dir / 'free_energy_analysis.png', 
                          bbox_inches='tight', dpi=300,
                          facecolor='white', edgecolor='none')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error generating free energy analysis plot: {str(e)}")
            plt.close('all')
            return None

    def plot_state_action_analysis(self, save_dir: Path = None):
        """Create comprehensive state-action transition analysis."""
        try:
            data = self._extract_demo_data()
            steps = data["steps"]
            
            if len(steps) == 0:
                logger.warning("No time steps found in data")
                return None
            
            # Get raw data and ensure numeric type
            actions = np.array(data["actions"], dtype=int)
            observations = np.array(data["observations"], dtype=int)
            
            # Define consistent state and action orders
            states = ['Low', 'Medium', 'High']
            actions_list = ['Decrease', 'Stay', 'Increase']
            
            # Create transition matrix
            transition_matrix = np.zeros((3, 3, 3))  # from_state × to_state × action
            for s1, s2, a in zip(observations[:-1], observations[1:], actions[:-1]):
                transition_matrix[s1, s2, a] += 1
            
            fig = plt.figure(figsize=(20, 15))
            gs = GridSpec(2, 2, figure=fig)
            
            # 1. Transition Heatmap
            ax1 = fig.add_subplot(gs[0, 0])
            
            # Create heatmap data
            heatmap_data = []
            row_labels = []
            for s1 in range(3):
                for a in range(3):
                    heatmap_data.append(transition_matrix[s1, :, a])
                    row_labels.append(f"{states[s1]}-{actions_list[a]}")
            
            heatmap_data = np.array(heatmap_data)
            
            # Plot heatmap
            sns.heatmap(heatmap_data, annot=True, fmt='.0f',
                       xticklabels=states,
                       yticklabels=row_labels,
                       cmap='YlOrRd', ax=ax1)
            ax1.set_title('State-Action Transition Heatmap', pad=20, fontweight='bold')
            ax1.set_xlabel('Next State')
            ax1.set_ylabel('Current State - Action')
            
            # 2. State Transition Graph
            ax2 = fig.add_subplot(gs[0, 1])
            G = nx.DiGraph()
            
            # Add nodes
            for state in states:
                G.add_node(state)
            
            # Compute edge weights and labels
            edge_weights = {}
            for s1 in range(3):
                for s2 in range(3):
                    weight = transition_matrix[s1, s2, :].sum()
                    if weight > 0:
                        # Create action distribution string
                        action_dist = [f"{actions_list[a]}:{transition_matrix[s1, s2, a]:.0f}"
                                     for a in range(3) if transition_matrix[s1, s2, a] > 0]
                        edge_weights[(states[s1], states[s2])] = {
                            'weight': weight,
                            'label': '\n'.join(action_dist)
                        }
            
            # Add edges
            for (s1, s2), data in edge_weights.items():
                G.add_edge(s1, s2, **data)
            
            # Draw graph
            pos = nx.spring_layout(G, k=1, iterations=50)
            nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                                 node_size=2000, ax=ax2)
            nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
            
            # Draw edges with varying widths
            edges = G.edges()
            weights = [G[u][v]['weight']/5 for u, v in edges]
            nx.draw_networkx_edges(G, pos, width=weights, ax=ax2,
                                 edge_color='gray', arrows=True,
                                 arrowsize=20)
            
            # Add edge labels
            edge_labels = nx.get_edge_attributes(G, 'label')
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
            ax2.set_title('State Transition Graph', pad=20, fontweight='bold')
            
            # 3. Action Distribution by State
            ax3 = fig.add_subplot(gs[1, 0])
            state_action_dist = np.zeros((3, 3))  # state × action
            for s in range(3):
                total = transition_matrix[s, :, :].sum()
                if total > 0:
                    state_action_dist[s] = transition_matrix[s, :, :].sum(axis=0) / total
            
            df_state_action = pd.DataFrame(state_action_dist,
                                         index=states,
                                         columns=actions_list)
            df_state_action.plot(kind='bar', stacked=True, ax=ax3,
                               color=self.action_colors)
            ax3.set_title('Action Distribution by Current State', pad=20, fontweight='bold')
            ax3.set_xlabel('Current State')
            ax3.set_ylabel('Proportion of Actions')
            ax3.legend(title='Action Taken', bbox_to_anchor=(1.05, 1))
            
            # 4. Outcome Distribution by Action
            ax4 = fig.add_subplot(gs[1, 1])
            action_outcome_dist = np.zeros((3, 3))  # action × outcome
            for a in range(3):
                total = transition_matrix[:, :, a].sum()
                if total > 0:
                    action_outcome_dist[a] = transition_matrix[:, :, a].sum(axis=0) / total
            
            df_action_outcome = pd.DataFrame(action_outcome_dist,
                                           index=actions_list,
                                           columns=states)
            df_action_outcome.plot(kind='bar', stacked=True, ax=ax4,
                                 color=self.state_colors)
            ax4.set_title('Outcome Distribution by Action', pad=20, fontweight='bold')
            ax4.set_xlabel('Action Taken')
            ax4.set_ylabel('Proportion of Outcomes')
            ax4.legend(title='Resulting State', bbox_to_anchor=(1.05, 1))
            
            plt.suptitle('State-Action Transition Analysis', 
                        fontsize=24, fontweight='bold', y=1.02)
            fig.set_constrained_layout(True)
            
            if save_dir:
                plt.savefig(save_dir / 'state_action_analysis.png', 
                          bbox_inches='tight', dpi=300,
                          facecolor='white', edgecolor='none')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error generating state-action analysis plot: {str(e)}")
            plt.close('all')
            return None

    def plot_belief_phase_space(self, save_dir: Path = None):
        """Plot belief trajectories in phase space."""
        data = self._extract_demo_data()
        beliefs = data["beliefs"]
        posterior_beliefs = np.array([b["posterior"] for b in beliefs])
        
        fig = plt.figure(figsize=(15, 15))
        gs = GridSpec(2, 2, figure=fig)
        
        # Define state labels
        states = ['Low', 'Medium', 'High']
        
        # 1. Low vs Medium belief
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(posterior_beliefs[:, 0], posterior_beliefs[:, 1], 'b-', linewidth=2)
        scatter = ax1.scatter(posterior_beliefs[:, 0], posterior_beliefs[:, 1], 
                   c=np.arange(len(posterior_beliefs)), cmap='viridis')
        ax1.set_xlabel(f'Belief in {states[0]} State')
        ax1.set_ylabel(f'Belief in {states[1]} State')
        ax1.set_title(f'{states[0]} vs {states[1]} Belief Trajectory', pad=20)
        plt.colorbar(scatter, ax=ax1, label='Time Step')
        
        # 2. Medium vs High belief
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(posterior_beliefs[:, 1], posterior_beliefs[:, 2], 'r-', linewidth=2)
        scatter = ax2.scatter(posterior_beliefs[:, 1], posterior_beliefs[:, 2], 
                   c=np.arange(len(posterior_beliefs)), cmap='viridis')
        ax2.set_xlabel(f'Belief in {states[1]} State')
        ax2.set_ylabel(f'Belief in {states[2]} State')
        ax2.set_title(f'{states[1]} vs {states[2]} Belief Trajectory', pad=20)
        plt.colorbar(scatter, ax=ax2, label='Time Step')
        
        # 3. Low vs High belief
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(posterior_beliefs[:, 0], posterior_beliefs[:, 2], 'g-', linewidth=2)
        scatter = ax3.scatter(posterior_beliefs[:, 0], posterior_beliefs[:, 2], 
                   c=np.arange(len(posterior_beliefs)), cmap='viridis')
        ax3.set_xlabel(f'Belief in {states[0]} State')
        ax3.set_ylabel(f'Belief in {states[2]} State')
        ax3.set_title(f'{states[0]} vs {states[2]} Belief Trajectory', pad=20)
        plt.colorbar(scatter, ax=ax3, label='Time Step')
        
        # 4. 3D phase space
        ax4 = fig.add_subplot(gs[1, 1], projection='3d')
        scatter = ax4.scatter(posterior_beliefs[:, 0], 
                            posterior_beliefs[:, 1],
                            posterior_beliefs[:, 2],
                            c=np.arange(len(posterior_beliefs)),
                            cmap='viridis')
        ax4.plot(posterior_beliefs[:, 0],
                 posterior_beliefs[:, 1],
                 posterior_beliefs[:, 2],
                 'k-', alpha=0.5)
        ax4.set_xlabel(f'{states[0]} State')
        ax4.set_ylabel(f'{states[1]} State')
        ax4.set_zlabel(f'{states[2]} State')
        ax4.set_title('3D Belief Trajectory', pad=20)
        plt.colorbar(scatter, ax=ax4, label='Time Step')
        
        fig.set_constrained_layout(True)
        
        if save_dir:
            plt.savefig(save_dir / 'belief_phase_space.png', bbox_inches='tight', dpi=300)
        
        return fig

    def plot_value_components(self, save_dir: Path = None):
        """Plot epistemic and pragmatic value components analysis."""
        try:
            data = self._extract_demo_data()
            steps = data["steps"]
            
            if len(steps) == 0:
                logger.warning("No time steps found in data")
                return None
            
            # Convert and validate value data
            try:
                epistemic_values = data["epistemic_values"]
                pragmatic_values = data["pragmatic_values"]
                
                if epistemic_values.size == 0 or pragmatic_values.size == 0:
                    logger.warning("Empty value arrays found")
                    return None
                    
                if len(epistemic_values.shape) != 2 or len(pragmatic_values.shape) != 2:
                    logger.warning("Invalid value array shapes")
                    return None
                    
            except Exception as e:
                logger.error(f"Error accessing value data: {str(e)}")
                return None
            
            # Create figure
            fig = plt.figure(figsize=(15, 15))
            gs = GridSpec(3, 2, height_ratios=[2, 2, 1], figure=fig)
            
            # Define action labels and colors
            actions = ['Decrease', 'Stay', 'Increase']
            colors = ['#f1c40f', '#9b59b6', '#1abc9c']  # Yellow, Purple, Turquoise
            
            try:
                # 1. Epistemic Values by Action
                ax1 = fig.add_subplot(gs[0, 0])
                for i in range(epistemic_values.shape[1]):
                    ax1.plot(steps, epistemic_values[:, i], 
                            label=actions[i], color=colors[i],
                            marker='o', markersize=4, linewidth=2)
                ax1.set_title('Epistemic Value by Action', pad=20)
                ax1.set_xlabel('Time Step')
                ax1.set_ylabel('Epistemic Value')
                ax1.legend(title='Actions')
                
                # 2. Pragmatic Values by Action
                ax2 = fig.add_subplot(gs[0, 1])
                for i in range(pragmatic_values.shape[1]):
                    ax2.plot(steps, pragmatic_values[:, i], 
                            label=actions[i], color=colors[i],
                            marker='o', markersize=4, linewidth=2)
                ax2.set_title('Pragmatic Value by Action', pad=20)
                ax2.set_xlabel('Time Step')
                ax2.set_ylabel('Pragmatic Value')
                ax2.legend(title='Actions')
                
                # 3. Total Values (Epistemic + Pragmatic)
                ax3 = fig.add_subplot(gs[1, 0])
                total_values = -epistemic_values + pragmatic_values  # Same combination as in G
                for i in range(total_values.shape[1]):
                    ax3.plot(steps, total_values[:, i], 
                            label=actions[i], color=colors[i],
                            marker='o', markersize=4, linewidth=2)
                ax3.set_title('Combined Value (G) by Action', pad=20)
                ax3.set_xlabel('Time Step')
                ax3.set_ylabel('Combined Value')
                ax3.legend(title='Actions')
                
                # 4. Value Component Ratios
                ax4 = fig.add_subplot(gs[1, 1])
                eps = 1e-10  # Small constant to avoid division by zero
                ratios = np.abs(epistemic_values) / (np.abs(epistemic_values) + np.abs(pragmatic_values) + eps)
                for i in range(ratios.shape[1]):
                    ax4.plot(steps, ratios[:, i], 
                            label=actions[i], color=colors[i],
                            marker='o', markersize=4, linewidth=2)
                ax4.set_title('Epistemic/Pragmatic Value Ratio', pad=20)
                ax4.set_xlabel('Time Step')
                ax4.set_ylabel('Epistemic Value Ratio')
                ax4.legend(title='Actions')
                
                # 5. Average Values Comparison
                ax5 = fig.add_subplot(gs[2, :])
                x = np.arange(len(actions))
                width = 0.35
                
                avg_epistemic = np.mean(epistemic_values, axis=0)
                avg_pragmatic = np.mean(pragmatic_values, axis=0)
                
                ax5.bar(x - width/2, avg_epistemic, width, label='Avg Epistemic', color='blue', alpha=0.6)
                ax5.bar(x + width/2, avg_pragmatic, width, label='Avg Pragmatic', color='red', alpha=0.6)
                ax5.set_title('Average Value Components by Action', pad=20)
                ax5.set_xticks(x)
                ax5.set_xticklabels(actions)
                ax5.legend()
                
                fig.set_constrained_layout(True)
                
                if save_dir:
                    plt.savefig(save_dir / 'value_components.png', bbox_inches='tight', dpi=300)
                
                return fig
                
            except Exception as e:
                logger.error(f"Error plotting value components: {str(e)}")
                plt.close(fig)
                return None
                
        except Exception as e:
            logger.error(f"Error generating value components plot: {str(e)}")
            plt.close('all')
            return None

    def plot_summary_dashboard(self, save_dir: Path = None):
        """Create and save a comprehensive dashboard of all visualizations."""
        try:
            data = self._extract_demo_data()
            steps = data["steps"]
            
            if len(steps) == 0:
                logger.warning("No time steps found in data")
                return None
            
            fig = plt.figure(figsize=(20, 24))
            gs = GridSpec(4, 2, height_ratios=[2, 2, 2, 1], figure=fig)
            
            # Define consistent state and action orders
            states = ['Low', 'Medium', 'High']
            actions = ['Decrease', 'Stay', 'Increase']
            
            # Define color scheme
            state_colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
            action_colors = ['#f1c40f', '#9b59b6', '#1abc9c']  # Yellow, Purple, Turquoise
            
            # 1. Belief Evolution with enhanced styling
            ax1 = fig.add_subplot(gs[0, 0])
            beliefs = data["beliefs"]
            posterior_beliefs = np.array([b["posterior"] for b in beliefs])
            for i in range(posterior_beliefs.shape[1]):
                ax1.plot(steps, posterior_beliefs[:, i], 
                        label=states[i], 
                        color=state_colors[i],
                        marker='o', markersize=6, linewidth=2)
            ax1.set_title('Belief Evolution', pad=20, fontweight='bold')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Probability')
            ax1.legend(title='States', title_fontsize=12)
            ax1.set_ylim(-0.05, 1.05)  # Add padding to y-axis
            
            # 2. Policy Evolution with enhanced styling
            ax2 = fig.add_subplot(gs[0, 1])
            policy_posteriors = data["policy_posteriors"]
            for i in range(policy_posteriors.shape[1]):
                ax2.plot(steps, policy_posteriors[:, i], 
                        label=actions[i], 
                        color=action_colors[i],
                        marker='o', markersize=6, linewidth=2)
            ax2.set_title('Policy Evolution', pad=20, fontweight='bold')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Probability')
            ax2.legend(title='Actions', title_fontsize=12)
            ax2.set_ylim(-0.05, 1.05)  # Add padding to y-axis
            
            # 3. Free Energies with enhanced styling
            ax3 = fig.add_subplot(gs[1, 0])
            free_energies = data["free_energies"]
            vfe = np.array([fe["variational"] for fe in free_energies])
            for i in range(vfe.shape[1]):
                ax3.plot(steps, vfe[:, i], 
                        label=states[i], 
                        color=state_colors[i],
                        marker='o', markersize=6, linewidth=2)
            ax3.set_title('Variational Free Energy', pad=20, fontweight='bold')
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('Free Energy')
            ax3.legend(title='States', title_fontsize=12)
            
            # 4. Expected Free Energy with enhanced styling
            ax4 = fig.add_subplot(gs[1, 1])
            efe = np.array([fe["expected"] for fe in free_energies])
            for i in range(efe.shape[1]):
                ax4.plot(steps, efe[:, i], 
                        label=actions[i], 
                        color=action_colors[i],
                        marker='o', markersize=6, linewidth=2)
            ax4.set_title('Expected Free Energy', pad=20, fontweight='bold')
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Free Energy')
            ax4.legend(title='Actions', title_fontsize=12)
            
            # 5. State-Action Heatmap with enhanced styling
            ax5 = fig.add_subplot(gs[2, :])
            transitions = pd.DataFrame({
                'From': [states[o] for o in data["observations"][:-1]],
                'To': [states[o] for o in data["observations"][1:]],
                'Action': [actions[a] for a in data["actions"][:-1]]
            })
            heatmap_data = pd.crosstab([transitions['From'], transitions['Action']], 
                                      transitions['To'])
            heatmap_data = heatmap_data.reindex(index=pd.MultiIndex.from_product([states, actions]),
                                               columns=states)
            heatmap_data = heatmap_data.fillna(0)  # Replace NaN with 0
            sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', ax=ax5, fmt='.0f',
                       cbar_kws={'label': 'Transition Count'},
                       annot_kws={'size': 10, 'weight': 'bold'})
            ax5.set_title('State-Action Transition Heatmap', pad=20, fontweight='bold')
            
            # 6. Action and Observation Sequences with enhanced styling
            ax6 = fig.add_subplot(gs[3, 0])
            ax6.plot(steps, data["actions"], color='#2980b9', marker='o', 
                    markersize=8, linewidth=2, markerfacecolor='white')
            ax6.set_yticks(range(len(actions)))
            ax6.set_yticklabels(actions)
            ax6.set_title('Action Sequence', pad=20, fontweight='bold')
            ax6.set_xlabel('Time Step')
            # Add background colors for different actions
            ax6.set_ylim(-0.5, 2.5)
            for i in range(len(actions)):
                ax6.axhspan(i-0.5, i+0.5, color=action_colors[i], alpha=0.1)
            
            ax7 = fig.add_subplot(gs[3, 1])
            ax7.plot(steps, data["observations"], color='#c0392b', marker='o',
                    markersize=8, linewidth=2, markerfacecolor='white')
            ax7.set_yticks(range(len(states)))
            ax7.set_yticklabels(states)
            ax7.set_title('Observation Sequence', pad=20, fontweight='bold')
            ax7.set_xlabel('Time Step')
            # Add background colors for different states
            ax7.set_ylim(-0.5, 2.5)
            for i in range(len(states)):
                ax7.axhspan(i-0.5, i+0.5, color=state_colors[i], alpha=0.1)
            
            plt.suptitle('Active Inference Analysis Dashboard', 
                        fontsize=24, fontweight='bold', y=1.02)
            fig.set_constrained_layout(True)
            
            if save_dir:
                plt.savefig(save_dir / 'summary_dashboard.png', 
                           bbox_inches='tight', dpi=300,
                           facecolor='white', edgecolor='none')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error generating summary dashboard: {str(e)}")
            plt.close('all')
            return None

    def plot_exploration_exploitation(self, save_dir: Path = None):
        """Plot comprehensive analysis of exploration-exploitation dynamics."""
        try:
            data = self._extract_demo_data()
            steps = data["steps"]
            
            if len(steps) == 0:
                logger.warning("No time steps found in data")
                return None
            
            # Get required data
            epistemic_values = data["epistemic_values"]
            pragmatic_values = data["pragmatic_values"]
            policy_posteriors = data["policy_posteriors"]
            policy_priors = data["policy_priors"]
            actions = data["actions"]
            observations = data["observations"]
            
            # Create figure
            fig = plt.figure(figsize=(20, 24))
            gs = GridSpec(5, 2, height_ratios=[2, 2, 2, 2, 1], figure=fig)
            
            # Define colors and styles
            action_colors = ['#f1c40f', '#9b59b6', '#1abc9c']  # Yellow, Purple, Turquoise
            actions_list = ['Decrease', 'Stay', 'Increase']
            cmap = plt.cm.viridis
            
            # 1. Epistemic vs Pragmatic Value Scatter with Time Evolution
            ax1 = fig.add_subplot(gs[0, 0])
            for i in range(len(actions_list)):
                mask = actions == i
                scatter = ax1.scatter(epistemic_values[mask, i], pragmatic_values[mask, i],
                                    c=steps[mask], cmap=cmap, label=actions_list[i],
                                    alpha=0.7, s=100)
            ax1.set_xlabel('Epistemic Value')
            ax1.set_ylabel('Pragmatic Value')
            ax1.set_title('Epistemic vs Pragmatic Values by Action', pad=20, fontweight='bold')
            ax1.legend(title='Selected Action', title_fontsize=12)
            plt.colorbar(scatter, ax=ax1, label='Time Step')
            
            # 2. Policy Uncertainty Analysis
            ax2 = fig.add_subplot(gs[0, 1])
            # Compute different uncertainty metrics
            policy_entropy = -np.sum(policy_posteriors * np.log(policy_posteriors + 1e-10), axis=1)
            policy_prior_entropy = -np.sum(policy_priors * np.log(policy_priors + 1e-10), axis=1)
            kl_divergence = np.sum(policy_posteriors * (np.log(policy_posteriors + 1e-10) - 
                                                      np.log(policy_priors + 1e-10)), axis=1)
            
            ax2.plot(steps, policy_entropy, 'b-', linewidth=2, label='Policy Entropy')
            ax2.plot(steps, policy_prior_entropy, 'g--', linewidth=2, label='Prior Entropy')
            ax2.plot(steps, kl_divergence, 'r:', linewidth=2, label='KL Divergence')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Uncertainty')
            ax2.set_title('Policy Uncertainty Metrics', pad=20, fontweight='bold')
            ax2.legend(title='Metrics', title_fontsize=12)
            
            # 3. Exploration-Exploitation Balance
            ax3 = fig.add_subplot(gs[1, :])
            window_size = 10
            eps = 1e-10
            
            # Compute smoothed ratios
            exploration_ratio = np.abs(epistemic_values).mean(axis=1)
            exploitation_ratio = np.abs(pragmatic_values).mean(axis=1)
            total = exploration_ratio + exploitation_ratio + eps
            exploration_ratio /= total
            exploitation_ratio /= total
            
            # Apply rolling average
            def rolling_mean(x, w):
                return np.convolve(x, np.ones(w), 'valid') / w
                
            smooth_explore = rolling_mean(exploration_ratio, window_size)
            smooth_exploit = rolling_mean(exploitation_ratio, window_size)
            plot_steps = steps[window_size-1:]
            
            # Create stacked area plot
            ax3.fill_between(plot_steps, 0, smooth_explore, 
                           color='blue', alpha=0.3, label='Exploration')
            ax3.fill_between(plot_steps, smooth_explore, 
                           smooth_explore + smooth_exploit,
                           color='red', alpha=0.3, label='Exploitation')
            
            ax3.plot(plot_steps, smooth_explore, 'b-', linewidth=2)
            ax3.plot(plot_steps, smooth_exploit, 'r-', linewidth=2)
            
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('Ratio')
            ax3.set_title(f'Exploration-Exploitation Balance ({window_size}-step moving average)', 
                         pad=20, fontweight='bold')
            ax3.legend(title='Components', title_fontsize=12)
            
            # 4. State-Action Exploration Patterns
            ax4 = fig.add_subplot(gs[2, 0])
            state_counts = np.zeros((3, 3))  # states × actions
            for s, a in zip(observations, actions):
                state_counts[int(s), int(a)] += 1
            
            sns.heatmap(state_counts, annot=True, fmt='.0f',
                       xticklabels=actions_list,
                       yticklabels=['Low', 'Medium', 'High'],
                       cmap='YlOrRd', ax=ax4)
            ax4.set_xlabel('Selected Action')
            ax4.set_ylabel('Current State')
            ax4.set_title('State-Dependent Action Selection Patterns', pad=20, fontweight='bold')
            
            # 5. Temporal Action Preference
            ax5 = fig.add_subplot(gs[2, 1])
            window_size = 10
            action_prefs = np.zeros((len(steps) - window_size + 1, len(actions_list)))
            
            for i in range(len(steps) - window_size + 1):
                window = actions[i:i+window_size]
                for a in range(len(actions_list)):
                    action_prefs[i, a] = np.mean(window == a)
            
            for i, action in enumerate(actions_list):
                ax5.plot(steps[window_size-1:], action_prefs[:, i],
                        label=action, color=action_colors[i], linewidth=2)
            
            ax5.set_title(f'Temporal Action Preferences ({window_size}-step window)', 
                         pad=20, fontweight='bold')
            ax5.set_xlabel('Time Step')
            ax5.set_ylabel('Action Preference')
            ax5.legend(title='Actions', title_fontsize=12)
            
            plt.suptitle('Exploration-Exploitation Analysis Dashboard', 
                        fontsize=24, fontweight='bold', y=1.02)
            fig.set_constrained_layout(True)
            
            if save_dir:
                plt.savefig(save_dir / 'exploration_exploitation.png', 
                          bbox_inches='tight', dpi=300,
                          facecolor='white', edgecolor='none')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error generating exploration-exploitation plot: {str(e)}")
            plt.close('all')
            return None

    def plot_action_selection_dynamics(self, save_dir: Path = None):
        """Plot detailed analysis of action selection dynamics."""
        try:
            data = self._extract_demo_data()
            steps = data["steps"]
            
            if len(steps) == 0:
                logger.warning("No time steps found in data")
                return None
            
            # Get required data
            policy_posteriors = data["policy_posteriors"]
            policy_priors = data["policy_priors"]
            actions = data["actions"]
            observations = data["observations"]
            epistemic_values = data["epistemic_values"]
            pragmatic_values = data["pragmatic_values"]
            
            # Create figure
            fig = plt.figure(figsize=(20, 24))
            gs = GridSpec(5, 2, height_ratios=[2, 2, 2, 2, 1], figure=fig)
            
            # Define colors and styles
            action_colors = ['#f1c40f', '#9b59b6', '#1abc9c']  # Yellow, Purple, Turquoise
            actions_list = ['Decrease', 'Stay', 'Increase']
            cmap = plt.cm.viridis
            
            # 1. Action Selection Probability Evolution
            ax1 = fig.add_subplot(gs[0, :])
            for i in range(policy_posteriors.shape[1]):
                ax1.plot(steps, policy_posteriors[:, i], 
                        label=actions_list[i], color=action_colors[i],
                        marker='o', markersize=4, linewidth=2)
                # Add shading for selected actions
                selected_mask = actions == i
                ax1.fill_between(steps, 0, policy_posteriors[:, i], 
                               where=selected_mask, color=action_colors[i], alpha=0.2)
            ax1.set_title('Action Selection Probabilities with Chosen Actions', pad=20, fontweight='bold')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Selection Probability')
            ax1.legend(title='Actions', title_fontsize=12)
            ax1.set_ylim(-0.05, 1.05)
            
            # 2. Decision Confidence Analysis
            ax2 = fig.add_subplot(gs[1, 0])
            max_probs = np.max(policy_posteriors, axis=1)
            prob_margins = np.sort(policy_posteriors, axis=1)[:, -1] - np.sort(policy_posteriors, axis=1)[:, -2]
            
            ax2.plot(steps, max_probs, 'b-', label='Highest Probability', linewidth=2)
            ax2.plot(steps, prob_margins, 'r--', label='Decision Margin', linewidth=2)
            ax2.set_title('Decision Confidence Metrics', pad=20, fontweight='bold')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Probability')
            ax2.legend(title='Metrics', title_fontsize=12)
            
            # 3. Action Switching Analysis
            ax3 = fig.add_subplot(gs[1, 1])
            action_switches = np.diff(actions) != 0
            switch_points = np.where(action_switches)[0] + 1
            
            # Plot action sequence
            ax3.plot(steps, actions, 'k-', linewidth=2, label='Action Sequence')
            # Highlight switch points
            ax3.scatter(steps[switch_points], actions[switch_points], 
                       color='red', s=100, label='Action Switches')
            
            ax3.set_yticks(range(len(actions_list)))
            ax3.set_yticklabels(actions_list)
            ax3.set_title('Action Switching Pattern', pad=20, fontweight='bold')
            ax3.set_xlabel('Time Step')
            ax3.legend(title='Events', title_fontsize=12)
            
            # 4. Value-Based Decision Analysis
            ax4 = fig.add_subplot(gs[2, 0])
            selected_total_values = np.zeros(len(steps))
            for i, (step, action) in enumerate(zip(steps, actions)):
                selected_total_values[i] = -epistemic_values[i, action] + pragmatic_values[i, action]
            
            # Plot total value of selected actions
            ax4.plot(steps, selected_total_values, 'g-', linewidth=2, label='Selected Action Value')
            # Add scatter points colored by action
            for i, action in enumerate(actions_list):
                mask = actions == i
                ax4.scatter(steps[mask], selected_total_values[mask], 
                          color=action_colors[i], label=action, s=100)
            
            ax4.set_title('Value of Selected Actions', pad=20, fontweight='bold')
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Total Value (G)')
            ax4.legend(title='Actions', title_fontsize=12)
            
            # 5. Decision Threshold Analysis
            ax5 = fig.add_subplot(gs[2, 1])
            # Compute value differences between best and second-best actions
            value_diffs = []
            total_values = -epistemic_values + pragmatic_values
            for values in total_values:
                sorted_values = np.sort(values)
                value_diffs.append(sorted_values[-1] - sorted_values[-2])
            
            ax5.plot(steps, value_diffs, 'b-', linewidth=2, label='Value Difference')
            # Add threshold reference line
            ax5.axhline(y=np.mean(value_diffs), color='r', linestyle='--', 
                       label='Average Threshold')
            
            ax5.set_title('Decision Value Differences', pad=20, fontweight='bold')
            ax5.set_xlabel('Time Step')
            ax5.set_ylabel('Value Difference')
            ax5.legend(title='Metrics', title_fontsize=12)
            
            # 6. State-Dependent Action Selection
            ax6 = fig.add_subplot(gs[3, :])
            state_action_probs = np.zeros((3, 3))  # states × actions
            state_counts = np.zeros(3)
            
            for obs, act in zip(observations, actions):
                state_action_probs[int(obs), int(act)] += 1
                state_counts[int(obs)] += 1
            
            # Normalize by state counts
            for s in range(3):
                if state_counts[s] > 0:
                    state_action_probs[s, :] /= state_counts[s]
            
            sns.heatmap(state_action_probs, annot=True, fmt='.2f',
                       xticklabels=actions_list,
                       yticklabels=['Low', 'Medium', 'High'],
                       cmap='YlOrRd', ax=ax6)
            ax6.set_title('State-Dependent Action Selection Probabilities', pad=20, fontweight='bold')
            ax6.set_xlabel('Selected Action')
            ax6.set_ylabel('Current State')
            
            # 7. Temporal Action Preference
            ax7 = fig.add_subplot(gs[4, :])
            window_size = 10
            action_prefs = np.zeros((len(steps) - window_size + 1, len(actions_list)))
            
            for i in range(len(steps) - window_size + 1):
                window = actions[i:i+window_size]
                for a in range(len(actions_list)):
                    action_prefs[i, a] = np.mean(window == a)
            
            for i, action in enumerate(actions_list):
                ax7.plot(steps[window_size-1:], action_prefs[:, i],
                        label=action, color=action_colors[i], linewidth=2)
            
            ax7.set_title(f'Temporal Action Preferences ({window_size}-step window)', 
                         pad=20, fontweight='bold')
            ax7.set_xlabel('Time Step')
            ax7.set_ylabel('Action Preference')
            ax7.legend(title='Actions', title_fontsize=12)
            
            plt.suptitle('Action Selection Dynamics Analysis', 
                        fontsize=24, fontweight='bold', y=1.02)
            fig.set_constrained_layout(True)
            
            if save_dir:
                plt.savefig(save_dir / 'action_selection_dynamics.png', 
                          bbox_inches='tight', dpi=300,
                          facecolor='white', edgecolor='none')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error generating action selection dynamics plot: {str(e)}")
            plt.close('all')
            return None

def setup_directories() -> tuple[Path, Path]:
    """Set up the necessary directories for input and output."""
    # Get the current directory (where the script is located)
    current_dir = Path(__file__).parent
    
    # Set up logs and visualizations directories
    logs_dir = current_dir / 'logs'
    vis_dir = current_dir / 'visualizations'
    
    # Create directories if they don't exist
    logs_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)
    
    return logs_dir, vis_dir

def main():
    """Main function to demonstrate visualization capabilities."""
    try:
        # Set up directories
        logs_dir, vis_dir = setup_directories()
        logger.info(f"Set up directories - logs: {logs_dir}, visualizations: {vis_dir}")
        
        # Find all log files
        log_files = list(logs_dir.glob("active_inference_demo_*.json"))
        if not log_files:
            logger.error(f"No log files found in {logs_dir}")
            return
            
        # Sort by modification time and print available files
        log_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        logger.info("Available log files:")
        for i, f in enumerate(log_files):
            mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"{i+1}. {f.name} (modified: {mtime})")
            
        # Select most recent file
        latest_log = log_files[0]
        logger.info(f"\nProcessing most recent log file: {latest_log}")
        
        # Create timestamp-based subdirectory for this visualization run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = vis_dir / f"run_{timestamp}"
        run_dir.mkdir(exist_ok=True)
        logger.info(f"Created visualization directory: {run_dir}")
        
        # Create visualizer
        viz = ActiveInferenceVisualizer(latest_log)
        
        # Generate and save all visualizations
        logger.info("Generating and saving visualizations...")
        
        # Define visualization functions with their filenames
        files_to_generate = [
            ('belief_evolution.png', viz.plot_belief_evolution),
            ('policy_evolution.png', viz.plot_policy_evolution),
            ('free_energy_analysis.png', viz.plot_free_energy_components),
            ('state_action_analysis.png', viz.plot_state_action_analysis),
            ('belief_phase_space.png', viz.plot_belief_phase_space),
            ('value_components.png', viz.plot_value_components),
            ('summary_dashboard.png', viz.plot_summary_dashboard),
            ('exploration_exploitation.png', viz.plot_exploration_exploitation),
            ('action_selection_dynamics.png', viz.plot_action_selection_dynamics)
        ]
        
        successful_plots = 0
        for filename, plot_func in files_to_generate:
            try:
                filepath = run_dir / filename
                fig = plot_func(run_dir)
                
                if fig is not None:
                    # Ensure the figure is properly saved
                    try:
                        fig.savefig(filepath, bbox_inches='tight', dpi=300)
                        plt.close(fig)
                        logger.info(f"Successfully generated and saved: {filepath}")
                        successful_plots += 1
                    except Exception as e:
                        logger.error(f"Error saving {filename}: {str(e)}")
                else:
                    logger.warning(f"Failed to generate {filename} - no figure returned")
            except Exception as e:
                logger.error(f"Error generating {filename}: {str(e)}")
            finally:
                plt.close('all')  # Ensure all figures are closed
        
        # Log summary
        total_plots = len(files_to_generate)
        logger.info(f"\nVisualization complete. Successfully generated {successful_plots}/{total_plots} plots.")
        if successful_plots > 0:
            logger.info(f"Visualization directory: {run_dir}")
        else:
            logger.warning("No plots were successfully generated.")
        
    except Exception as e:
        logger.error(f"Error in visualization process: {str(e)}")
    finally:
        plt.close('all')  # Ensure all figures are closed

if __name__ == "__main__":
    main() 