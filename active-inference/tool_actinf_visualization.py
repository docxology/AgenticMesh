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

class ActiveInferenceVisualizer:
    """Visualizer for Active Inference agent data."""
    
    def __init__(self, log_file: str):
        """Initialize visualizer with log file path."""
        self.log_file = Path(log_file)
        self.data = self._load_data()
        self.setup_style()
        
    def setup_style(self):
        """Configure plot styling."""
        plt.style.use('default')
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")
        
        # Enhanced formatting with better readability
        plt.rcParams.update({
            'figure.figsize': [12, 8],
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'legend.title_fontsize': 14,
            'figure.titlesize': 18,
            'figure.dpi': 300,
            'figure.constrained_layout.use': True,  # Use constrained_layout instead of tight_layout
            'axes.spines.top': False,  # Remove top spine
            'axes.spines.right': False,  # Remove right spine
            'axes.grid': True,
            'grid.alpha': 0.3
        })

    def _load_data(self) -> List[Dict]:
        """Load and parse JSON log file."""
        data = []
        with open(self.log_file) as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # Skip invalid JSON lines
        return data
    
    def _extract_demo_data(self):
        """Extract relevant data from log entries."""
        steps = []
        beliefs = []
        policy_priors = []
        policy_posteriors = []
        free_energies = []
        actions = []
        observations = []
        
        for entry in self.data:
            if entry.get("event") == "agent_step":
                step_data = entry.get("json_data", {}) if "json_data" in entry else entry
                
                # Extract step number
                steps.append(step_data.get("step", 0))
                
                # Extract beliefs
                belief_data = step_data.get("beliefs", {})
                beliefs.append({
                    "prior": [float(x) for x in belief_data.get("prior", [])],
                    "posterior": [float(x) for x in belief_data.get("posterior", [])]
                })
                
                # Extract policy information
                policy_data = step_data.get("policy", {})
                policy_priors.append([float(x) for x in policy_data.get("prior", [])])
                policy_posteriors.append([float(x) for x in policy_data.get("posterior", [])])
                
                # Extract free energies
                free_energy_data = step_data.get("free_energy", {})
                free_energies.append({
                    "variational": [float(x) for x in free_energy_data.get("variational", [])],
                    "expected": [float(x) for x in free_energy_data.get("expected", [])]
                })
                
                # Extract actions and observations
                action_data = step_data.get("action", {})
                actions.append(action_data.get("name", ""))
                
                obs_data = step_data.get("observation", {})
                observations.append(obs_data.get("name", ""))
        
        return {
            "steps": steps,
            "beliefs": beliefs,
            "policy_priors": policy_priors,
            "policy_posteriors": policy_posteriors,
            "free_energies": free_energies,
            "actions": actions,
            "observations": observations
        }
    
    def plot_belief_evolution(self, save_dir: Path = None):
        """Plot the evolution of beliefs over time with enhanced formatting."""
        data = self._extract_demo_data()
        steps = data["steps"]
        beliefs = data["beliefs"]
        
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 1, height_ratios=[2, 2, 1], figure=fig)
        
        # Plot prior beliefs
        ax1 = fig.add_subplot(gs[0])
        prior_beliefs = np.array([b["prior"] for b in beliefs])
        for i in range(prior_beliefs.shape[1]):
            ax1.plot(steps, prior_beliefs[:, i], 
                    label=f'State {["Low", "Medium", "High"][i]}', 
                    marker='o', markersize=4, linewidth=2)
        ax1.set_title('Prior Beliefs Evolution', pad=20)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Prior Belief Probability')
        ax1.legend(title='States', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Plot posterior beliefs
        ax2 = fig.add_subplot(gs[1])
        posterior_beliefs = np.array([b["posterior"] for b in beliefs])
        for i in range(posterior_beliefs.shape[1]):
            ax2.plot(steps, posterior_beliefs[:, i], 
                    label=f'State {["Low", "Medium", "High"][i]}', 
                    marker='o', markersize=4, linewidth=2)
        ax2.set_title('Posterior Beliefs Evolution', pad=20)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Posterior Belief Probability')
        ax2.legend(title='States', bbox_to_anchor=(1.02, 1), loc='upper left')
        
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
    
    def plot_policy_evolution(self, save_dir: Path = None):
        """Plot the evolution of policy priors and posteriors with enhanced formatting."""
        data = self._extract_demo_data()
        steps = data["steps"]
        
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 1, height_ratios=[2, 2, 1], figure=fig)
        
        # Plot policy priors
        ax1 = fig.add_subplot(gs[0])
        policy_priors = np.array(data["policy_priors"])
        for i in range(policy_priors.shape[1]):
            ax1.plot(steps, policy_priors[:, i], 
                    label=f'Action {["Decrease", "Stay", "Increase"][i]}', 
                    marker='o', markersize=4, linewidth=2)
        ax1.set_title('Policy Prior Evolution', pad=20)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Prior Probability')
        ax1.legend(title='Actions', bbox_to_anchor=(1.02, 1), loc='upper left')
        
        # Plot policy posteriors
        ax2 = fig.add_subplot(gs[1])
        policy_posteriors = np.array(data["policy_posteriors"])
        for i in range(policy_posteriors.shape[1]):
            ax2.plot(steps, policy_posteriors[:, i], 
                    label=f'Action {["Decrease", "Stay", "Increase"][i]}', 
                    marker='o', markersize=4, linewidth=2)
        ax2.set_title('Policy Posterior Evolution', pad=20)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Posterior Probability')
        ax2.legend(title='Actions', bbox_to_anchor=(1.02, 1), loc='upper left')
        
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

    def plot_free_energy_components(self, save_dir: Path = None):
        """Plot detailed free energy analysis."""
        data = self._extract_demo_data()
        steps = data["steps"]
        free_energies = data["free_energies"]
        
        fig = plt.figure(figsize=(15, 15))
        gs = GridSpec(3, 2, height_ratios=[2, 2, 1], figure=fig)
        
        # Plot variational free energy
        ax1 = fig.add_subplot(gs[0, 0])
        vfe = np.array([fe["variational"] for fe in free_energies])
        for i in range(vfe.shape[1]):
            ax1.plot(steps, vfe[:, i], 
                    label=f'State {["Low", "Medium", "High"][i]}', 
                    marker='o', markersize=4, linewidth=2)
        ax1.set_title('Variational Free Energy by State', pad=20)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Free Energy')
        ax1.legend(title='States')
        
        # Plot expected free energy
        ax2 = fig.add_subplot(gs[0, 1])
        efe = np.array([fe["expected"] for fe in free_energies])
        for i in range(efe.shape[1]):
            ax2.plot(steps, efe[:, i], 
                    label=f'Action {["Decrease", "Stay", "Increase"][i]}', 
                    marker='o', markersize=4, linewidth=2)
        ax2.set_title('Expected Free Energy by Action', pad=20)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Free Energy')
        ax2.legend(title='Actions')
        
        # Plot cumulative free energy
        ax3 = fig.add_subplot(gs[1, 0])
        cum_vfe = np.cumsum(vfe.mean(axis=1))
        ax3.plot(steps, cum_vfe, 'k-', linewidth=2, label='Cumulative VFE')
        ax3.set_title('Cumulative Variational Free Energy', pad=20)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Cumulative Free Energy')
        ax3.legend()
        
        # Plot free energy differences
        ax4 = fig.add_subplot(gs[1, 1])
        vfe_diff = np.diff(vfe.mean(axis=1))
        ax4.bar(steps[1:], vfe_diff, alpha=0.6, color='purple')
        ax4.set_title('Free Energy Changes Between Steps', pad=20)
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Free Energy Difference')
        
        # Plot average free energies
        ax5 = fig.add_subplot(gs[2, :])
        avg_vfe = vfe.mean(axis=0)
        avg_efe = efe.mean(axis=0)
        x = np.arange(len(avg_vfe))
        width = 0.35
        ax5.bar(x - width/2, avg_vfe, width, label='Avg VFE', color='blue', alpha=0.6)
        ax5.bar(x + width/2, avg_efe, width, label='Avg EFE', color='red', alpha=0.6)
        ax5.set_title('Average Free Energies', pad=20)
        ax5.set_xticks(x)
        ax5.set_xticklabels(['Low', 'Medium', 'High'])
        ax5.legend()
        
        fig.set_constrained_layout(True)
        
        if save_dir:
            plt.savefig(save_dir / 'free_energy_analysis.png', bbox_inches='tight', dpi=300)
        
        return fig

    def plot_state_action_analysis(self, save_dir: Path = None):
        """Create comprehensive state-action transition analysis."""
        data = self._extract_demo_data()
        actions = data["actions"]
        observations = data["observations"]
        
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(2, 2, figure=fig)
        
        # Define consistent state and action orders
        state_order = ['Low', 'Medium', 'High']
        action_order = ['Decrease', 'Stay', 'Increase']
        
        # 1. Transition Heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        transitions = pd.DataFrame({
            'From': observations[:-1],
            'To': observations[1:],
            'Action': actions[:-1]
        })
        
        # Create raw counts heatmap
        heatmap_data = pd.crosstab([transitions['From'], transitions['Action']], 
                                  transitions['To'])
        heatmap_data = heatmap_data.reindex(index=pd.MultiIndex.from_product([state_order, action_order]),
                                           columns=state_order)
        heatmap_data = heatmap_data.fillna(0)  # Replace NaN with 0
        sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', ax=ax1, fmt='.0f',
                   cbar_kws={'label': 'Transition Count'})
        ax1.set_title('State-Action Transition Heatmap', pad=20)
        
        # 2. State Transition Graph
        ax2 = fig.add_subplot(gs[0, 1])
        G = nx.DiGraph()
        for i in range(len(observations)-1):
            G.add_edge(observations[i], observations[i+1], 
                      action=actions[i])
        pos = nx.spring_layout(G, k=1, iterations=50)
        nx.draw(G, pos, ax=ax2, with_labels=True, node_color='lightblue',
                node_size=2000, font_size=12, font_weight='bold')
        edge_labels = nx.get_edge_attributes(G, 'action')
        nx.draw_networkx_edge_labels(G, pos, edge_labels)
        ax2.set_title('State Transition Graph', pad=20)
        
        # 3. Action Distribution by State
        ax3 = fig.add_subplot(gs[1, 0])
        action_by_state = pd.crosstab(transitions['From'], transitions['Action'], 
                                     normalize='index')
        action_by_state = action_by_state.reindex(index=state_order, columns=action_order)
        action_by_state = action_by_state.fillna(0)  # Replace NaN with 0
        action_by_state.plot(kind='bar', stacked=True, ax=ax3)
        ax3.set_title('Action Distribution by Current State', pad=20)
        ax3.set_xlabel('Current State')
        ax3.set_ylabel('Proportion of Actions')
        ax3.legend(title='Action Taken')
        
        # 4. Outcome Distribution by Action
        ax4 = fig.add_subplot(gs[1, 1])
        outcome_by_action = pd.crosstab(transitions['Action'], transitions['To'], 
                                       normalize='index')
        outcome_by_action = outcome_by_action.reindex(index=action_order, columns=state_order)
        outcome_by_action = outcome_by_action.fillna(0)  # Replace NaN with 0
        outcome_by_action.plot(kind='bar', stacked=True, ax=ax4)
        ax4.set_title('Outcome Distribution by Action', pad=20)
        ax4.set_xlabel('Action Taken')
        ax4.set_ylabel('Proportion of Outcomes')
        ax4.legend(title='Resulting State')
        
        fig.set_constrained_layout(True)
        
        if save_dir:
            plt.savefig(save_dir / 'state_action_analysis.png', bbox_inches='tight', dpi=300)
        
        return fig

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

    def plot_summary_dashboard(self, save_dir: Path = None):
        """Create and save a comprehensive dashboard of all visualizations."""
        data = self._extract_demo_data()
        steps = data["steps"]
        
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
        policy_posteriors = np.array(data["policy_posteriors"])
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
            'From': data["observations"][:-1],
            'To': data["observations"][1:],
            'Action': data["actions"][:-1]
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
        action_indices = [actions.index(a) for a in data["actions"]]
        ax6.plot(steps, action_indices, color='#2980b9', marker='o', 
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
        observation_indices = [states.index(o) for o in data["observations"]]
        ax7.plot(steps, observation_indices, color='#c0392b', marker='o',
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
    # Set up directories
    logs_dir, vis_dir = setup_directories()
    
    # Find the most recent log file
    log_files = list(logs_dir.glob("active_inference_demo_*.json"))
    if not log_files:
        print(f"No log files found in {logs_dir}")
        return
    
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    print(f"Processing log file: {latest_log}")
    
    # Create timestamp-based subdirectory for this visualization run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = vis_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    # Create visualizer
    viz = ActiveInferenceVisualizer(latest_log)
    
    # Generate and save all visualizations
    print("Generating visualizations...")
    viz.plot_belief_evolution(run_dir)
    viz.plot_policy_evolution(run_dir)
    viz.plot_free_energy_components(run_dir)
    viz.plot_state_action_analysis(run_dir)
    viz.plot_belief_phase_space(run_dir)
    viz.plot_summary_dashboard(run_dir)
    
    print(f"Visualizations saved to: {run_dir}")
    
    # Close all figures to free memory
    plt.close('all')

if __name__ == "__main__":
    main() 