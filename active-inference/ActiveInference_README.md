# Active Inference Agent Implementation

## Overview
This repository implements an Active Inference agent for Partially Observable Markov Decision Processes (POMDPs) in discrete time and state spaces. The implementation follows the free energy principle and active inference framework developed by Karl Friston and colleagues.

## Components

### 1. Active Inference Agent (`tool_active_inference.py`)
- Implements core active inference algorithms
- Handles belief updating and action selection
- Manages model parameters and state
- Updates and maintains action habits

### 2. Visualization Tool (`tool_actinf_visualization.py`)
Comprehensive visualization suite providing:

#### Belief Analysis
- **Belief Evolution**: Tracks prior and posterior beliefs over time
- **Belief Phase Space**: Visualizes belief trajectories in 2D and 3D
- **Update Magnitudes**: Monitors the size of belief updates

#### Policy Analysis
- **Policy Evolution**: Shows how action preferences change
- **Action Distribution**: Displays action selection patterns
- **State-Action Analysis**: Comprehensive transition analysis

#### Free Energy Analysis
- **Variational Free Energy**: Per-state free energy tracking
- **Expected Free Energy**: Action-specific expected free energies
- **Cumulative Analysis**: Tracks free energy changes over time

#### Summary Dashboard
Provides a comprehensive overview including:
- Current belief states
- Policy preferences
- Free energy components
- State-action transitions
- Action and observation sequences

### 3. Configuration System
Uses YAML-based configuration for:
- Model parameters
- Agent architecture
- Visualization settings
- Logging preferences

## Usage

### Installation
```bash
pip install numpy scipy pandas matplotlib seaborn networkx
```

### Running the Agent
```python
from active_inference_tool import ActiveInferenceTool

# Initialize the agent
agent = ActiveInferenceTool()

# Configure parameters
parameters = {
    "n_states": 3,              # Low, Medium, High
    "n_observations": 3,        # Matches states
    "n_actions": 3,             # Decrease, Stay, Increase
    "likelihood_matrix": A,     # Observation model [n_observations × n_states]
    "transition_matrix": B,     # State dynamics [n_states × n_states × n_actions]
    "preferences": C,           # Prior preferences [n_observations]
    "initial_beliefs": D,       # Initial state distribution [n_states]
    "habit_matrix": E          # Action preferences [n_actions]
}

# Initialize and run
result = await agent.execute("init", parameters)
next_result = await agent.execute("step", {"observation": current_observation})
```

### Generating Visualizations
```bash
cd active-inference
python3 tool_actinf_visualization.py
```

This will:
1. Process the most recent agent log file
2. Generate comprehensive visualizations
3. Save results in `visualizations/run_TIMESTAMP/`

Generated files include:
- `belief_evolution.png`: Belief tracking over time
- `policy_evolution.png`: Action preference development
- `free_energy_analysis.png`: Free energy components
- `state_action_analysis.png`: Transition patterns
- `belief_phase_space.png`: Belief trajectories
- `summary_dashboard.png`: Complete overview

## Implementation Details

### Active Inference Cycle
1. **Perception**: Update beliefs using variational free energy
2. **Planning**: Compute expected free energy for actions
3. **Action**: Sample action from policy distribution
4. **Learning**: Update habits based on outcomes

### Key Features
- One-hot action sampling from policy posterior
- Comprehensive belief tracking
- Detailed free energy analysis
- Advanced visualization capabilities
- Robust numerical stability

### Visualization Features
- Consistent state/action ordering (Low, Medium, High)
- Professional color schemes
- Enhanced readability
- Interactive elements
- High-resolution output

## Configuration

The agent is configured through `agent-active-inference.yaml`, specifying:
- Model dimensions
- Learning parameters
- Visualization preferences
- Logging settings

## Development

### Adding Features
1. Extend core classes in `tool_active_inference.py`
2. Update visualization in `tool_actinf_visualization.py`
3. Modify configuration in `agent-active-inference.yaml`
4. Update documentation

### Best Practices
- Use type hints
- Follow error handling patterns
- Maintain comprehensive logging
- Write clear documentation

## References

1. Friston, K., et al. (2017). Active Inference: A Process Theory. Neural Computation, 29(1), 1-49.
2. Da Costa, L., et al. (2020). Active inference on discrete state-spaces: A synthesis. Journal of Mathematical Psychology, 99, 102447.

## License
MIT License - See LICENSE file for details
