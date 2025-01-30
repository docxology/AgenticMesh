import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from tool_environment import EnvironmentTool
from tool_active_inference import ActiveInferenceTool

class JsonFormatter(logging.Formatter):
    """Custom formatter that outputs logs in JSON format."""
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        
        if hasattr(record, "json_data"):
            log_data.update(record.json_data)
            
        return json.dumps(log_data)

def setup_logging(log_dir: str = "logs"):
    """Set up logging to both file and console with JSON formatting."""
    # Create logs directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"active_inference_demo_{timestamp}.json"
    
    # Create JSON formatter
    json_formatter = JsonFormatter()
    
    # Set up file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(json_formatter)
    file_handler.setLevel(logging.INFO)
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(json_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []  # Remove any existing handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized", extra={
        "json_data": {
            "event": "logging_initialized",
            "log_file": str(log_file)
        }
    })
    
    return logger

async def run_demo(n_steps: int = 50):
    """Run a demo of the environment and active inference agent interaction."""
    logger = logging.getLogger(__name__)
    
    # Initialize tools
    env = EnvironmentTool()
    agent = ActiveInferenceTool()
    
    demo_start_time = datetime.now()
    logger.info("Starting demo", extra={
        "json_data": {
            "event": "demo_start",
            "start_time": demo_start_time.isoformat(),
            "n_steps": n_steps,
            "environment_info": {
                "n_states": len(env.states),
                "states": env.states,
                "n_actions": len(env.actions),
                "actions": env.actions
            },
            "agent_info": {
                "preferences": agent.C.tolist(),
                "initial_beliefs": agent.beliefs.tolist()
            }
        }
    })
    
    try:
        # Get initial observation
        env_result = await env.execute("demo", {})
        observation = env_result["observation"]
        logger.info("Initial observation received", extra={
            "json_data": {
                "event": "initial_observation",
                "observation": env_result["observation_name"],
                "observation_index": observation
            }
        })
        
        # Run interaction loop
        for step in range(n_steps):
            step_start_time = datetime.now()
            
            # Agent processes observation and selects action
            agent_result = await agent.execute("demo", {"observation": observation})
            action = agent_result["selected_action"]
            action_name = agent_result["selected_action_name"]
            
            logger.info("Agent step completed", extra={
                "json_data": {
                    "event": "agent_step",
                    "step": step + 1,
                    "observation": {
                        "index": observation,
                        "name": env_result["observation_name"]
                    },
                    "beliefs": {
                        "prior": [f"{p:.3f}" for p in agent_result["belief_prior"]],
                        "posterior": [f"{p:.3f}" for p in agent_result["belief_posterior"]]
                    },
                    "free_energy": {
                        "variational": [f"{f:.3f}" for f in agent_result["variational_free_energy"]],
                        "expected": [f"{g:.3f}" for g in agent_result["expected_free_energy"]]
                    },
                    "policy": {
                        "prior": [f"{p:.3f}" for p in agent_result["policy_prior"]],
                        "posterior": [f"{p:.3f}" for p in agent_result["policy_posterior"]]
                    },
                    "action": {
                        "name": action_name,
                        "index": action
                    },
                    "timing": {
                        "step_start": step_start_time.isoformat(),
                        "processing_duration": (datetime.now() - step_start_time).total_seconds()
                    }
                }
            })
            
            # Environment processes action and returns new observation
            env_result = await env.execute("demo", {"action": action_name})
            observation = env_result["observation"]
            
            step_duration = (datetime.now() - step_start_time).total_seconds()
            logger.info("Environment step completed", extra={
                "json_data": {
                    "event": "environment_step",
                    "step": step + 1,
                    "observation": {
                        "previous": {
                            "name": env_result["observation_name"],
                            "index": observation
                        }
                    },
                    "action": {
                        "name": action_name,
                        "index": action
                    },
                    "timing": {
                        "step_duration": step_duration
                    }
                }
            })
            
        demo_duration = (datetime.now() - demo_start_time).total_seconds()
        logger.info("Demo completed successfully", extra={
            "json_data": {
                "event": "demo_complete",
                "total_steps": n_steps,
                "timing": {
                    "start_time": demo_start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "total_duration": demo_duration
                },
                "final_state": {
                    "beliefs": [f"{p:.3f}" for p in agent.beliefs],
                    "policy_prior": [f"{p:.3f}" for p in agent.policy_prior],
                    "observation": {
                        "name": env_result["observation_name"],
                        "index": observation
                    }
                },
                "history": {
                    "belief_history": [[f"{p:.3f}" for p in b] for b in agent_result["history"]["belief_history"]],
                    "action_history": [env.actions[a] for a in agent_result["history"]["action_history"]],
                    "free_energy_history": [f"{f:.3f}" for f in agent_result["history"]["free_energy_history"]],
                    "policy_prior_history": [[f"{p:.3f}" for p in pp] for pp in agent_result["history"]["policy_prior_history"]]
                }
            }
        })
        
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user", extra={
            "json_data": {
                "event": "demo_interrupted",
                "completed_steps": step if 'step' in locals() else 0,
                "duration": (datetime.now() - demo_start_time).total_seconds()
            }
        })
    except Exception as e:
        logger.error("Demo failed", extra={
            "json_data": {
                "event": "demo_error",
                "error": str(e),
                "error_type": type(e).__name__,
                "completed_steps": step if 'step' in locals() else 0,
                "duration": (datetime.now() - demo_start_time).total_seconds()
            }
        })
        raise

if __name__ == "__main__":
    # Set up logging
    logger = setup_logging()
    
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        pass  # Already handled in run_demo
    except Exception as e:
        logger.error("Fatal error", extra={
            "json_data": {
                "event": "fatal_error",
                "error": str(e),
                "error_type": type(e).__name__
            }
        }) 