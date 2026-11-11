# AgenticMesh — canonical backlog

Open, owner-local work only. Completed work belongs in `CHANGELOG.md` and Git history.

## Workflowz review — 2026-11-10

- [ ] **AM-01 (P0): Replace `MathTool`’s Python `eval` sandbox with an AST-based arithmetic evaluator.** `tool.py:64-72` passes caller text to `eval`; disabling `__builtins__` does not block object-graph or module-attribute access, contrary to the no-attribute-access constraint in `AGENTS.md:54-55`. Outcome: only numeric operators and explicitly whitelisted `math` constants/functions evaluate; attribute access, imports, comprehensions, and arbitrary Python objects raise `ValueError`, while existing arithmetic calls keep working. Acceptance probe from repo root:
  ```bash
  python3 - <<'PY'
  import asyncio
  from tool import MathTool
  async def main():
      tool = MathTool()
      result = await tool.execute("ok", {"expression": "math.sqrt(9) + 1"})
      assert result["value"] == 4
      for expression in ("math.__dict__", "().__class__", "__import__('os')"):
          try:
              await tool.execute("bad", {"expression": expression})
          except ValueError:
              pass
          else:
              raise AssertionError(f"accepted unsafe expression: {expression}")
  asyncio.run(main())
  PY
  ```
  Expected: all assertions pass and all three unsafe expressions are rejected. Dependencies: none; add focused stdlib `unittest` coverage and preserve the documented expression examples without expanding the evaluator to general Python.

- [ ] **AM-02 (P1): Make `habit_matrix` genuinely optional in `ActiveInferenceTool._initialize_model`.** `active-inference/tool_active_inference.py:130-145` omits it from `required_params` but unconditionally indexes it; this raises `KeyError` before the uniform-default branch at `active-inference/tool_active_inference.py:190-197` can run. Outcome: omitted habits produce a uniform `E` and `policy_prior`; supplied habits are shape-checked, nonnegative, finite, and normalized. Acceptance probe from repo root:
  ```bash
  python3 - <<'PY'
  import sys, numpy as np
  sys.path.insert(0, "active-inference")
  from tool_active_inference import ActiveInferenceTool
  agent = ActiveInferenceTool()
  params = {"n_states": 3, "n_observations": 3, "n_actions": 3,
            "likelihood_matrix": agent.A, "transition_matrix": agent.B,
            "preferences": agent.C, "initial_beliefs": agent.D}
  agent._initialize_model(params)
  np.testing.assert_allclose(agent.E, np.full(3, 1/3))
  agent._initialize_model({**params, "habit_matrix": [1.0, 1.0, 2.0]})
  np.testing.assert_allclose(agent.E, [0.25, 0.25, 0.5])
  PY
  ```
  Expected: both initializations complete with the asserted distributions. Dependencies: none; add regression coverage for omitted and explicit habits rather than making habits required.

- [ ] **AM-03 (P1): Compute the pragmatic expected-free-energy term as the documented observation KL divergence.** The comment at `active-inference/tool_active_inference.py:298-302` says KL divergence, but the implementation calculates only cross-entropy `-(q * log(p)).sum()`, omitting `q * log(q)`; that omission can change action rankings because expected observation entropy is action-dependent. Outcome: `pragmatic_values` and `G` use `sum(q * (log(q) - log(p)))` with stable zero-probability handling. Acceptance probe from repo root:
  ```bash
  python3 - <<'PY'
  import sys, numpy as np
  sys.path.insert(0, "active-inference")
  from tool_active_inference import ActiveInferenceTool
  agent = ActiveInferenceTool()
  agent._compute_expected_free_energy()
  q = agent.A @ (agent.B[:, :, 0] @ agent.beliefs)
  p = np.exp(agent.C); p /= p.sum()
  expected = np.sum(q * (np.log(q + 1e-16) - np.log(p + 1e-16)))
  np.testing.assert_allclose(agent.metrics["pragmatic_values"][0], expected)
  PY
  ```
  Expected: the recorded first-action pragmatic value equals the independently computed KL. Dependencies: none; add a two-action regression where cross-entropy and KL rank actions differently, and do not alter the separate epistemic term in this item.

- [ ] **AM-04 (P1): Make the documented YAML configuration the actual model source and align its schema/orientation.** `active-inference/ActiveInference_README.md:116-120` says the agent is configured by `agent-active-inference.yaml`, but no runtime caller uses `ActiveInferenceUtils.load_configuration`; the helper expects top-level `model` at `active-inference/active_inference_utils.py:30-42`, whereas the file stores values under `simulation.agent` at `active-inference/agent-active-inference.yaml:29-73`. The YAML likelihood rows at `agent-active-inference.yaml:39-44` also do not satisfy the column-normalized `A` contract checked at `tool_active_inference.py:141-153`, and `active_inference_utils.py:59-62` checks transitions along the wrong axis for `B[next_state, current_state, action]`. Outcome: simulation/agent initialization loads one validated YAML schema, duplicate hardcoded matrix initialization is removed, likelihood/transition orientation is explicit, and exact normalized beliefs/habits replace the `0.333` approximations. Acceptance: add `tests/test_configuration.py` and run `python3 -m unittest tests.test_configuration`; expected tests prove the shipped YAML loads, a temporary YAML preference/habit change reaches the initialized agent, invalid matrix axes are rejected, and the demo uses that configuration. Dependencies: AM-02 for optional-habit semantics; update both READMEs and keep the shipped three-state behavior equivalent.

- [ ] **AM-05 (P1): Return nonzero process status when either command-line entry point fails.** The top-level simulation handler logs and suppresses exceptions at `active-inference/tool_simulation_active_inference.py:218-232`; visualization similarly catches its outer exception at `active-inference/tool_actinf_visualization.py:1648-1653`, returns normally for no logs at `:1581-1584`, and can finish with zero/failed plots at `:1637-1646` without failure status. Outcome: uncaught simulation errors, missing/invalid input logs, and incomplete requested plot generation produce exit status 1; successful complete runs return 0. Acceptance: add `tests/test_cli.py` using temporary directories/subprocesses and run `python3 -m unittest tests.test_cli`; expected assertions cover nonzero missing-log, invalid-log, failed-plot, and injected simulation-failure exits plus successful zero exits. Dependencies: none; retain useful JSON diagnostics and intentional `KeyboardInterrupt` handling, but do not convert failures into successful exits.

- [ ] **AM-06 (P2): Anchor default simulation logs beside the active-inference module so the visualizer reads the run just produced.** The documented root invocation at `AGENTS.md:42-47` uses relative `logs` through `setup_logging` at `active-inference/tool_simulation_active_inference.py:24-31`, creating `<cwd>/logs`; the visualizer instead reads `active-inference/logs` at `active-inference/tool_actinf_visualization.py:1557-1568`. Outcome: default output goes to the same module-relative log directory from every working directory, while an explicit `log_dir` remains honored. Acceptance: add a test that runs setup/demo from a temporary working directory, then run `python3 -m unittest tests.test_simulation.LoggingPathTests`; expected a new JSONL file appears under `active-inference/logs` (or an injected module-equivalent temporary directory) and is the file selected by visualization, with no new `<cwd>/logs`. Dependencies: coordinate with AM-05’s subprocess fixtures; preserve committed sample logs.

- [ ] **AM-07 (P2): Correct the `ToolComponent.execute` parameter contract to raw values.** The abstract signature at `tool_component.py:57-63` declares `Dict[str, ToolParameterModel]`, while its docstring and all implementations (`tool.py:60`, `tool_active_inference.py:405`, and `tool_environment.py:136`) accept runtime values indexed by name. Outcome: the abstract annotation is `Dict[str, Any]` and generated/static API documentation no longer tells implementers to pass descriptor objects. Acceptance probe:
  ```bash
  python3 - <<'PY'
  from typing import Any, Dict, get_type_hints
  from tool_component import ToolComponent
  assert get_type_hints(ToolComponent.execute)["parameters"] == Dict[str, Any]
  PY
  ```
  Expected: assertion passes. Dependencies: none; update any signature tables only if they reproduce the incorrect annotation, without changing runtime dispatch.

- [ ] **AM-08 (P2): Pin and exercise the documented interpreter as Python 3.15.2, retaining the declared dependency floors.** Current commands use unversioned `python3` (`README.md:70-72`, `AGENTS.md:42-46`), and the repository has no Python version pin or CI; `active-inference/requirements.txt:1-16` supplies only package floors. Outcome: add a 3.15.2 runtime pin and a CI workflow that installs the existing requirements and runs the focused regression tests on exactly 3.15.2; document the same interpreter in run instructions. Acceptance: CI is green on `python3.15` 3.15.2 and locally `python3.15 -m unittest discover -s tests` passes from repo root. Dependencies: AM-01 through AM-05 supply behavioral tests; do not raise or blindly refresh any NumPy/SciPy/Pandas/visualization floors, and if an actual 3.15.2 incompatibility is demonstrated, record the exact package constraint and choose the oldest supported 3.15.2-compatible floor rather than changing packages speculatively.
