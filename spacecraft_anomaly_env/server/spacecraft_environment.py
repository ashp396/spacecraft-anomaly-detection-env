"""
SpacecraftAnomalyEnvironment
============================
Core environment class. Implements OpenEnv RFC-002 interface:
  reset() → SpacecraftObservation
  step(action) → SpacecraftObservation
  state  → SpacecraftState  (property)

State management
----------------
Each call to reset() creates a fresh episode with:
  - a new UUID episode_id
  - a randomly sampled anomaly from the task's anomaly_pool
  - clean telemetry state seeded from the episode_id hash (reproducible)

The environment is *stateful per session*. Concurrent sessions are supported
only when SUPPORTS_CONCURRENT_SESSIONS=False (single instance in Docker).
"""

from __future__ import annotations

import uuid
from typing import Optional

import numpy as np

from ..models import (
    RecommendationType,
    SpacecraftAction,
    SpacecraftObservation,
    SpacecraftState,
    ActionType,
)
from .tasks import (
    ANOMALY_RECOMMENDATIONS,
    TASK_BY_ID,
    compute_reward,
    sample_task_anomaly,
)
from .telemetry import (
    ANOMALY_BY_ID,
    SENSOR_SPECS,
    SUBSYSTEMS,
    TelemetryState,
    generate_readings,
    get_subsystem_of,
)


class SpacecraftAnomalyEnvironment:
    """
    Spacecraft anomaly detection environment.

    Supports three task difficulties via task_id:
      task_easy   — single sensor, 10 steps
      task_medium — cascade, 15 steps, requires recommendation
      task_hard   — multi-subsystem + dropout, 20 steps, requires escalation
    """

    def __init__(self, task_id: str = "task_easy"):
        if task_id not in TASK_BY_ID:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASK_BY_ID)}")
        self._task_id = task_id
        self._task = TASK_BY_ID[task_id]
        self._state: SpacecraftState = SpacecraftState()
        self._tel_state: Optional[TelemetryState] = None
        self._escalated: bool = False

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> SpacecraftObservation:
        """Initialize a new episode and return the initial observation."""
        episode_id = str(uuid.uuid4())

        # Deterministic seed derived from episode_id (for reproducibility)
        rng_seed = seed if seed is not None else (
            int(uuid.UUID(episode_id).int % (2**31))
        )

        # Sample anomaly for this episode
        anomaly_id = sample_task_anomaly(self._task_id, seed=rng_seed)
        anomaly = ANOMALY_BY_ID[anomaly_id]

        # Build telemetry state
        self._tel_state = TelemetryState(
            rng=np.random.default_rng(rng_seed),
            step=0,
            active_anomaly_id=anomaly_id,
        )

        # Apply dropout mask for hard task
        dropout_fraction = self._task.get("dropout_fraction", 0.0)
        if dropout_fraction > 0.0:
            all_sensors = list(SENSOR_SPECS.keys())
            rng_drop = np.random.default_rng(rng_seed + 9999)
            n_drop = max(1, int(len(all_sensors) * dropout_fraction))
            dropout_sensors = rng_drop.choice(
                all_sensors, size=n_drop, replace=False
            ).tolist()
            self._tel_state.dropout_sensors = dropout_sensors

        # Build server-side state
        self._state = SpacecraftState(
            episode_id=episode_id,
            step_count=0,
            task_id=self._task_id,
            task_difficulty=self._task["difficulty"],
            anomaly_id=anomaly_id,
            anomaly_subsystem=anomaly["subsystem"],
            anomaly_severity=anomaly["severity"].value,
            correct_recommendation=ANOMALY_RECOMMENDATIONS.get(anomaly_id, RecommendationType.NO_ACTION).value,
            max_steps=self._task["max_steps"],
            dropout_fraction=dropout_fraction,
            flags_raised=[],
            recommendations_made=[],
            first_correct_detection_step=None,
            episode_complete=False,
        )
        self._escalated = False

        telemetry = generate_readings(self._tel_state, dropout_fraction)

        return SpacecraftObservation(
            telemetry=telemetry,
            step_count=0,
            steps_remaining=self._task["max_steps"],
            active_flags=[],
            last_action_result="Episode initialized. Nominal telemetry stream active.",
            reward=compute_reward(
                self._state, self._task, [], [], None, False
            ),
            done=False,
            success=True,
            info={
                "episode_id": episode_id,
                "task_id": self._task_id,
                "task_difficulty": self._task["difficulty"],
                "task_description": self._task["description"],
                "subsystems": list(SUBSYSTEMS.keys()),
                "sensors": list(SENSOR_SPECS.keys()),
                "dropout_sensors": self._tel_state.dropout_sensors,
            },
        )

    def step(self, action: SpacecraftAction) -> SpacecraftObservation:
        """Execute one agent action and return the resulting observation."""
        if self._tel_state is None:
            raise RuntimeError("Call reset() before step().")

        self._tel_state.step += 1
        self._state.step_count += 1
        steps_remaining = self._state.max_steps - self._state.step_count

        result_msg = self._execute_action(action)

        # Advance telemetry
        telemetry = generate_readings(
            self._tel_state,
            self._state.dropout_fraction,
        )

        # Check termination
        done = (
            steps_remaining <= 0
            or self._state.episode_complete
        )
        if done:
            self._state.episode_complete = True

        # Compute shaped reward
        reward = compute_reward(
            self._state,
            self._task,
            self._state.flags_raised,
            self._state.recommendations_made,
            action,
            self._escalated,
        )

        return SpacecraftObservation(
            telemetry=telemetry,
            step_count=self._state.step_count,
            steps_remaining=max(0, steps_remaining),
            active_flags=list(self._state.flags_raised),
            last_action_result=result_msg,
            reward=reward,
            done=done,
            success=True,
            info={
                "episode_id": self._state.episode_id,
                "task_id": self._task_id,
                "escalated": self._escalated,
                "step_count": self._state.step_count,
            },
        )

    @property
    def state(self) -> SpacecraftState:
        """Return current server-side episode state (ground truth included)."""
        return self._state

    # ------------------------------------------------------------------
    # Internal action dispatch
    # ------------------------------------------------------------------

    def _execute_action(self, action: SpacecraftAction) -> str:
        """Dispatch action and return human-readable result string."""
        at = action.action_type

        if at == ActionType.QUERY_SUBSYSTEM:
            return self._do_query(action)
        elif at == ActionType.FLAG_ANOMALY:
            return self._do_flag(action)
        elif at == ActionType.CLEAR_FLAG:
            return self._do_clear(action)
        elif at == ActionType.RECOMMEND:
            return self._do_recommend(action)
        elif at == ActionType.REQUEST_SUPPORT:
            return self._do_escalate(action)
        elif at == ActionType.NO_OP:
            return "No action taken. Telemetry stream updated."
        else:
            return f"Unknown action type: {at}"

    def _do_query(self, action: SpacecraftAction) -> str:
        sub = action.subsystem
        if sub not in SUBSYSTEMS:
            return f"Unknown subsystem '{sub}'. Valid: {list(SUBSYSTEMS.keys())}"
        sensors = SUBSYSTEMS[sub]
        readings = {
            s: self._tel_state.ar_state.get(s, 0.0)
            for s in sensors
        }
        return (
            f"Subsystem '{sub}' queried. "
            f"Contains sensors: {sensors}. "
            f"AR noise state: {readings}."
        )

    def _do_flag(self, action: SpacecraftAction) -> str:
        sensor = action.sensor
        if not sensor or sensor not in SENSOR_SPECS:
            return f"Invalid sensor '{sensor}'."

        sev = action.severity.value if action.severity else "warning"
        sub = get_subsystem_of(sensor)

        # Check for duplicate flag on same sensor
        existing = [f for f in self._state.flags_raised if f["sensor"] == sensor]
        if existing:
            return f"Sensor '{sensor}' already flagged. Use clear_flag to rescind."

        flag_entry = {
            "sensor": sensor,
            "subsystem": sub,
            "severity": sev,
            "step": self._state.step_count,
            "confidence": action.confidence,
        }
        self._state.flags_raised.append(flag_entry)

        # Track first correct detection for speed bonus
        anomaly = ANOMALY_BY_ID.get(self._state.anomaly_id, {})
        if (
            self._state.first_correct_detection_step is None
            and sensor in anomaly.get("affected", [])
        ):
            self._state.first_correct_detection_step = self._state.step_count

        return (
            f"Anomaly flag raised: sensor='{sensor}', "
            f"subsystem='{sub}', severity='{sev}', "
            f"confidence={action.confidence:.2f}."
        )

    def _do_clear(self, action: SpacecraftAction) -> str:
        sensor = action.sensor
        before = len(self._state.flags_raised)
        self._state.flags_raised = [
            f for f in self._state.flags_raised if f["sensor"] != sensor
        ]
        removed = before - len(self._state.flags_raised)
        return f"Cleared {removed} flag(s) for sensor '{sensor}'."

    def _do_recommend(self, action: SpacecraftAction) -> str:
        if not action.recommendation:
            return "recommend action requires a 'recommendation' field."
        rec = action.recommendation.value
        self._state.recommendations_made.append(rec)
        return f"Recovery recommendation logged: '{rec}'."

    def _do_escalate(self, action: SpacecraftAction) -> str:
        sub = action.subsystem or "unknown"
        self._escalated = True
        return (
            f"Ground support requested for subsystem '{sub}'. "
            f"Escalation logged at step {self._state.step_count}."
        )
