"""
Test Suite — Spacecraft Anomaly Detection OpenEnv
=================================================
Tests cover:
  - Environment reset/step/state interface compliance
  - Reward range [0.0, 1.0] for all tasks
  - Grader determinism (same seed → same score)
  - Telemetry physics (readings within physical bounds)
  - Action validation
"""

from __future__ import annotations

import sys
import os

# Allow running from repo root without installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

from ..models import (
    ActionType,
    RecommendationType,
    SeverityLevel,
    SpacecraftAction,
)
from ..server.spacecraft_environment import SpacecraftAnomalyEnvironment
from ..server.telemetry import (
    SENSOR_SPECS,
    SUBSYSTEMS,
    TelemetryState,
    generate_readings,
)
from ..server.tasks import (
    TASKS,
    TASK_BY_ID,
    compute_reward,
    sample_task_anomaly,
)


# ---------------------------------------------------------------------------
# Telemetry physics tests
# ---------------------------------------------------------------------------

class TestTelemetryPhysics:
    def test_nominal_readings_within_safe_bounds(self):
        """All nominal readings should stay within ±20% of safe limits."""
        state = TelemetryState(rng=np.random.default_rng(0))
        for _ in range(20):
            readings = generate_readings(state)
            for sensor, val in readings.items():
                if val is None:
                    continue
                spec = SENSOR_SPECS[sensor]
                safe_min, safe_max = spec["safe"]
                margin = abs(safe_max - safe_min) * 0.2
                assert val >= safe_min - margin, (
                    f"{sensor}={val} below safe_min={safe_min}"
                )
                assert val <= safe_max + margin, (
                    f"{sensor}={val} above safe_max={safe_max}"
                )

    def test_anomaly_shifts_primary_sensor(self):
        """Active anomaly should push primary sensor away from nominal."""
        state_normal = TelemetryState(rng=np.random.default_rng(42))
        state_anomaly = TelemetryState(rng=np.random.default_rng(42),
                                       active_anomaly_id="battery_undervoltage")

        # Step to step 10 so anomaly has developed
        for _ in range(10):
            generate_readings(state_normal)
            state_anomaly.step += 1
        
        readings_n = generate_readings(state_normal)
        readings_a = generate_readings(state_anomaly)

        # Battery voltage should be lower with anomaly
        assert readings_a["battery_voltage"] < readings_n["battery_voltage"] - 1.0

    def test_dropout_returns_none(self):
        """Dropout sensors return None."""
        state = TelemetryState(rng=np.random.default_rng(0),
                               dropout_sensors=["battery_voltage", "cpu_temp"])
        readings = generate_readings(state)
        assert readings["battery_voltage"] is None
        assert readings["cpu_temp"] is None
        assert readings["tank_pressure"] is not None


# ---------------------------------------------------------------------------
# Environment interface tests
# ---------------------------------------------------------------------------

class TestEnvironmentInterface:

    @pytest.fixture(params=["task_easy", "task_medium", "task_hard"])
    def env(self, request):
        return SpacecraftAnomalyEnvironment(task_id=request.param)

    def test_reset_returns_observation(self, env):
        obs = env.reset(seed=42)
        assert obs.success is True
        assert obs.done is False
        assert obs.step_count == 0
        assert len(obs.telemetry) == 16  # all sensors present

    def test_reset_produces_clean_state(self, env):
        env.reset(seed=42)
        env.step(SpacecraftAction(action_type=ActionType.NO_OP))
        env.step(SpacecraftAction(action_type=ActionType.NO_OP))
        # Second reset should give step_count=0 and no flags
        obs = env.reset(seed=99)
        assert obs.step_count == 0
        assert obs.active_flags == []

    def test_state_property_returns_state(self, env):
        env.reset(seed=42)
        s = env.state
        assert s.episode_id != ""
        assert s.step_count == 0

    def test_step_increments_count(self, env):
        env.reset(seed=42)
        action = SpacecraftAction(action_type=ActionType.NO_OP)
        obs = env.step(action)
        assert obs.step_count == 1
        obs = env.step(action)
        assert obs.step_count == 2

    def test_episode_terminates_at_max_steps(self, env):
        task = TASK_BY_ID[env._task_id]
        env.reset(seed=42)
        obs = None
        for _ in range(task["max_steps"] + 5):
            obs = env.step(SpacecraftAction(action_type=ActionType.NO_OP))
            if obs.done:
                break
        assert obs is not None and obs.done

    def test_reward_always_in_range(self, env):
        env.reset(seed=42)
        for _ in range(5):
            obs = env.step(SpacecraftAction(action_type=ActionType.NO_OP))
            assert 0.0 <= obs.reward.total <= 1.0


# ---------------------------------------------------------------------------
# Action tests
# ---------------------------------------------------------------------------

class TestActions:

    def test_flag_anomaly_recorded(self):
        env = SpacecraftAnomalyEnvironment("task_easy")
        env.reset(seed=42)
        action = SpacecraftAction(
            action_type=ActionType.FLAG_ANOMALY,
            sensor="battery_voltage",
            severity=SeverityLevel.WARNING,
            confidence=0.9,
        )
        obs = env.step(action)
        assert len(obs.active_flags) == 1
        assert obs.active_flags[0]["sensor"] == "battery_voltage"

    def test_duplicate_flag_rejected(self):
        env = SpacecraftAnomalyEnvironment("task_easy")
        env.reset(seed=42)
        action = SpacecraftAction(
            action_type=ActionType.FLAG_ANOMALY,
            sensor="battery_voltage",
            severity=SeverityLevel.WARNING,
        )
        env.step(action)
        obs = env.step(action)
        # Still only one flag
        assert len(obs.active_flags) == 1

    def test_clear_flag_removes_entry(self):
        env = SpacecraftAnomalyEnvironment("task_easy")
        env.reset(seed=42)
        env.step(SpacecraftAction(
            action_type=ActionType.FLAG_ANOMALY,
            sensor="battery_voltage",
            severity=SeverityLevel.WARNING,
        ))
        obs = env.step(SpacecraftAction(
            action_type=ActionType.CLEAR_FLAG,
            sensor="battery_voltage",
        ))
        assert len(obs.active_flags) == 0

    def test_recommend_logged(self):
        env = SpacecraftAnomalyEnvironment("task_medium")
        env.reset(seed=42)
        env.step(SpacecraftAction(
            action_type=ActionType.RECOMMEND,
            recommendation=RecommendationType.SAFE_MODE,
        ))
        assert "safe_mode" in env.state.recommendations_made

    def test_invalid_sensor_gives_error_message(self):
        env = SpacecraftAnomalyEnvironment("task_easy")
        env.reset(seed=42)
        obs = env.step(SpacecraftAction(
            action_type=ActionType.FLAG_ANOMALY,
            sensor="nonexistent_sensor_xyz",
            severity=SeverityLevel.WARNING,
        ))
        assert "Invalid sensor" in obs.last_action_result


# ---------------------------------------------------------------------------
# Grader determinism tests
# ---------------------------------------------------------------------------

class TestGraderDeterminism:

    def test_same_seed_same_anomaly(self):
        for task_id in ["task_easy", "task_medium", "task_hard"]:
            a1 = sample_task_anomaly(task_id, seed=42)
            a2 = sample_task_anomaly(task_id, seed=42)
            assert a1 == a2

    def test_different_seed_may_differ(self):
        # Not guaranteed to differ for every task, but likely for easy (4 options)
        anomalies = set(sample_task_anomaly("task_easy", seed=i) for i in range(10))
        assert len(anomalies) > 1

    def test_reward_deterministic_across_runs(self):
        """Same seed, same actions → same final reward."""
        actions = [
            SpacecraftAction(action_type=ActionType.NO_OP),
            SpacecraftAction(
                action_type=ActionType.FLAG_ANOMALY,
                sensor="battery_voltage",
                severity=SeverityLevel.WARNING,
            ),
            SpacecraftAction(
                action_type=ActionType.RECOMMEND,
                recommendation=RecommendationType.REDUCE_POWER,
            ),
        ]

        def run_and_score():
            env = SpacecraftAnomalyEnvironment("task_easy")
            env.reset(seed=42)
            last_reward = 0.0
            for a in actions:
                obs = env.step(a)
                last_reward = obs.reward.total
            return last_reward

        r1 = run_and_score()
        r2 = run_and_score()
        assert abs(r1 - r2) < 1e-9

    def test_correct_action_scores_higher_than_random(self):
        """Correct flag + recommendation should beat random no_op."""
        env_correct = SpacecraftAnomalyEnvironment("task_easy")
        env_correct.reset(seed=42)
        anomaly_id = env_correct.state.anomaly_id

        from ..server.telemetry import ANOMALY_BY_ID
        from ..server.tasks import ANOMALY_RECOMMENDATIONS
        anomaly = ANOMALY_BY_ID[anomaly_id]
        correct_sensor = anomaly["primary_sensor"]
        correct_sev = anomaly["severity"].value
        correct_rec = ANOMALY_RECOMMENDATIONS.get(anomaly_id)

        for _ in range(3):
            env_correct.step(SpacecraftAction(action_type=ActionType.NO_OP))
        obs_correct = env_correct.step(SpacecraftAction(
            action_type=ActionType.FLAG_ANOMALY,
            sensor=correct_sensor,
            severity=SeverityLevel(correct_sev),
            confidence=0.95,
        ))
        score_correct = obs_correct.reward.total

        env_random = SpacecraftAnomalyEnvironment("task_easy")
        env_random.reset(seed=42)
        for _ in range(4):
            obs_random = env_random.step(SpacecraftAction(action_type=ActionType.NO_OP))
        score_random = obs_random.reward.total

        assert score_correct > score_random


# ---------------------------------------------------------------------------
# Task spec tests
# ---------------------------------------------------------------------------

class TestTaskSpec:
    def test_three_tasks_exist(self):
        assert len(TASKS) >= 3

    def test_difficulty_progression(self):
        diffs = [t["difficulty"] for t in TASKS]
        assert "easy" in diffs
        assert "medium" in diffs
        assert "hard" in diffs

    def test_all_anomaly_pools_non_empty(self):
        for task in TASKS:
            assert len(task["anomaly_pool"]) >= 1

    def test_max_steps_increases_with_difficulty(self):
        easy   = next(t for t in TASKS if t["difficulty"] == "easy")
        medium = next(t for t in TASKS if t["difficulty"] == "medium")
        hard   = next(t for t in TASKS if t["difficulty"] == "hard")
        assert easy["max_steps"] < medium["max_steps"] <= hard["max_steps"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
