"""
Task Definitions & Graders
Three tasks with deterministic, reproducible graders (0.0–1.0 scoring).

Task 1  EASY   : Single-sensor anomaly identification
Task 2  MEDIUM : Multi-sensor root-cause analysis + severity classification
Task 3  HARD   : Multi-subsystem cascade with sensor dropout + recovery sequencing

Grading philosophy
All graders are *pure functions* of (state, action)  no LLM-as-judge.
This guarantees determinism across runs and hardware.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .telemetry import (
    ANOMALY_BY_ID,
    ANOMALY_CATALOGUE,
    SENSOR_SPECS,
    SUBSYSTEMS,
    Severity,
    get_subsystem_of,
)
from ..models import (
    RecommendationType,
    SeverityLevel,
    SpacecraftAction,
    SpacecraftState,
)


# Correct recovery recommendations per anomaly

ANOMALY_RECOMMENDATIONS: Dict[str, RecommendationType] = {
    "battery_undervoltage":       RecommendationType.REDUCE_POWER,
    "solar_array_degradation":    RecommendationType.REDUCE_POWER,
    "gyro_bias_drift":            RecommendationType.ATTITUDE_HOLD,
    "comms_signal_dropout":       RecommendationType.ISOLATE_COMMS,
    "reaction_wheel_bearing_wear":RecommendationType.ATTITUDE_HOLD,
    "thermal_cpu_overload":       RecommendationType.THERMAL_VENT,
    "propulsion_leak":            RecommendationType.SHUTDOWN_THRUSTER,
    "power_attitude_cascade":     RecommendationType.SAFE_MODE,
    "comms_thermal_combined":     RecommendationType.SAFE_MODE,
}


# Task descriptors

TASKS: List[Dict] = [
    {
        "id": "task_easy",
        "name": "Single-sensor anomaly identification",
        "difficulty": "easy",
        "description": (
            "A single sensor is exhibiting anomalous readings. "
            "Identify the correct sensor and flag it with the right severity. "
            "No cascade effects. Max 10 steps."
        ),
        "max_steps": 10,
        "dropout_fraction": 0.0,
        "anomaly_pool": [
            "battery_undervoltage",
            "solar_array_degradation",
            "gyro_bias_drift",
            "comms_signal_dropout",
        ],
        "require_recommendation": False,
        "require_root_cause": False,
        "expected_baseline_score": 0.72,
    },
    {
        "id": "task_medium",
        "name": "Root-cause analysis with cascade effects",
        "difficulty": "medium",
        "description": (
            "A primary fault has cascaded to 1-2 secondary sensors. "
            "Identify the root-cause subsystem and primary sensor, "
            "classify severity, and recommend an appropriate recovery action. "
            "Max 15 steps."
        ),
        "max_steps": 15,
        "dropout_fraction": 0.0,
        "anomaly_pool": [
            "reaction_wheel_bearing_wear",
            "thermal_cpu_overload",
            "propulsion_leak",
        ],
        "require_recommendation": True,
        "require_root_cause": True,
        "expected_baseline_score": 0.51,
    },
    {
        "id": "task_hard",
        "name": "Multi-subsystem cascade with sensor dropout and recovery sequencing",
        "difficulty": "hard",
        "description": (
            "A critical fault cascades across multiple subsystems. "
            "15-20% of sensors are reporting None (dropout). "
            "The agent must: identify the primary fault subsystem, "
            "correctly triage by severity, recommend the correct recovery action, "
            "and escalate to ground support — all within 20 steps. "
            "False positive flags are penalized."
        ),
        "max_steps": 20,
        "dropout_fraction": 0.18,
        "anomaly_pool": [
            "power_attitude_cascade",
            "comms_thermal_combined",
        ],
        "require_recommendation": True,
        "require_root_cause": True,
        "require_escalation": True,
        "expected_baseline_score": 0.28,
    },
]

TASK_BY_ID: Dict[str, Dict] = {t["id"]: t for t in TASKS}


# Graders

def _detection_score(state: SpacecraftState, flags: List[Dict]) -> float:
    """Did the agent flag *any* sensor belonging to the anomalous subsystem?"""
    if not state.anomaly_subsystem:
        return 0.0
    for flag in flags:
        if get_subsystem_of(flag.get("sensor", "")) == state.anomaly_subsystem:
            return 1.0
    return 0.0


def _localization_score(state: SpacecraftState, flags: List[Dict]) -> float:
    """
    Did the agent flag the *primary* sensor of the root anomaly?
    Returns 1.0 for exact match, 0.5 for correct subsystem but wrong sensor.
    """
    if not state.anomaly_id:
        return 0.0
    anomaly = ANOMALY_BY_ID[state.anomaly_id]
    primary = anomaly["primary_sensor"]

    for flag in flags:
        if flag.get("sensor") == primary:
            return 1.0

    # Partial credit: correct subsystem
    for flag in flags:
        if get_subsystem_of(flag.get("sensor", "")) == state.anomaly_subsystem:
            return 0.5

    return 0.0


def _severity_score(state: SpacecraftState, flags: List[Dict]) -> float:
    """
    Was the severity of the flag for the primary sensor correct?
    Partial credit (0.3) for adjacent severity (warning vs critical).
    """
    if not state.anomaly_id:
        return 0.0
    anomaly = ANOMALY_BY_ID[state.anomaly_id]
    primary = anomaly["primary_sensor"]
    true_severity = anomaly["severity"].value  # "info" | "warning" | "critical"

    _adjacent = {
        "info": {"info": 1.0, "warning": 0.3, "critical": 0.0},
        "warning": {"info": 0.3, "warning": 1.0, "critical": 0.3},
        "critical": {"info": 0.0, "warning": 0.3, "critical": 1.0},
    }

    for flag in flags:
        if flag.get("sensor") == primary:
            agent_sev = flag.get("severity", "")
            return _adjacent.get(true_severity, {}).get(agent_sev, 0.0)

    return 0.0


def _action_score(state: SpacecraftState, recommendations: List[str]) -> float:
    """Was the recommended recovery action correct?"""
    if not state.anomaly_id:
        return 0.0
    correct = ANOMALY_RECOMMENDATIONS.get(state.anomaly_id)
    if correct is None:
        return 0.0
    for rec in recommendations:
        if rec == correct.value:
            return 1.0
    # Partial credit for safe_mode when critical
    anomaly = ANOMALY_BY_ID[state.anomaly_id]
    if anomaly["severity"] == Severity.CRITICAL and "safe_mode" in recommendations:
        return 0.5
    return 0.0


def _speed_bonus(state: SpacecraftState, max_steps: int) -> float:
    if state.first_correct_detection_step is None:
        return 0.0
    ratio = 1.0 - (state.first_correct_detection_step / max_steps)
    return max(0.0, ratio)


def _false_positive_penalty(
    state: SpacecraftState,
    flags: List[Dict],
) -> float:
    """
    Penalize flags on sensors that are NOT part of the anomaly.
    Returns value in [0,1]; environment subtracts this from reward.
    """
    if not state.anomaly_id:
        return 0.0
    anomaly = ANOMALY_BY_ID[state.anomaly_id]
    affected_sensors = set(anomaly["affected"])
    fp_count = sum(
        1 for f in flags if f.get("sensor") not in affected_sensors
    )
    # Each FP costs 0.1, capped at 0.5
    return min(0.5, fp_count * 0.1)


def _escalation_score(state: SpacecraftState) -> float:
    """Hard task: did agent call request_support?"""
    return 1.0 if getattr(state, "_escalated", False) else 0.0


# Per-task composite grader

def compute_reward(
    state: SpacecraftState,
    task: Dict,
    flags: List[Dict],
    recommendations: List[str],
    action: Optional[SpacecraftAction] = None,
    escalated: bool = False,
) -> "RewardBreakdown":  
    """
    Compute the shaped reward for the current state.
    Called every step (provides dense signal, not just terminal).

    Weights by task difficulty:
      easy   : detection 0.7, localization 0.3, severity 0.2, speed 0.1, fp -0.1
      medium : detection 0.3, localization 0.3, severity 0.2, action 0.3, speed 0.1, fp -0.15
      hard   : detection 0.25, localization 0.25, severity 0.15, action 0.2,
               escalation 0.1, speed 0.05, fp -0.2
    """
    from ..models import RewardBreakdown  # local import to avoid circular

    difficulty = task["difficulty"]
    max_steps = task["max_steps"]

    det  = _detection_score(state, flags)
    loc  = _localization_score(state, flags)
    sev  = _severity_score(state, flags)
    act  = _action_score(state, recommendations) if task.get("require_recommendation") else 0.0
    spd  = _speed_bonus(state, max_steps)
    fp   = _false_positive_penalty(state, flags)

    if difficulty == "easy":
        total = (
            0.50 * det +
            0.25 * loc +
            0.15 * sev +
            0.10 * spd -
            0.10 * fp
        )
    elif difficulty == "medium":
        total = (
            0.25 * det +
            0.25 * loc +
            0.15 * sev +
            0.25 * act +
            0.10 * spd -
            0.15 * fp
        )
    else:  # hard
        esc = 1.0 if escalated else 0.0
        total = (
            0.20 * det +
            0.20 * loc +
            0.15 * sev +
            0.20 * act +
            0.10 * esc +
            0.05 * spd -
            0.20 * fp
        )

    total = int(round(total))

    return RewardBreakdown(
        detection=int(round(det)),
        localization=int(round(loc)),
        severity=int(round(sev)),
        action=int(round(act)),
        speed_bonus=int(round(spd)),
        fp_penalty=int(round(fp)),
        total=total,
    )


def sample_task_anomaly(task_id: str, seed: Optional[int] = None) -> str:
    """Deterministically or randomly sample an anomaly_id for a task."""
    task = TASK_BY_ID[task_id]
    pool = task["anomaly_pool"]
    rng = random.Random(seed)
    return rng.choice(pool)
