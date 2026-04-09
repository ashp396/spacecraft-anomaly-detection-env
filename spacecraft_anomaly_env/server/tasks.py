"""
Task Definitions & Graders
==========================
Three tasks with deterministic, reproducible graders.

Task 1 — EASY   : Single-sensor anomaly identification
Task 2 — MEDIUM : Multi-sensor root-cause analysis + severity classification
Task 3 — HARD   : Multi-subsystem cascade with sensor dropout + recovery sequencing

Grading philosophy
------------------
All graders are *pure functions* of (state, action) — no LLM-as-judge.
This guarantees determinism across runs and hardware.

Score contract
--------------
Every score returned by this module is strictly inside the open interval (0, 1).
Exact 0.0 and 1.0 are never returned. clamp_score() enforces this at one
central point — the last line of compute_reward — so no downstream consumer
needs to worry about boundary values.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

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

# ---------------------------------------------------------------------------
# Score clamping — single authoritative function
# ---------------------------------------------------------------------------

_SCORE_MIN = 0.0001   # strictly > 0.0
_SCORE_MAX = 0.9999   # strictly < 1.0


def clamp_score(value: float) -> float:
    """
    Clamp a raw reward to the open interval (_SCORE_MIN, _SCORE_MAX).
    Called once at the end of compute_reward so every score leaving
    this module satisfies the validator's strict (0, 1) requirement.

    Handles all four boundary routes:
      1. All-zero components (no agent actions) -> _SCORE_MIN not 0.0
      2. FP-driven negatives                   -> _SCORE_MIN not 0.0
      3. round(very_small, 4) == 0.0           -> _SCORE_MIN not 0.0
      4. Perfect easy/medium run               -> _SCORE_MAX not 1.0
    """
    return max(_SCORE_MIN, min(_SCORE_MAX, float(value)))


# ---------------------------------------------------------------------------
# Correct recovery recommendations per anomaly
# ---------------------------------------------------------------------------

ANOMALY_RECOMMENDATIONS: Dict[str, RecommendationType] = {
    "battery_undervoltage":        RecommendationType.REDUCE_POWER,
    "solar_array_degradation":     RecommendationType.REDUCE_POWER,
    "gyro_bias_drift":             RecommendationType.ATTITUDE_HOLD,
    "comms_signal_dropout":        RecommendationType.ISOLATE_COMMS,
    "reaction_wheel_bearing_wear": RecommendationType.ATTITUDE_HOLD,
    "thermal_cpu_overload":        RecommendationType.THERMAL_VENT,
    "propulsion_leak":             RecommendationType.SHUTDOWN_THRUSTER,
    "power_attitude_cascade":      RecommendationType.SAFE_MODE,
    "comms_thermal_combined":      RecommendationType.SAFE_MODE,
}


# ---------------------------------------------------------------------------
# Task descriptors
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Component scorers
# (return raw floats in [0,1]; clamping happens only in compute_reward)
# ---------------------------------------------------------------------------

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
    1.0 for exact primary sensor match, 0.5 for correct subsystem / wrong sensor.
    """
    if not state.anomaly_id:
        return 0.0
    anomaly = ANOMALY_BY_ID[state.anomaly_id]
    primary = anomaly["primary_sensor"]

    for flag in flags:
        if flag.get("sensor") == primary:
            return 1.0

    for flag in flags:
        if get_subsystem_of(flag.get("sensor", "")) == state.anomaly_subsystem:
            return 0.5

    return 0.0


def _severity_score(state: SpacecraftState, flags: List[Dict]) -> float:
    """
    Severity accuracy for the primary sensor flag.
    Partial credit (0.3) for adjacent severity level.
    """
    if not state.anomaly_id:
        return 0.0
    anomaly       = ANOMALY_BY_ID[state.anomaly_id]
    primary       = anomaly["primary_sensor"]
    true_severity = anomaly["severity"].value

    _adjacent = {
        "info":     {"info": 1.0, "warning": 0.3, "critical": 0.0},
        "warning":  {"info": 0.3, "warning": 1.0, "critical": 0.3},
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
    # Partial credit: safe_mode for any critical anomaly
    anomaly = ANOMALY_BY_ID[state.anomaly_id]
    if anomaly["severity"] == Severity.CRITICAL and "safe_mode" in recommendations:
        return 0.5
    return 0.0


def _speed_bonus(state: SpacecraftState, max_steps: int) -> float:
    """
    Reward early detection. Linear decay: 1 - step/max_steps.
    Never returns exactly 1.0 — earliest possible detection is step 1
    (step_count is incremented before the flag action executes),
    giving max = 1 - 1/max_steps < 1.0.
    Returns 0.0 when no correct detection has occurred yet.
    """
    if state.first_correct_detection_step is None:
        return 0.0
    ratio = 1.0 - (state.first_correct_detection_step / max_steps)
    return max(0.0, ratio)


def _false_positive_penalty(state: SpacecraftState, flags: List[Dict]) -> float:
    """
    Penalize flags on sensors outside the affected set.
    0.1 per false-positive flag, capped at 0.5.
    """
    if not state.anomaly_id:
        return 0.0
    affected = set(ANOMALY_BY_ID[state.anomaly_id]["affected"])
    fp_count = sum(1 for f in flags if f.get("sensor") not in affected)
    return min(0.5, fp_count * 0.1)


# ---------------------------------------------------------------------------
# Per-task composite grader
# ---------------------------------------------------------------------------

def compute_reward(
    state: SpacecraftState,
    task: Dict,
    flags: List[Dict],
    recommendations: List[str],
    action: Optional[SpacecraftAction] = None,
    escalated: bool = False,
) -> "RewardBreakdown":  # noqa: F821 — imported at call site
    """
    Compute the shaped reward for the current state.
    Called every step (dense signal, not just terminal).

    Returns a RewardBreakdown whose .total is strictly inside (0, 1) —
    never exactly 0.0 or 1.0 — enforced by clamp_score().
    """
    from ..models import RewardBreakdown  # local import avoids circular dependency

    difficulty = task["difficulty"]
    max_steps  = task["max_steps"]

    det = _detection_score(state, flags)
    loc = _localization_score(state, flags)
    sev = _severity_score(state, flags)
    act = _action_score(state, recommendations) if task.get("require_recommendation") else 0.0
    spd = _speed_bonus(state, max_steps)
    fp  = _false_positive_penalty(state, flags)

    if difficulty == "easy":
        raw = (
            0.50 * det +
            0.25 * loc +
            0.15 * sev +
            0.10 * spd -
            0.10 * fp
        )
    elif difficulty == "medium":
        raw = (
            0.25 * det +
            0.25 * loc +
            0.15 * sev +
            0.25 * act +
            0.10 * spd -
            0.15 * fp
        )
    else:  # hard
        esc = 1.0 if escalated else 0.0
        raw = (
            0.20 * det +
            0.20 * loc +
            0.15 * sev +
            0.20 * act +
            0.10 * esc +
            0.05 * spd -
            0.20 * fp
        )

    # Single clamping point — never returns 0.0 or 1.0
    total = clamp_score(raw)

    return RewardBreakdown(
        detection=round(det, 4),
        localization=round(loc, 4),
        severity=round(sev, 4),
        action=round(act, 4),
        speed_bonus=round(spd, 4),
        fp_penalty=round(fp, 4),
        total=round(total, 4),
    )


def sample_task_anomaly(task_id: str, seed: Optional[int] = None) -> str:
    """Deterministically sample an anomaly_id for a task."""
    task = TASK_BY_ID[task_id]
    pool = task["anomaly_pool"]
    rng  = random.Random(seed)
    return rng.choice(pool)
