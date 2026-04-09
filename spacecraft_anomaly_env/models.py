"""
OpenEnv Typed Models
Pydantic v2 models for the Spacecraft Anomaly Detection environment.

Follows OpenEnv RFC-002 spec:
  - Action    : what the agent sends
  - Observation: what the agent receives (includes reward, done)
  - State     : episode metadata (server-side)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# Action types

class ActionType(str, Enum):
    QUERY_SUBSYSTEM   = "query_subsystem"    # get extra detail on a subsystem
    FLAG_ANOMALY      = "flag_anomaly"       # record a finding
    CLEAR_FLAG        = "clear_flag"         # rescind a false-positive flag
    RECOMMEND         = "recommend"          # issue recovery recommendation
    REQUEST_SUPPORT   = "request_support"    # escalate to ground team
    NO_OP             = "no_op"              # pass / observe only


class SeverityLevel(str, Enum):
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "critical"


class RecommendationType(str, Enum):
    NO_ACTION      = "no_action"
    SAFE_MODE      = "safe_mode"
    REBOOT         = "reboot"
    REDUCE_POWER   = "reduce_power"
    ATTITUDE_HOLD  = "attitude_hold"
    THERMAL_VENT   = "thermal_vent"
    ISOLATE_COMMS  = "isolate_comms"
    SHUTDOWN_THRUSTER = "shutdown_thruster"


class SpacecraftAction(BaseModel):
    """
    Typed action the agent sends to the environment.

    Fields
    ------
    action_type : ActionType
        Which action to perform.
    subsystem : str, optional
        Target subsystem name (for query_subsystem / request_support).
    sensor : str, optional
        Target sensor name (for flag_anomaly / clear_flag).
    severity : SeverityLevel, optional
        Severity assessment when flagging an anomaly.
    recommendation : RecommendationType, optional
        Recovery action when using 'recommend'.
    confidence : float [0.0-1.0]
        Agent's confidence in this action.
    rationale : str, optional
        Free-text justification (not used in grading, helps debugging).
    """

    action_type: ActionType
    subsystem: Optional[str] = None
    sensor: Optional[str] = None
    severity: Optional[SeverityLevel] = None
    recommendation: Optional[RecommendationType] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    rationale: Optional[str] = None

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 4)


# Observation (includes reward per RFC-002 §4.2)

class RewardBreakdown(BaseModel):
    """Granular reward components (all in [0,1] before weighting)."""
    detection:     int = Field(0, ge=0, le=1)  # found an anomaly at all?
    localization:  int = Field(0, ge=0, le=1)  # right subsystem?
    severity:      int = Field(0, ge=0, le=1)  # correct severity?
    action:        int = Field(0, ge=0, le=1)  # recovery action correct?
    speed_bonus:   int = Field(0, ge=0, le=1)  # detected early?
    fp_penalty:    int = Field(0, ge=0, le=1)  # false-positive cost
    total:         int = Field(0, ge=0, le=1)  # weighted composite


class SpacecraftObservation(BaseModel):
    """
    Observation returned by reset() and step().

    Fields
    telemetry : Dict[str, Optional[float]]
        Current sensor readings keyed by sensor name.
        None = sensor dropout (hard task only).
    step_count : int
        Steps elapsed in this episode.
    steps_remaining : int
        Budget of actions left.
    active_flags : List[Dict]
        All anomaly flags the agent has raised this episode.
    last_action_result : str
        Human-readable outcome of the last action.
    subsystem_detail : Dict, optional
        Extra subsystem data returned by query_subsystem.
    reward : RewardBreakdown
        Reward for the *previous* action (0 on reset).
    done : bool
        Episode termination flag.
    success : bool
        OpenEnv spec requires this field.
    info : Dict[str, Any]
        Auxiliary diagnostics (task_id, episode_id, etc.).
    """

    telemetry: Dict[str, Optional[float]]
    step_count: int = 0
    steps_remaining: int = 20
    active_flags: List[Dict[str, Any]] = Field(default_factory=list)
    last_action_result: str = "Episode started."
    subsystem_detail: Optional[Dict[str, Any]] = None
    reward: RewardBreakdown = Field(default_factory=RewardBreakdown)
    done: bool = False
    success: bool = True
    info: Dict[str, Any] = Field(default_factory=dict)


# State (server-side episode metadata)

class SpacecraftState(BaseModel):
    """
    Server-side state returned by state() endpoint.
    Not sent to the agent directly during training.
    """

    episode_id: str = ""
    step_count: int = 0
    task_id: str = ""
    task_difficulty: str = "easy"
    anomaly_id: Optional[str] = None          # ground truth (hidden from agent)
    anomaly_subsystem: Optional[str] = None   # ground truth
    anomaly_severity: Optional[str] = None    # ground truth
    correct_recommendation: Optional[str] = None
    max_steps: int = 20
    dropout_fraction: float = 0.0
    flags_raised: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations_made: List[str] = Field(default_factory=list)
    first_correct_detection_step: Optional[int] = None
    episode_complete: bool = False
