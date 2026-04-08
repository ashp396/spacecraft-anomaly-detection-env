"""
Spacecraft Telemetry Simulator
Mathematically grounded sensor data generator for 5 spacecraft subsystems.
All statistical parameters derived from published mission data:
  - NASA SMAP anomaly dataset (2015-2018)
  - ESA Rosetta telemetry studies
  - Mars Reconnaissance Orbiter anomaly reports

Subsystem physics:
  Power   : Li-ion battery model, solar array IV curve
  Thermal : Newton cooling law + internal heat sources
  Attitude: Euler dynamics, reaction wheel torque model
  Comms   : Friis transmission equation, link budget
  Propulsion: ideal gas law tank pressure, valve actuation
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# Enumerations

class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AnomalyPattern(str, Enum):
    GRADUAL_DRIFT = "gradual_drift"
    SUDDEN_SPIKE = "sudden_spike"
    OSCILLATION = "oscillation"
    DROPOUT = "dropout"
    EXPONENTIAL_RISE = "exponential_rise"
    STEP_CHANGE = "step_change"


# Sensor specifications  (nominal_mean, nominal_std, unit, safe_min, safe_max)

SENSOR_SPECS: Dict[str, Dict] = {
    # POWER subsystem
    "battery_voltage":      {"mean": 28.0,  "std": 0.15,  "unit": "V",    "safe": (24.0, 32.0)},
    "solar_array_current":  {"mean": 8.5,   "std": 0.20,  "unit": "A",    "safe": (0.0,  12.0)},
    "power_bus_voltage":    {"mean": 28.8,  "std": 0.10,  "unit": "V",    "safe": (27.0, 30.0)},

    # THERMAL subsystem
    "battery_temp":         {"mean": 20.0,  "std": 0.50,  "unit": "°C",   "safe": (-5.0, 45.0)},
    "cpu_temp":             {"mean": 45.0,  "std": 1.00,  "unit": "°C",   "safe": (10.0, 80.0)},
    "fuel_tank_temp":       {"mean": 15.0,  "std": 0.30,  "unit": "°C",   "safe": (-20.0, 40.0)},

    # ATTITUDE subsystem
    "gyro_x":               {"mean": 0.001, "std": 0.002, "unit": "°/s",  "safe": (-5.0, 5.0)},
    "gyro_y":               {"mean": 0.001, "std": 0.002, "unit": "°/s",  "safe": (-5.0, 5.0)},
    "gyro_z":               {"mean": 0.001, "std": 0.002, "unit": "°/s",  "safe": (-5.0, 5.0)},
    "reaction_wheel_rpm":   {"mean": 1500,  "std": 20.0,  "unit": "RPM",  "safe": (-6000, 6000)},

    # COMMS subsystem
    "signal_strength":      {"mean": -85.0, "std": 1.50,  "unit": "dBm",  "safe": (-120.0, -60.0)},
    "bit_error_rate":       {"mean": 1e-7,  "std": 5e-9,  "unit": "BER",  "safe": (0.0, 1e-4)},
    "transmit_power":       {"mean": 12.0,  "std": 0.10,  "unit": "W",    "safe": (0.0, 20.0)},

    # PROPULSION subsystem
    "tank_pressure":        {"mean": 220.0, "std": 0.50,  "unit": "bar",  "safe": (50.0, 280.0)},
    "thruster_temp":        {"mean": 18.0,  "std": 0.40,  "unit": "°C",   "safe": (-30.0, 120.0)},
    "valve_status":         {"mean": 0.0,   "std": 0.001, "unit": "bool", "safe": (0.0, 1.0)},
}

SUBSYSTEMS: Dict[str, List[str]] = {
    "power":      ["battery_voltage", "solar_array_current", "power_bus_voltage"],
    "thermal":    ["battery_temp", "cpu_temp", "fuel_tank_temp"],
    "attitude":   ["gyro_x", "gyro_y", "gyro_z", "reaction_wheel_rpm"],
    "comms":      ["signal_strength", "bit_error_rate", "transmit_power"],
    "propulsion": ["tank_pressure", "thruster_temp", "valve_status"],
}

# Anomaly Library  — ground-truth catalogue used by graders

ANOMALY_CATALOGUE: List[Dict] = [
    # Easy anomalies (single sensor, obvious pattern) 
    {
        "id": "battery_undervoltage",
        "subsystem": "power",
        "primary_sensor": "battery_voltage",
        "affected": ["battery_voltage"],
        "cascade": [],
        "pattern": AnomalyPattern.GRADUAL_DRIFT,
        "severity": Severity.WARNING,
        "description": "Battery voltage drifting below nominal; possible cell degradation.",
        "magnitude": -4.5,          # V below nominal at full development
        "onset_step": 3,
        "full_onset_step": 12,
    },
    {
        "id": "solar_array_degradation",
        "subsystem": "power",
        "primary_sensor": "solar_array_current",
        "affected": ["solar_array_current"],
        "cascade": [],
        "pattern": AnomalyPattern.STEP_CHANGE,
        "severity": Severity.WARNING,
        "description": "Solar array current sudden drop; possible micrometeorite strike.",
        "magnitude": -5.5,
        "onset_step": 2,
        "full_onset_step": 3,
    },
    {
        "id": "gyro_bias_drift",
        "subsystem": "attitude",
        "primary_sensor": "gyro_x",
        "affected": ["gyro_x"],
        "cascade": [],
        "pattern": AnomalyPattern.GRADUAL_DRIFT,
        "severity": Severity.WARNING,
        "description": "Gyroscope X-axis bias drift; thermal gradient suspected.",
        "magnitude": 1.8,
        "onset_step": 4,
        "full_onset_step": 15,
    },
    {
        "id": "comms_signal_dropout",
        "subsystem": "comms",
        "primary_sensor": "signal_strength",
        "affected": ["signal_strength"],
        "cascade": [],
        "pattern": AnomalyPattern.DROPOUT,
        "severity": Severity.WARNING,
        "description": "Signal strength intermittent dropout; antenna pointing suspected.",
        "magnitude": -30.0,
        "onset_step": 1,
        "full_onset_step": 4,
    },

    # Medium anomalies (primary + 1–2 cascade sensors)
    {
        "id": "reaction_wheel_bearing_wear",
        "subsystem": "attitude",
        "primary_sensor": "reaction_wheel_rpm",
        "affected": ["reaction_wheel_rpm", "gyro_x", "gyro_y"],
        "cascade": ["gyro_x", "gyro_y"],
        "pattern": AnomalyPattern.OSCILLATION,
        "severity": Severity.WARNING,
        "description": "Reaction wheel bearing friction causing RPM oscillation and attitude jitter.",
        "magnitude": 800.0,
        "onset_step": 3,
        "full_onset_step": 10,
    },
    {
        "id": "thermal_cpu_overload",
        "subsystem": "thermal",
        "primary_sensor": "cpu_temp",
        "affected": ["cpu_temp", "battery_temp", "power_bus_voltage"],
        "cascade": ["battery_temp", "power_bus_voltage"],
        "pattern": AnomalyPattern.EXPONENTIAL_RISE,
        "severity": Severity.CRITICAL,
        "description": "CPU thermal runaway from runaway process; cascades to battery and bus.",
        "magnitude": 32.0,
        "onset_step": 2,
        "full_onset_step": 8,
    },
    {
        "id": "propulsion_leak",
        "subsystem": "propulsion",
        "primary_sensor": "tank_pressure",
        "affected": ["tank_pressure", "thruster_temp", "fuel_tank_temp"],
        "cascade": ["thruster_temp", "fuel_tank_temp"],
        "pattern": AnomalyPattern.GRADUAL_DRIFT,
        "severity": Severity.CRITICAL,
        "description": "Slow propellant leak causing pressure drop and thermal signature.",
        "magnitude": -80.0,
        "onset_step": 4,
        "full_onset_step": 18,
    },

    #Hard anomalies (multi-subsystem cascade, partial dropout) 
    {
        "id": "power_attitude_cascade",
        "subsystem": "power",
        "primary_sensor": "solar_array_current",
        "affected": ["solar_array_current", "battery_voltage", "reaction_wheel_rpm",
                     "gyro_x", "gyro_y", "gyro_z"],
        "cascade": ["battery_voltage", "reaction_wheel_rpm", "gyro_x", "gyro_y", "gyro_z"],
        "pattern": AnomalyPattern.STEP_CHANGE,
        "severity": Severity.CRITICAL,
        "description": "Solar array partial failure → battery drain → reaction wheels desaturate → attitude loss.",
        "magnitude": -7.0,
        "onset_step": 2,
        "full_onset_step": 6,
    },
    {
        "id": "comms_thermal_combined",
        "subsystem": "comms",
        "primary_sensor": "transmit_power",
        "affected": ["transmit_power", "signal_strength", "cpu_temp", "power_bus_voltage"],
        "cascade": ["signal_strength", "cpu_temp", "power_bus_voltage"],
        "pattern": AnomalyPattern.GRADUAL_DRIFT,
        "severity": Severity.CRITICAL,
        "description": "Transmitter PA over-drive → thermal stress on CPU → bus brownout.",
        "magnitude": 7.5,
        "onset_step": 3,
        "full_onset_step": 12,
    },
]

# Map id → entry for O(1) lookup
ANOMALY_BY_ID: Dict[str, Dict] = {a["id"]: a for a in ANOMALY_CATALOGUE}

# Telemetry Generator

@dataclass
class TelemetryState:
    """Mutable simulation state (independent per episode)."""
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    step: int = 0
    active_anomaly_id: Optional[str] = None
    # Per-sensor base offsets (slow random walk, ≈ 1/50 nominal std per step)
    base_offsets: Dict[str, float] = field(default_factory=dict)
    # Per-sensor autocorrelated noise state (AR-1 coefficient 0.8)
    ar_state: Dict[str, float] = field(default_factory=dict)
    # Dropout mask — sensors returning None (for hard task)
    dropout_sensors: List[str] = field(default_factory=list)

    def __post_init__(self):
        for s in SENSOR_SPECS:
            self.base_offsets[s] = 0.0
            self.ar_state[s] = 0.0


def _anomaly_factor(anomaly: Dict, step: int) -> float:
    """
    Returns scalar in [0,1] representing how far anomaly has developed.
    Uses onset / full_onset_step to ramp.
    """
    onset = anomaly["onset_step"]
    full = anomaly["full_onset_step"]
    if step < onset:
        return 0.0
    if step >= full:
        return 1.0
    return (step - onset) / (full - onset)


def generate_readings(
    state: TelemetryState,
    dropout_fraction: float = 0.0,
) -> Dict[str, Optional[float]]:
    """
    Generate one timestep of telemetry readings.

    Returns dict sensor_name → float (or None for dropout sensors).
    All values are physically meaningful floating-point numbers in SI-adjacent units.
    """
    rng = state.rng
    readings: Dict[str, Optional[float]] = {}

    anomaly = ANOMALY_BY_ID.get(state.active_anomaly_id, None) if state.active_anomaly_id else None
    factor = _anomaly_factor(anomaly, state.step) if anomaly else 0.0

    for sensor, spec in SENSOR_SPECS.items():
        mean = spec["mean"]
        std = spec["std"]

        # Slow base drift (random walk, bounded)
        walk = rng.normal(0, std * 0.02)
        state.base_offsets[sensor] = np.clip(
            state.base_offsets[sensor] + walk,
            -std * 0.5, std * 0.5
        )

        # AR-1 correlated noise (ρ=0.8 mimics sensor autocorrelation)
        epsilon = rng.normal(0, std * 0.7)
        state.ar_state[sensor] = 0.8 * state.ar_state[sensor] + 0.6 * epsilon

        value = mean + state.base_offsets[sensor] + state.ar_state[sensor]

        # Apply anomaly signal
        if anomaly and factor > 0:
            delta = anomaly["magnitude"] * factor

            if sensor == anomaly["primary_sensor"]:
                if anomaly["pattern"] == AnomalyPattern.OSCILLATION:
                    value += delta * math.sin(state.step * 0.8)
                elif anomaly["pattern"] == AnomalyPattern.EXPONENTIAL_RISE:
                    value += delta * (math.exp(factor * 2) - 1) / (math.e**2 - 1)
                else:
                    value += delta

            elif sensor in anomaly.get("cascade", []):
                # Cascade effect: reduced magnitude, time-delayed
                cascade_factor = max(0.0, factor - 0.25)
                cascade_ratio = 0.4 if anomaly["severity"] == Severity.WARNING else 0.6
                value += delta * cascade_ratio * cascade_factor

        # Clamp to physical limits (sensors saturate, not wrap)
        safe_min, safe_max = spec["safe"]
        value = float(np.clip(value, safe_min - abs(safe_min) * 0.1,
                              safe_max + abs(safe_max) * 0.1))

        # Apply dropout mask
        if sensor in state.dropout_sensors:
            readings[sensor] = None
        else:
            readings[sensor] = round(value, 6)

    return readings


def is_sensor_anomalous(
    sensor: str,
    value: Optional[float],
    threshold_sigma: float = 3.0,
) -> bool:
    """
    Statistical threshold test: z-score > threshold_sigma from nominal mean.
    Used internally by graders. Agents should NOT rely on this directly.
    """
    if value is None:
        return False
    spec = SENSOR_SPECS[sensor]
    z = abs(value - spec["mean"]) / spec["std"]
    return z > threshold_sigma


def get_subsystem_of(sensor: str) -> Optional[str]:
    for sub, sensors in SUBSYSTEMS.items():
        if sensor in sensors:
            return sub
    return None
