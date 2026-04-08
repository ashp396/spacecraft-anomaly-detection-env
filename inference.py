"""
inference.py — Baseline Agent for Spacecraft Anomaly Detection OpenEnv
Runs an LLM agent (via OpenAI-compatible client) against all three tasks
and reports reproducible scores using the mandatory log format.

Environment variables required:
  API_BASE_URL   — LLM API endpoint  (e.g. https://api-inference.huggingface.co/v1)
  MODEL_NAME     — Model identifier  (e.g. meta-llama/Llama-3.3-70B-Instruct)
  HF_TOKEN       — Hugging Face token / API key

Usage:
  python inference.py
  python inference.py task task_easy
  python inference.py env-url https://huggingface.co/spaces/ashp396/spacecraft-anomaly-detection
  python inference.py seed 42

Log format (mandatory — judges parse these lines):
  [START] {"task_id": ..., "episode": ...}
  [STEP]  {"task_id": ..., "episode": ..., "step": ..., "reward": ..., "done": ..., "action_type": ...}
  [END]   {"task_id": ..., "episode": ..., "final_score": ..., "steps_taken": ...}
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# Configuration

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME:   str = os.environ.get("MODEL_NAME",   "mistralai/Mistral-7B-Instruct-v0.2")
HF_TOKEN:     str = os.environ.get("HF_TOKEN",     "")
ENV_URL:      str = os.environ.get("ENV_URL",      "http://localhost:7860")

MAX_STEPS_PER_TASK = 20  
EPISODES_PER_TASK  = 3    
SLEEP_BETWEEN      = 0.5 

TASKS = ["task_easy", "task_medium", "task_hard"]

# OpenAI client
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# System prompt

SYSTEM_PROMPT = """You are an expert spacecraft mission operations engineer.
You monitor real-time telemetry from a spacecraft with 5 subsystems:
  power, thermal, attitude, comms, propulsion

Each subsystem has sensors reporting numeric values. You must:
1. Analyze the telemetry readings for anomalies
2. Flag the anomalous sensor with the correct severity
3. Recommend the appropriate recovery action
4. Escalate to ground support if needed

Available actions (respond ONLY with valid JSON, no markdown, no explanation):
{
  "action_type": "query_subsystem" | "flag_anomaly" | "clear_flag" | "recommend" | "request_support" | "no_op",
  "subsystem": "<power|thermal|attitude|comms|propulsion>",  // for query_subsystem / request_support
  "sensor": "<sensor_name>",       // for flag_anomaly / clear_flag
  "severity": "info|warning|critical",  // for flag_anomaly
  "recommendation": "no_action|safe_mode|reboot|reduce_power|attitude_hold|thermal_vent|isolate_comms|shutdown_thruster",  // for recommend
  "confidence": 0.0-1.0,
  "rationale": "brief reason"
}

Sensor nominal ranges (flag if reading deviates significantly):
  battery_voltage: ~28.0 V    (safe: 24-32 V)
  solar_array_current: ~8.5 A (safe: 0-12 A)
  power_bus_voltage: ~28.8 V  (safe: 27-30 V)
  battery_temp: ~20 °C        (safe: -5 to 45 °C)
  cpu_temp: ~45 °C            (safe: 10-80 °C)
  fuel_tank_temp: ~15 °C      (safe: -20 to 40 °C)
  gyro_x/y/z: ~0.001 °/s     (safe: -5 to 5 °/s)
  reaction_wheel_rpm: ~1500   (safe: -6000 to 6000)
  signal_strength: ~-85 dBm   (safe: -120 to -60 dBm)
  bit_error_rate: ~1e-7       (safe: 0 to 1e-4)
  transmit_power: ~12 W       (safe: 0-20 W)
  tank_pressure: ~220 bar     (safe: 50-280 bar)
  thruster_temp: ~18 °C       (safe: -30 to 120 °C)
  valve_status: ~0.0          (safe: 0-1)

Sensor value = null means DROPOUT (sensor offline, not anomalous by itself).
Look for CASCADE patterns: a fault in one subsystem often affects others.
"""

# LLM call

def call_llm(messages: List[Dict[str, str]], retry: int = 3) -> str:
    """Call the LLM with retry logic. Returns raw text."""
    for attempt in range(retry):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=300,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
            else:
                return json.dumps({
                    "action_type": "no_op",
                    "rationale": f"LLM error: {str(e)}"
                })


def parse_action(text: str) -> Dict[str, Any]:
    """Extract JSON action from LLM response, with fallback."""
    # Strip markdown fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return {"action_type": "no_op", "rationale": "parse_error"}

# Environment API helpers

def env_reset(env_url: str, task_id: str, seed: Optional[int] = None) -> Dict:
    payload = {"task_id": task_id}
    if seed is not None:
        payload["seed"] = seed
    r = requests.post(f"{env_url}/reset", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(env_url: str, action: Dict) -> Dict:
    r = requests.post(f"{env_url}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


# Single episode runner

def run_episode(
    task_id: str,
    episode_idx: int,
    env_url: str,
    seed: Optional[int] = None,
) -> float:

    obs = env_reset(env_url, task_id, seed=seed)
    episode_id = obs.get("info", {}).get("episode_id", f"ep-{episode_idx}")

    print("[START] " + json.dumps({
        "task_id": task_id,
        "episode": episode_idx,
        "episode_id": episode_id,
        "model": MODEL_NAME,
    }), flush=True)

    conversation: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    final_score = 0.0

    for step_idx in range(MAX_STEPS_PER_TASK):

        tel_str = _format_telemetry(obs.get("telemetry", {}))
        flags_str = _format_flags(obs.get("active_flags", []))

        user_msg = (
            f"Step {obs.get('step_count', step_idx)} | "
            f"Steps remaining: {obs.get('steps_remaining', 0)}\n\n"
            f"Telemetry:\n{tel_str}\n\n"
            f"Active flags: {flags_str}\n\n"
            f"Last result: {obs.get('last_action_result', '')}\n\n"
            f"Analyze and respond with ONE JSON action."
        )

        conversation.append({"role": "user", "content": user_msg})
        raw_response = call_llm(conversation)
        conversation.append({"role": "assistant", "content": raw_response})

        action = parse_action(raw_response)

        obs = env_step(env_url, action)
        reward_obj = obs.get("reward", {})
        step_reward = reward_obj.get("total", 0.0)
        done = obs.get("done", False)
        final_score = step_reward
        safe_score = max(0.0001, min(0.9999, final_score))

        print("[STEP] " + json.dumps({
            "task_id": task_id,
            "episode": episode_idx,
            "step": step_idx + 1,
            "reward": round(step_reward, 4),
            "done": done,
            "action_type": action.get("action_type", "unknown"),
            "reward_breakdown": {
                "detection":    round(reward_obj.get("detection", 0.0), 4),
                "localization": round(reward_obj.get("localization", 0.0), 4),
                "severity":     round(reward_obj.get("severity", 0.0), 4),
                "action":       round(reward_obj.get("action", 0.0), 4),
                "speed_bonus":  round(reward_obj.get("speed_bonus", 0.0), 4),
                "fp_penalty":   round(reward_obj.get("fp_penalty", 0.0), 4),
            }
        }), flush=True)

        time.sleep(SLEEP_BETWEEN)

        if done:
            break

    print("[END] " + json.dumps({
        "task_id": task_id,
        "episode": episode_idx,
        "final_score": round(safe_score, 4),
        "steps_taken": obs.get("step_count", step_idx + 1),
    }), flush=True)

    return final_score

# Formatting helpers

def _format_telemetry(telemetry: Dict) -> str:
    lines = []
    subsystems = {
        "power":      ["battery_voltage", "solar_array_current", "power_bus_voltage"],
        "thermal":    ["battery_temp", "cpu_temp", "fuel_tank_temp"],
        "attitude":   ["gyro_x", "gyro_y", "gyro_z", "reaction_wheel_rpm"],
        "comms":      ["signal_strength", "bit_error_rate", "transmit_power"],
        "propulsion": ["tank_pressure", "thruster_temp", "valve_status"],
    }
    for sub, sensors in subsystems.items():
        lines.append(f"  [{sub.upper()}]")
        for s in sensors:
            val = telemetry.get(s)
            if val is None:
                lines.append(f"    {s}: NULL (dropout)")
            else:
                lines.append(f"    {s}: {val:.4g}")
    return "\n".join(lines)


def _format_flags(flags: List[Dict]) -> str:
    if not flags:
        return "none"
    return "; ".join(
        f"{f['sensor']}({f['severity']})" for f in flags
    )


# Main

def main():
    parser = argparse.ArgumentParser(description="Spacecraft Anomaly Detection : Baseline Inference")
    parser.add_argument("--task",    default=None, help="Single task ID to run (default: all 3)")
    parser.add_argument("--env-url", default=ENV_URL, help="Environment server URL")
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_TASK, help="Episodes per task")
    parser.add_argument("--seed",    type=int, default=42, help="Random seed")
    args = parser.parse_args()

    tasks_to_run = [args.task] if args.task else TASKS

    # Validate env is reachable
    try:
        r = requests.get(f"{args.env_url}/health", timeout=10)
        r.raise_for_status()
        print(json.dumps({"type": "INFO", "message": f"Environment healthy: {r.json()}"}), flush=True)
    except Exception as e:
        print(json.dumps({"type": "ERROR", "message": f"Cannot reach env at {args.env_url}: {e}"}), flush=True)
        sys.exit(1)

    # Validate API credentials
    if not HF_TOKEN:
        print(json.dumps({"type": "WARNING", "message": "HF_TOKEN not set, API calls may fail"}), flush=True)

    summary: Dict[str, float] = {}

    for task_id in tasks_to_run:
        scores = []
        for ep in range(args.episodes):
            seed = args.seed + ep * 100
            try:
                score = run_episode(
                    task_id=task_id,
                    episode_idx=ep,
                    env_url=args.env_url,
                    seed=seed,
                )
                scores.append(score)
            except Exception as e:
                print(json.dumps({
                    "type": "ERROR",
                    "task_id": task_id,
                    "episode": ep,
                    "error": str(e)
                }), flush=True)
                scores.append(0.0)

        avg = round(sum(scores) / len(scores), 4) if scores else 0.0
        safe_avg = max(0.0001, min(0.9999, avg))
        summary[task_id] = safe_avg
        mean = sum(summary.values()) / len(summary) if summary else 0.0
        safe_mean = max(0.0001, min(0.9999, mean))

    # Final summary
    print(json.dumps({
        "type": "SUMMARY",
        "model": MODEL_NAME,
        "episodes_per_task": args.episodes,
        "scores": summary
        "mean_score": round(safe_mean, 4)
    }), flush=True)


if __name__ == "__main__":
    main()
