# Spacecraft Anomaly Detection OpenEnv

An OpenEnv environment that simulates real-world spacecraft mission operations. AI agents analyse streaming telemetry from 5 spacecraft subsystems, detect faults, classify severity, and recommend recovery procedures, exactly as a human flight operations engineer does.

Backed by statistical parameters from published mission data (NASA SMAP anomaly dataset, ESA Rosetta telemetry studies, MRO anomaly reports).


## Why this domain

Spacecraft anomaly detection is a real operational task performed 24 hours a day at NASA, ESA, ISRO, and commercial operators. Human operators monitor hundreds of sensor channels simultaneously, reason about cascading faults under time pressure, and issue recovery commands. This environment faithfully models that workflow:

- **Physically grounded** sensor simulation (Li-ion battery model, Newton cooling, reaction wheel dynamics, Friis link budget, ideal gas tank pressure)
- **Cascade anomaly** physics: a primary fault propagates to secondary sensors with time delay and reduced magnitude, just like real hardware failures
- **Sensor dropout** on the hard task (18% of sensors return `null`) simulating communication gaps or sensor failures
- **Deterministic graders** no LLM-as-judge, no ambiguity


## Architecture

```
spacecraft_anomaly_env/
├── __init__.py                  # public API exports
├── models.py                    # Pydantic typed models (Action, Observation, State)
├── openenv.yaml                 # OpenEnv spec metadata
├── inference.py                 # baseline inference script (mandatory name)
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── server/
│   ├── __init__.py
│   ├── app.py                   # FastAPI server (create_app factory)
│   ├── spacecraft_environment.py# Core environment: reset/step/state
│   ├── telemetry.py             # Physics simulator + anomaly catalogue
│   └── tasks.py                 # Task definitions + graders
└── tests/
    └── test_environment.py      # Full test suite
```

---

## Subsystems & Sensors

| Subsystem | Sensors | Nominal | Unit |
|-----------|---------|---------|------|
| Power | battery_voltage | 28.0 | V |
| | solar_array_current | 8.5 | A |
| | power_bus_voltage | 28.8 | V |
| Thermal | battery_temp | 20.0 | °C |
| | cpu_temp | 45.0 | °C |
| | fuel_tank_temp | 15.0 | °C |
| Attitude | gyro_x/y/z | 0.001 | °/s |
| | reaction_wheel_rpm | 1500 | RPM |
| Comms | signal_strength | -85.0 | dBm |
| | bit_error_rate | 1e-7 | BER |
| | transmit_power | 12.0 | W |
| Propulsion | tank_pressure | 220.0 | bar |
| | thruster_temp | 18.0 | °C |
| | valve_status | 0.0 | bool |

Telemetry uses AR-1 autocorrelated noise (ρ=0.8) and a slow random walk for base drift — matching real sensor behaviour.


## Action Space

```json
{
  "action_type": "query_subsystem | flag_anomaly | clear_flag | recommend | request_support | no_op",
  "subsystem":       "power | thermal | attitude | comms | propulsion",
  "sensor":          "<sensor_name>",
  "severity":        "info | warning | critical",
  "recommendation":  "no_action | safe_mode | reboot | reduce_power | attitude_hold | thermal_vent | isolate_comms | shutdown_thruster",
  "confidence":      0.0,
  "rationale":       "optional free text"
}
```

## Observation Space

```json
{
  "telemetry":           {"battery_voltage": 27.43, "cpu_temp": null, ...},
  "step_count":          4,
  "steps_remaining":     11,
  "active_flags":        [{"sensor": "battery_voltage", "severity": "warning", "step": 3}],
  "last_action_result":  "Anomaly flag raised: sensor='battery_voltage' ...",
  "reward": {
    "detection":    0.0,
    "localization": 0.5,
    "severity":     0.3,
    "action":       0.0,
    "speed_bonus":  0.6,
    "fp_penalty":   0.0,
    "total":        0.42
  },
  "done":    false,
  "success": true,
  "info":    {"episode_id": "...", "task_id": "task_easy", ...}
}
```

---

## Tasks

### Task 1 : Easy: Single-sensor anomaly identification
- **Max steps**: 10
- **Anomaly pool**: battery undervoltage, solar array degradation, gyro bias drift, comms signal dropout
- **No cascade**, no dropout
- **Grader weights**: detection 50%, localization 25%, severity 15%, speed 10%
- **Expected baseline score**: 0.72

The agent sees telemetry where a single sensor drifts or spikes from nominal. The objective is to identify and flag that sensor with the correct severity.

### Task 2 : Medium: Root-cause analysis with cascade effects
- **Max steps**: 15  
- **Anomaly pool**: reaction wheel bearing wear, CPU thermal runaway, propulsion leak
- **1–2 cascade sensors** affected by secondary effects
- **Grader weights**: detection 25%, localization 25%, severity 15%, action 25%, speed 10%
- **Expected baseline score**: 0.51

The primary fault causes secondary sensors to deviate (with a time lag and reduced magnitude). The agent must identify the **root cause** (not just the most deviated sensor) and recommend the correct recovery action.

### Task 3 : Hard: Multi-subsystem cascade with dropout and escalation
- **Max steps**: 20
- **Anomaly pool**: power-attitude cascade, comms-thermal combined
- **18% sensor dropout** (sensors return `null`)
- **Cross-subsystem cascade** across 4–6 sensors
- **Grader weights**: detection 20%, localization 20%, severity 15%, action 20%, escalation 10%, speed 5%, FP penalty -20%
- **Expected baseline score**: 0.28

A critical fault propagates across multiple subsystems while nearly 1 in 5 sensors are offline. False positive flags are penalized. The agent must also call `request_support` to escalate to the ground team.


## Reward Function

Dense shaped reward computed **every step** (not just at episode end):

```
R_easy   = 0.50·detection + 0.25·localization + 0.15·severity + 0.10·speed - 0.10·FP
R_medium = 0.25·detection + 0.25·localization + 0.15·severity + 0.25·action + 0.10·speed - 0.15·FP
R_hard   = 0.20·detection + 0.20·localization + 0.15·severity + 0.20·action + 0.10·escalation + 0.05·speed - 0.20·FP
```

Each component is in [0, 1]; total is clipped to [0, 1].

**Speed bonus**: `1 - (first_correct_detection_step / max_steps)` — rewards detecting the anomaly early.  
**FP penalty**: `0.1 per false-positive flag`, capped at 0.5 — penalizes flagging normal sensors.


## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/reset` | Initialize episode. Body: `{"task_id": "task_easy", "seed": 42}` |
| POST | `/step` | Execute action. Body: SpacecraftAction JSON |
| GET | `/state` | Server-side state (includes ground truth for evaluation) |
| GET | `/health` | Liveness probe |
| GET | `/tasks` | List all tasks with metadata |
| GET | `/docs` | Interactive OpenAPI docs |


## Setup & Usage

### Local (Python)

```bash
git clone https://github.com/your-username/spacecraft-anomaly-env
cd spacecraft-anomaly-env
pip install -e ".[dev]"

# Start server (default: task_easy)
uvicorn spacecraft_anomaly_env.server.app:app --host 0.0.0.0 --port 7860

# Or a specific task:
TASK_ID=task_hard uvicorn spacecraft_anomaly_env.server.app:app --port 7860
```

### Docker

```bash
# Build
docker build -t spacecraft-anomaly-env .

# Run (default task)
docker run -p 7860:7860 spacecraft-anomaly-env

# Run specific task
docker run -p 7860:7860 -e TASK_ID=task_hard spacecraft-anomaly-env

# Verify health
curl http://localhost:7860/health
```

### Quick API test

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy", "seed": 42}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "flag_anomaly", "sensor": "battery_voltage", "severity": "warning", "confidence": 0.9}'
```


## Running Inference

```bash
# Set credentials
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="your_token_here"
export ENV_URL="http://localhost:7860"

# Run all 3 tasks (3 episodes each)
python inference.py

# Run single task
python inference.py --task task_easy

# Custom seed and episodes
python inference.py --seed 123 --episodes 5
```

The script emits structured stdout logs:
```json
{"type": "[START]", "task_id": "task_easy", "episode": 0}
{"type": "[STEP]",  "task_id": "task_easy", "episode": 0, "step": 1, "reward": 0.0, "done": false, ...}
{"type": "[END]",   "task_id": "task_easy", "episode": 0, "final_score": 0.68, "steps_taken": 8}
{"type": "SUMMARY", "scores": {"task_easy": 0.72, "task_medium": 0.51, "task_hard": 0.28}}
```


## Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```


## Baseline Scores

Scores obtained using `meta-llama/Llama-3.3-70B-Instruct`, 3 episodes per task, seed=42:

| Task | Difficulty | Avg Score |
|------|-----------|-----------|
| task_easy | Easy | 0.72 |
| task_medium | Medium | 0.51 |
| task_hard | Hard | 0.28 |
| **Mean** | — | **0.50** |


## Hugging Face Spaces Deployment

```bash
# Push to your HF namespace
pip install openenv-core
openenv push --repo-id your-username/spacecraft-anomaly-env

# Or deploy manually via HF web UI:
# 1. Create new Space → Docker SDK
# 2. Push this repo to the Space git remote
# 3. Set env var: TASK_ID=task_easy
```

The Space will be available at `https://your-username-spacecraft-anomaly-env.hf.space`.


## Anomaly Catalogue

| ID | Subsystem | Pattern | Severity | Cascade |
|----|-----------|---------|----------|---------|
| battery_undervoltage | power | gradual_drift | warning | none |
| solar_array_degradation | power | step_change | warning | none |
| gyro_bias_drift | attitude | gradual_drift | warning | none |
| comms_signal_dropout | comms | dropout | warning | none |
| reaction_wheel_bearing_wear | attitude | oscillation | warning | gyro_x, gyro_y |
| thermal_cpu_overload | thermal | exponential_rise | critical | battery_temp, power_bus_voltage |
| propulsion_leak | propulsion | gradual_drift | critical | thruster_temp, fuel_tank_temp |
| power_attitude_cascade | power | step_change | critical | battery_voltage, reaction_wheel_rpm, gyro_x/y/z |
| comms_thermal_combined | comms | gradual_drift | critical | signal_strength, cpu_temp, power_bus_voltage |


## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TASK_ID` | `task_easy` | Active task |
| `API_BASE_URL` | HF Inference API | LLM endpoint |
| `MODEL_NAME` | Llama-3.3-70B | Model identifier |
| `HF_TOKEN` | — | HuggingFace / API key |
| `ENV_URL` | `http://localhost:7860` | Env server URL for inference.py |


## License

MIT
