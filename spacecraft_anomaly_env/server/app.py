"""
FastAPI Server
Exposes the SpacecraftAnomalyEnvironment over HTTP.

Endpoints (OpenEnv spec §3.1):
  POST /reset        — initialize episode, returns SpacecraftObservation
  POST /step         — execute action, returns SpacecraftObservation
  GET  /state        — returns SpacecraftState (ground truth metadata)
  GET  /health       — liveness probe
  GET  /tasks        — list available tasks
  GET  /docs         — OpenAPI documentation (FastAPI built-in)
  GET  /             — redirect to /docs

Environment variables:
  TASK_ID            — default task (task_easy | task_medium | task_hard)
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from ..models import SpacecraftAction, SpacecraftObservation, SpacecraftState
from .spacecraft_environment import SpacecraftAnomalyEnvironment
from .tasks import TASKS

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None
  
# App factory (matches OpenEnv create_app pattern

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# App factory (matches OpenEnv create_app pattern)
# ---------------------------------------------------------------------------

def create_app(task_id: Optional[str] = None) -> FastAPI:
    _task_id = task_id or os.getenv("TASK_ID", "task_easy")

    env = SpacecraftAnomalyEnvironment(task_id=_task_id)

    app = FastAPI(
        title="Spacecraft Anomaly Detection OpenEnv",
        description=(
            "Real-world spacecraft telemetry anomaly detection environment "
            "for RL agent training. Implements OpenEnv RFC-002 spec."
        ),
        version="1.0.0",
    )

    # Health 
    @app.get("/health")
    def health():
        return {"status": "ok", "task_id": _task_id}

    @app.get("/")
    def root():
        return RedirectResponse(url="/docs")

    # Task listing 

    @app.get("/tasks")
    def list_tasks():
        return {
            "tasks": [
                {
                    "id": t["id"],
                    "name": t["name"],
                    "difficulty": t["difficulty"],
                    "description": t["description"],
                    "max_steps": t["max_steps"],
                    "expected_baseline_score": t["expected_baseline_score"],
                }
                for t in TASKS
            ]
        }

    # reset 

    @app.get("/reset", response_model=SpacecraftObservation)
    def reset(req: ResetRequest = ResetRequest()):
        nonlocal env
        tid = req.task_id or _task_id
        try:
            env = SpacecraftAnomalyEnvironment(task_id=tid)
            obs = env.reset(seed=req.seed)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return obs

    # step 

    @app.post("/step", response_model=SpacecraftObservation)
    def step(action: SpacecraftAction):
        try:
            obs = env.step(action)
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return obs

    # state

    @app.get("/state", response_model=SpacecraftState)
    def state():
        return env.state

    return app


# Default app instance (used by uvicorn entrypoint)

def _get_app():
    """Lazy factory so tests that stub FastAPI don't crash at import."""
    return create_app()

# Only materialise at module level when running under uvicorn (not during tests)
import os as _os
if _os.environ.get("SPACECRAFT_ENV_TESTING") != "1":
    app = _get_app()
