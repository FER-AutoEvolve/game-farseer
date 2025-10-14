from __future__ import annotations
import dataclasses
import datetime
import enum
import json
import logging
import argparse
import os
import time
import uuid
from typing import Any, List

import pydantic
from pydantic import Field

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from core import Result, Unit              
from level_design import build_prompt, call_llm, LLMPlan
from configuration import FastApiConfiguration 


def to_pascal_case(s: str) -> str:
    '''
    Converts a snake_case string to PascalCase.
    '''
    return ''.join(word.capitalize() for word in s.split('_'))

class DeathCauses(enum.Enum):
    '''
    Possible causes of death in the game.
    '''
    WALL = "wall"
    SELF = "self"
    OTHER = "other"

class FoodPos(enum.Enum):
    '''
    Possible food positions in the game.
    '''
    NORMAL = "normal"
    NEAR_WALL = "near_wall"
    FAR_FROM_WALL = "far_from_wall"

class WallPattern(enum.Enum):
    '''
    Possible wall patterns in the game.
    '''
    RANDOM = "random"
    LETTER = "letter"

@pydantic.dataclasses.dataclass(config={"alias_generator": to_pascal_case, "populate_by_name": True}, frozen=True)
class Telemetry:
    '''
    Telemetry data sent from the game client after a level is completed.
    '''
    current_level_id: str | None = dataclasses.field(default=None)  # ID of the level that was just played
    next_level_id: str | None = dataclasses.field(default=None)    # ID of the next level to be played (if known)
    max_food_available: int | None = dataclasses.field(default=None) #Total number of food items available in this level
    start_time: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.now) # When the level started
    end_time: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.now)   # When the level ended
    average_time_to_food: float | None = dataclasses.field(default=None) # Average time in seconds between collecting food items
    score: int = dataclasses.field(default=0) # Final score achieved in the level
    total_food_collected: int = dataclasses.field(default=0) # Total number of food items collected
    total_distance_traveled: int = dataclasses.field(default=0) # Total distance traveled by the snake
    total_turns: int = dataclasses.field(default=0)    # Total number of turns made by the snake
    turn_frequency: float | None = dataclasses.field(default=None) # Average number of turns per minute
    death_cause: DeathCauses | None = dataclasses.field(default=None) # Cause of death if the level was failed
    path_efficiency: float | None = dataclasses.field(default=None)    # Ratio of optimal path length to actual path length (0.0 to 1.0)
    is_level_successful: bool = dataclasses.field(default=False) # Whether the level was completed successfully
    user_rated_difficulty: int | None = Field(default=None, ge=1, le=10) # 1-10 scale, 1 = easiest, 10 = hardest
    snake_speed: float | None = dataclasses.field(default=1.0)      # current snake speed
    obstacles_count: int | None = dataclasses.field(default=0)      # number of obstacles in the level
    poison_count: int | None = dataclasses.field(default=0)         # number of poison items on the map
    #ovo sam dodala!!!
    food_position: FoodPos | None = dataclasses.field(default=None)
    wall_pattern: WallPattern | None = dataclasses.field(default=None)
    wall_blocks: int = dataclasses.field(default=0)

    @property
    def food_completion_ratio(self) -> float | None:
        """
        Returns how much of the total food was collected (0.0–1.0),
        or None if max_food_available is not provided.
        """
        if not self.max_food_available or self.max_food_available <= 0:
            return None
        return min(self.total_food_collected / self.max_food_available, 1.0)

@dataclasses.dataclass(frozen=False)
class ApiServer:
    '''
    Snake Auto-Designer API server that dynamically suggests the next game level
    based on player telemetry and LLM-generated plans.
    '''
    _apiConfiguration: FastApiConfiguration
    _logger: logging.Logger = dataclasses.field(default=logging.getLogger(__name__))
    _app: FastAPI = dataclasses.field(default_factory=lambda: FastAPI(
        title="Snake Auto-Designer API (LLM-driven)",
        version="2.1"
    ), init=False)
    _server: Any = dataclasses.field(default=None)

    def start_server(self) -> Result[Unit]:
        '''
        Starts the FastAPI server.
        Returns:
            Result[Unit]: Result indicating success or failure.
        '''
        try:
            self._app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"]
            )
            self._define_endpoints()
            level_name = logging.getLevelName(self._logger.getEffectiveLevel()).lower()
            self._server = uvicorn.run(
                self._app,
                host=self._apiConfiguration.host,
                port=self._apiConfiguration.port,
                log_level=level_name,
                access_log=True,
            )
            return Result.ok(Unit())
        except Exception as e:
            self._logger.exception("Failed to start server")
            return Result.err(str(e))

    def stop_server(self) -> Result[Unit]:
        '''
        Stops the FastAPI server.
        Returns:
            Result[Unit]: Result indicating success or failure.
        '''
        try:
            if self._server:
                self._server.stop()
            return Result.ok(Unit())
        except Exception as e:
            self._logger.error(f"Failed to stop server: {e}")
            return Result.err(str(e))

    def _define_endpoints(self) -> None:
        '''
        Defines all FastAPI endpoints.
        '''

        import time as _time

        # --- access log middleware (method, path, status, duration) ---
        @self._app.middleware("http")
        async def _log_requests(request, call_next):
            start = _time.perf_counter()
            response = None
            try:
                response = await call_next(request)
                return response
            finally:
                dur_ms = (_time.perf_counter() - start) * 1000
                self._logger.info("%s %s -> %s (%.1f ms)",
                                  request.method,
                                  request.url.path,
                                  getattr(response, "status_code", "?"),
                                  dur_ms)

        @self._app.get("/health")
        async def _health():
            '''
            Healthcheck endpoint.
            Returns:
                dict: { "status": "healthy" }
            '''
            return Result.ok({"status": "healthy"}).__dict__

        @self._app.post("/suggest-level")
        async def _suggest_level(t: Telemetry = Body(...)):
            '''
            Suggests a new game level based on telemetry and LLM recommendations.
            Args:
                t (Telemetry): Telemetry data sent from the game client.
            Returns:
                dict: Suggested new level data including objective and narrative.
            '''
            request_id = str(uuid.uuid4())[:8]
            t0 = time.perf_counter()
            self._logger.info("[req=%s] suggest-level start", request_id)

            try:
                # 1) Calculate level duration
                try:
                    duration_sec = max(
                        0.0,
                        (t.end_time - t.start_time).total_seconds()
                        if isinstance(t.end_time, datetime.datetime) and isinstance(t.start_time, datetime.datetime)
                        else 0.0,
                    )
                except Exception:
                    duration_sec = 0.0

                session_len_sec = duration_sec

                food_per_min = None
                if session_len_sec and session_len_sec > 0:
                    if t.total_food_collected:
                        food_per_min = 60.0 * float(t.total_food_collected) / float(session_len_sec)
                    elif t.average_time_to_food and t.average_time_to_food > 0:
                        food_per_min = 60.0 / float(t.average_time_to_food)

                # 2) Map telemetry for LLM
                telemetry_summary = {
                    "death_reason": t.death_cause.value if t.death_cause else None,
                    "duration_sec": duration_sec,
                    "success": t.is_level_successful,
                    "score": t.score,
                    "food_collected": t.total_food_collected,
                    "avg_time_between_food_sec": t.average_time_to_food,
                    "food_per_min": food_per_min,                       # NOVO
                    "food_completion_ratio": t.food_completion_ratio,   # NOVO (property)
                    "max_food_available": t.max_food_available,         # NOVO
                    "turn_frequency": t.turn_frequency,
                    "total_turns": t.total_turns,
                    "total_distance_traveled": t.total_distance_traveled,
                    "path_efficiency": t.path_efficiency,
                    "user_rated_difficulty": t.user_rated_difficulty,
                }

                # 3) Current config
                current_config = {
                    "snake_speed": t.snake_speed,
                    "obstacles_count": t.obstacles_count,
                    "poison_count": t.poison_count,
                    "food_position": (t.food_position.value if t.food_position else "normal"),
                    "wall_pattern": (t.wall_pattern.value if t.wall_pattern else "random"),
                    "wall_blocks": t.wall_blocks,
                }

                # 4) Limits
                limits = {
                    "snake_speed": {"min": 0.6, "max": 2.0, "rel_min": 0.85, "rel_max": 1.15},
                    "obstacles_count": {"min": 0, "max": 30, "delta_min": -3, "delta_max": 3},
                    "poison_count": {"min": 0, "max": 10, "delta_min": -3, "delta_max": 3},

                    # nova polja
                    "food_position": {"enum": ["normal", "near_wall", "far_from_wall"]},
                    "wall_pattern": {"enum": ["random", "letter"]},
                    "wall_blocks": {"min": 0, "max": 30, "delta_min": -8, "delta_max": 8},
                }

                # 5) Build prompt and call LLM
                cfg = self._load_local_config()
                api_key = cfg.get("OpenAiApiKey", "")
                model = cfg.get("OpenAiModel", "gpt-4.1")
                temperature = float(cfg.get("OpenAiTemperature", 0.5))

                if not api_key:
                        self._logger.error("[req=%s] Missing OpenAI API key", request_id)
                        return Result.err("Missing OpenAI API key.").__dict__

                prompt = build_prompt(telemetry_summary, current_config, limits)
                llm_res = await call_llm(prompt, api_key=api_key, model=model, temperature=temperature, request_id=request_id)
                if llm_res.is_err():
                    self._logger.error("[req=%s] LLM error: %s", request_id, llm_res.message)
                    return Result.err(llm_res.message).__dict__

                plan, narrative, directive = llm_res.value
                self._logger.info("[req=%s] plan objective=%s actions=%d",
                                    request_id, plan.objective, len(plan.actions))


                # 6) Append coefficients
                narrative_with_coeffs = self._append_coeffs_to_narrative(narrative, plan)

                # 7) Level IDs
                based_on = t.current_level_id or "unknown"
                new_id = t.next_level_id or (f"{based_on}-next" if based_on != "unknown" else "next")

                # 8) Send directive to Code Overseer (if configured)
                cfg = self._load_local_config()
                print(cfg)
                overseer_configured = bool(cfg.get("overseer_configured", True))
                
                overseer_result = self._send_directive_to_overseer(directive)
                if overseer_result.is_err():
                    if overseer_configured:
                        self._logger.warning(f"Failed to send directive to Code Overseer: {overseer_result.message}")
                        return Result.err(f"Failed to send directive to Code Overseer: {overseer_result.message}").__dict__
                    else:
                        self._logger.warning("[req=%s] Overseer optional, continuing: %s", request_id, overseer_result.message)
                else:
                    self._logger.info("[req=%s] Overseer ok- directive successfully sent to Code Overseer.", request_id)

                payload = {
                        "based_on_level_id": based_on,
                        "new_level_id": new_id,
                        "objective": plan.objective,
                        "directive": directive,
                        "narrative": narrative_with_coeffs,
                    }

            
                dt = (time.perf_counter() - t0) * 1000
                self._logger.info("[req=%s] suggest-level ok -> %s (%.1f ms)", request_id, new_id, dt)
                return Result.ok(payload).__dict__

            except Exception as e:
                self._logger.exception("[req=%s] Unhandled error in /suggest-level", request_id)
                return Result.err(f"Unexpected error: {e}").__dict__



    def _load_local_config(self) -> dict:
        """
        Loads configuration from a JSON file and environment variables.
        Priority: explicit _config_path -> ENV CONFIG_PATH -> configuration.local.json
        """
        cfg = {}

        config_path = (
            getattr(self, "_config_path", None)
            or os.environ.get("CONFIG_PATH")
            or "configuration.local.json"
        )
        abs_path = os.path.abspath(config_path)

        # log – koji path stvarno koristimo
        self._logger.info("Config path resolved to: %s", abs_path)

        if os.path.exists(abs_path):
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f) or {}
                # log – koje ključeve smo učitali
                self._logger.info("Loaded configuration from: %s (keys: %s)", abs_path, ", ".join(sorted(cfg.keys())))
            except Exception as e:
                self._logger.warning("Failed to load config file %s: %s", abs_path, e)
                cfg = {}
        else:
            self._logger.warning("Config file not found at %s; relying on ENV/defaults", abs_path)

        # ENV override
        if os.environ.get("OPEN_AI_API_KEY"):
            cfg["OpenAiApiKey"] = os.environ["OPEN_AI_API_KEY"]
        if os.environ.get("OPENAI_MODEL"):
            cfg["OpenAiModel"] = os.environ["OPENAI_MODEL"]
        if os.environ.get("CODE_OVERSEER_ENDPOINT"):
            cfg["CodeOverseerEndpoint"] = os.environ["CODE_OVERSEER_ENDPOINT"]
        if os.environ.get("OVERSEER_CONFIGURED"):
            cfg["OverseerConfigured"] = os.environ["OVERSEER_CONFIGURED"].lower() in ("1", "true", "yes")

         
        cfg.setdefault("OpenAiModel", "gpt-4.1")
        cfg.setdefault("OpenAiTemperature", 0.5)

        # još jedan log: ima li API key nakon svega
        self._logger.info("Config check: OpenAiApiKey present = %s", "OpenAiApiKey" in cfg and bool(cfg["OpenAiApiKey"]))
        return cfg


    @staticmethod
    def _format_coeff_line(target: str, mode: str, value) -> str:
        if mode == "relative":
            return f"{target} ×{value}"
        if mode == "delta":
            sign = "+" if isinstance(value, (int, float)) and value >= 0 else ""
            return f"{target} {sign}{value}"    
        return f"{target} = {value}"

    def _append_coeffs_to_narrative(self, narrative: str, plan: LLMPlan) -> str:
        '''
        Appends coefficient details from the LLM plan to the narrative.
        Args:
            narrative (str): The main narrative text.
            plan (LLMPlan): Plan containing actions and coefficients.
        Returns:
            str: Combined narrative with coefficients.
        '''
        lines: List[str] = [narrative.strip(), "", "Coefficients:"]
        for action in plan.actions:
            lines.append(f"- {self._format_coeff_line(action.target, action.mode, action.value)}")
        return "\n".join(lines).strip()

    def _send_directive_to_overseer(self, directive: str) -> Result[Unit]:
        '''
        Sends the strategic directive to the Code Overseer service.
        Args:
            directive (str): The strategic directive to send.
        Returns:
            Result[Unit]: Result indicating success or failure.
        '''
        import requests

        cfg = self._load_local_config()
        overseer_endpoint = cfg.get("CodeOverseerEndpoint", "").strip()
        if not overseer_endpoint:
            return Result.err("Code Overseer Endpoint is not configured.")

        try:
            response = requests.post(
                f"{overseer_endpoint}",
                json={ "ChangeStrategicDescription": f"{directive}" },
                timeout=5000
            )
            if response.status_code == 200:
                return Result.ok(Unit())
            else:
                return Result.err(f"Overseer responded with status code {response.status_code}.")
        except Exception as e:
            return Result.err(f"Failed to send directive to Overseer: {e}")


if __name__ == "__main__":
    '''
    Local test run
    '''

    parser = argparse.ArgumentParser(description="Snake Auto-Designer API")
    parser.add_argument(
        "--config",
        type=str,
        default="configuration.json",
        help="Putanja do JSON konfiguracijske datoteke (default: configuration.json)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    config = FastApiConfiguration(port=int(os.environ.get("PORT", 8000)), host="0.0.0.0")
    server = ApiServer(config)
    server._config_path = args.config  # save for reference

    logging.info("Starting server using config file: %s", args.config)

    result = server.start_server()
    if result.is_err():
        print(f"Failed to start API server: {result.message}")