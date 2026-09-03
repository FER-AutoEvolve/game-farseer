from __future__ import annotations
import asyncio
import dataclasses
import datetime
import enum
import json
import logging
import argparse
import os
import re
import threading
import time
import uuid
from typing import Any, List

import pydantic
from pydantic import Field

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from core import Result, Unit              
from level_design import build_prompt, call_llm, LLMPlan, PromptingModels
from configuration import FastApiConfiguration
from engagement_strategies import ENGAGEMENT_STRATEGIES
import keypoint_notification


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
    is_level_successfully_completed: bool = dataclasses.field(default=False) # Whether the level was completed successfully
    user_rated_difficulty: int | None = Field(default=None, ge=1, le=10) # 1-10 scale, 1 = easiest, 10 = hardest
    snake_speed: float | None = dataclasses.field(default=1.0)      # current snake speed
    obstacles_count: int | None = dataclasses.field(default=0)      # number of obstacles in the level
    #ovo sam dodala!!!
    food_position: FoodPos | None = dataclasses.field(default=None)
    wall_pattern: WallPattern | None = dataclasses.field(default=None)
    wall_blocks: int = dataclasses.field(default=0)
    level_tile_map: List[List[str]] | None = dataclasses.field(default=None)  # 2D array representing the level's tile map
    is_food_next_to_wall_at_death: bool | None = dataclasses.field(default=None) # Whether the food was next to a wall at the time of death
    average_riskiness: float | None = dataclasses.field(default=None)  # Average riskiness score during the level (near walls and body)
    best_food_directness: float | None = dataclasses.field(default=None)  # Average directness to food during the level

    @property
    def food_completion_ratio(self) -> float | None:
        """
        Returns how much of the total food was collected (0.0–1.0),
        or None if max_food_available is not provided.
        """
        if not self.max_food_available or self.max_food_available <= 0:
            return None
        return min(self.total_food_collected / self.max_food_available, 1.0)
    
current_engagement_strategies = []

@dataclasses.dataclass(frozen=False)
class ApiServer:
    '''
    Snake Auto-Designer API server that dynamically suggests the next game level
    based on player telemetry and LLM-generated plans.
    '''
    _apiConfiguration: FastApiConfiguration
    _logger: logging.Logger = dataclasses.field(default=logging.getLogger())
    _app: FastAPI = dataclasses.field(default_factory=lambda: FastAPI(
        title="Snake Auto-Designer API (LLM-driven)",
        version="2.1"
    ), init=False)
    _server: uvicorn.Server | None = dataclasses.field(default=None)
    _server_thread: threading.Thread | None = dataclasses.field(default=None, init=False)

    def start_server(self) -> Result[Unit]:
        '''
        Starts the FastAPI server in separate thread.
        Returns:
            Result[Unit]: Result indicating success or failure.
        '''
        try:
            self._define_endpoints()
            self._app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"]
            )
            server_config = uvicorn.Config(app=self._app, host=self._apiConfiguration.host, port=self._apiConfiguration.port, log_level=self._logger.level)
            self._server = uvicorn.Server(config=server_config)
            self._server_thread = threading.Thread(target=self._server.run, daemon=True)
            self._server_thread.start()
            return Result.ok(Unit())
        except Exception as e:
            return Result.err(str(e))
        
    def wait_for_server_to_stop(self) -> Result[Unit]:
        '''
        Waits for the FastAPI server to stop.
        Returns:
            Result[Unit]: Result indicating success or failure.
        '''
        try:
            if self._server_thread and self._server_thread.is_alive():
                self._server_thread.join()
            return Result.ok(Unit())
        except Exception as e:
            return Result.err(str(e))

    def stop_server(self) -> Result[Unit]:
        '''
        Stops the FastAPI server
        Returns:
            Result[Unit]: Result indicating success or failure.
        '''
        try:
            # stop the server
            if self._server:
                self._server.should_exit = True
            if self._server_thread:
                self._server_thread.join()
            return Result.ok(Unit())
        except Exception as e:
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
            return {"status": "healthy"}

        @self._app.post("/suggest-level")
        async def _suggest_level(t: Telemetry = Body(...)):
            '''
            Triggers level suggestion pipeline asynchronously.
            Args:
                t (Telemetry): Telemetry data sent from the game client.
            Returns:
                dict: Trigger acknowledgement.
            '''
            request_id = str(uuid.uuid4())[:8]
            self._logger.info("[req=%s] suggest-level start", request_id)
            self._logger.keypoint("Received game telemetry for level suggestion. Triggering async processing...", event_type=keypoint_notification.EventTypes.INFO)

            worker = threading.Thread(
                target=self._run_suggest_level_trigger,
                args=(t, request_id),
                daemon=True,
                name=f"suggest-level-{request_id}",
            )
            worker.start()

            return {
                "status": "accepted",
                "request_id": request_id,
                "message": "suggest-level processing triggered",
            }

    def _run_suggest_level_trigger(self, t: Telemetry, request_id: str) -> None:
        try:
            asyncio.run(self._process_suggest_level_request(t, request_id))
        except Exception:
            self._logger.exception("[req=%s] Background trigger execution failed", request_id)

    async def _process_suggest_level_request(self, t: Telemetry, request_id: str) -> None:
        t0 = time.perf_counter()
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
                "success": t.is_level_successfully_completed,
                "score": t.score,
                "food_collected": t.total_food_collected,
                "avg_time_between_food_sec": t.average_time_to_food,
                "food_per_min": food_per_min,
                "food_completion_ratio": t.food_completion_ratio,
                "max_food_available": t.max_food_available,
                "turn_frequency": t.turn_frequency,
                "total_turns": t.total_turns,
                "total_distance_traveled": t.total_distance_traveled,
                "path_efficiency": t.path_efficiency,
                "user_rated_difficulty": t.user_rated_difficulty,
                "is_food_next_to_wall_at_death": t.is_food_next_to_wall_at_death,
                "average_riskiness": t.average_riskiness,
                "best_food_directness": t.best_food_directness,
            }

            global current_engagement_strategies
            # 3) Current config
            current_config = {
                "snake_speed": t.snake_speed,
                "obstacles_count": t.obstacles_count,
                "food_position": (t.food_position.value if t.food_position else "normal"),
                "wall_pattern": (t.wall_pattern.value if t.wall_pattern else "random"),
                "wall_blocks": t.wall_blocks,
                "level_tile_map": t.level_tile_map,
                "current_engagement_strategies": current_engagement_strategies,
            }

            # 4) Limits
            limits = {
                "snake_speed": {"min": 0.6, "max": 2.0, "rel_min": 0.85, "rel_max": 1.15},
                "obstacles_count": {"min": 0, "max": 30, "delta_min": -3, "delta_max": 3},
                "food_position": {"enum": ["normal", "near_wall", "far_from_wall"]},
                "wall_pattern": {"enum": ["random", "letter"]},
                "wall_blocks": {"min": 0, "max": 30, "delta_min": -8, "delta_max": 8},
            }

            # 5) Build prompt and call LLM
            cfg = self._load_local_config()
            api_key = cfg["Llm"].get("ApiKey", "")
            model = PromptingModels(cfg["Llm"].get("Model"))
            url = cfg["Llm"].get("Url", "")
            headers = cfg["Llm"].get("Headers", {})
            temperature = float(cfg["Llm"].get("Temperature", 0.5))
            timeout_seconds = float(cfg["Llm"].get("TimeoutSeconds", 120.0))

            prompt = build_prompt(telemetry_summary, current_config, limits)
            self._logger.keypoint("Prompt for directive built. Calling LLM to get strategic directive...", event_type=keypoint_notification.EventTypes.INFO)

            llm_res = await call_llm(
                prompt,
                url=url,
                api_key=api_key,
                model=model,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                headers=headers,
                request_id=request_id,
            )
            if llm_res.is_err():
                self._logger.error("[req=%s] LLM error: %s", request_id, llm_res.message)
                self._logger.keypoint(f"Failed to get directive from LLM: {llm_res.message}", event_type=keypoint_notification.EventTypes.FAILURE)
                return

            plan, narrative, directive = llm_res.value
            self._logger.info("[req=%s] plan objective=%s actions=%d", request_id, plan.objective, len(plan.actions))

            current_engagement_strategies = []
            selected_engagement_strategies = re.findall(r'\[ENGAGEMENT_STRATEGY:([^\]]+)\]', directive)
            if selected_engagement_strategies:
                for strategy_list in selected_engagement_strategies:
                    strategies = [s.strip() for s in strategy_list.split(',')]
                    current_engagement_strategies.extend(strategies)

            self._logger.keypoint(f"LLM call completed. Received the following directive: {directive}", event_type=keypoint_notification.EventTypes.SUCCESS)

            # Preserve summary generation for logs/consumers that may parse keypoint streams.
            narrative_with_coeffs = self._append_coeffs_to_narrative(narrative, plan)

            based_on = t.current_level_id or "unknown"
            new_id = t.next_level_id or (f"{based_on}-next" if based_on != "unknown" else "next")

            # 6) Send directive to Code Overseer (if configured)
            cfg = self._load_local_config()
            overseer_configured = bool(cfg.get("CodeOverseerConfigured", True))
            self._logger.keypoint("Sending the strategic directive to Code Overseer...", event_type=keypoint_notification.EventTypes.INFO)

            overseer_result = self._send_directive_to_overseer(directive)
            if overseer_result.is_err():
                if overseer_configured:
                    self._logger.warning("[req=%s] Failed to send directive to Code Overseer: %s", request_id, overseer_result.message)
                else:
                    self._logger.warning("[req=%s] Overseer optional, continuing: %s", request_id, overseer_result.message)
                self._logger.keypoint(f"Failed to send directive to Code Overseer: {overseer_result.message}", event_type=keypoint_notification.EventTypes.FAILURE)
            else:
                self._logger.info("[req=%s] Overseer ok - directive successfully sent to Code Overseer.", request_id)
                self._logger.keypoint("Directive successfully sent to Code Overseer. My job is done!", event_type=keypoint_notification.EventTypes.SUCCESS)

            dt = (time.perf_counter() - t0) * 1000
            self._logger.info("[req=%s] suggest-level trigger completed -> %s (%.1f ms)", request_id, new_id, dt)
            self._logger.debug("[req=%s] narrative_with_coeffs: %s", request_id, narrative_with_coeffs)

        except Exception:
            self._logger.exception("[req=%s] Unhandled error in background suggest-level processing", request_id)



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

    @staticmethod
    def _expand_engagement_strategy_tags(directive: str) -> str:
        '''
        Replaces engagement strategy tags with explicit strategy descriptions.
        Example: [ENGAGEMENT_STRATEGY:ADD_POISON]
        '''
        pattern = r'\[ENGAGEMENT_STRATEGY:([^\]]+)\]'

        def _replace(match: re.Match[str]) -> str:
            keys = [k.strip() for k in match.group(1).split(',') if k.strip()]
            resolved: List[str] = []
            unresolved: List[str] = []

            for key in keys:
                description = ENGAGEMENT_STRATEGIES.get(key)
                if description:
                    resolved.append(f"{key}: {description}")
                else:
                    unresolved.append(key)

            if not resolved and not unresolved:
                return ""

            parts: List[str] = []
            if resolved:
                parts.append("Engagement strategies -> " + " | ".join(resolved))
            if unresolved:
                parts.append("Unknown strategies -> " + ", ".join(unresolved))
            return " ".join(parts)

        return re.sub(pattern, _replace, directive)

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
        directive_for_overseer = self._expand_engagement_strategy_tags(directive)

        try:
            response = requests.post(
                f"{overseer_endpoint}",
                json={"ChangeStrategicDescription": directive_for_overseer},
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
    config_obj = json.loads(open(args.config, "r", encoding="utf-8").read())

    fastapi_config = FastApiConfiguration(host=config_obj.get("Host", "0.0.0.0"), port=config_obj.get("Port", 8000))
    keypoint_notification_config = keypoint_notification.KeypointNotificationConfiguration.from_dict(config_obj.get("KeypointNotification", {}))
    if keypoint_notification_config.is_err():
        logging.warning("KeypointNotification config error: %s", keypoint_notification_config.message)
    else:
        keypoint_notification.configure_keypoint_notifier(keypoint_notification_config.value)

    server = ApiServer(fastapi_config)
    server._config_path = args.config  # save for reference

    logging.info("Starting server using config file: %s", args.config)

    result = server.start_server()
    if result.is_err():
        print(f"Failed to start API server: {result.message}")

    server.wait_for_server_to_stop()

    logging.info("Server has stopped.")
    logging.info("Exiting application.")