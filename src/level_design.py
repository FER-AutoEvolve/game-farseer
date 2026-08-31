# --------------------------------------------
# LLM SAM odlučuje treba li HARDER/EASIER/TUNE isključivo iz telemetrije.
# Vraća tri odjeljka: DIRECTIVE (1–2 rečenice), NARRATIVE (3–6 bulletsa), ACTIONS_JSON (strogi JSON).
# Dozvoljene mete: snake_speed, obstacles_count, food_position, wall_pattern, wall_blocks.
# Napomene o završetku runde: nema “života”; runda završava uspjehom (time/food target) ili
# neuspjehom (wall, self, timeout).
# --------------------------------------------

from __future__ import annotations
from typing import Any, Dict, List, Literal, Tuple, Union
import json
import re
import logging
import time
from enum import Enum

from pydantic import BaseModel, Field, ValidationError
from fastapi import HTTPException
from core import Result, Unit
from engagement_strategies import ENGAGEMENT_STRATEGIES

try:
    from openai import OpenAI  # OpenAI Python SDK v1.x
except Exception:  # pragma: no cover
    OpenAI = None

# --- module logger for LLM calls ---
log = logging.getLogger()

'''
Allowed targets that the LLM is permitted to adjust
'''
Target = Literal[
    "snake_speed",
    "obstacles_count",
    "food_position",
    "wall_pattern",
    "wall_blocks",
]

Mode = Literal["relative", "absolute", "delta", "set_enum"]  # Available adjustment modes for each target
Objective = Literal["HARDER", "EASIER", "TUNE"]              # Possible overall difficulty directions inferred by the LLM
FoodPos = Literal["normal", "near_wall", "far_from_wall"]    # Possible food placement strategies
WallPattern = Literal["random", "letter"]                    # Allowed styles for wall placement


class PromptingModels(Enum):
    ''' Enumeration of supported LLM providers.'''
    OPENAI-GPT-4_1 = "openai-gpt-4_1"
    OPENAI-GPT-5 = "openai-gpt-5"
    GPT_OSS_20B = "gpt_oss_20b"
    GPT_OSS_120B = "gpt_oss_120b"
    QWEN_CODER_30B = "qwen_coder_30b"
    GEMMA_4_31B_QAT = "gemma_4_31b_qat"
    GEMMA_4_26B_A4B_QAT = "gemma_4_26b_a4b_qat"
    QWEN_3_6_35B_A3B = "qwen_3_6_35b_a3b"
    QWEN_3_6_27B = "qwen_3_6_27b"
    NEMOTRON_3_SUPER = "nemotron_3_super"

    def get_model_name(self) -> str:
        '''Returns the default model name for the given provider.'''
        return _DEFAULT_PROMPTING_MODELS.get(self)


_DEFAULT_PROMPTING_MODELS: Dict[PromptingModels, str] = {
    PromptingModels.OPENAI-GPT-4_1: "gpt-4.1",
    PromptingModels.OPENAI-GPT-5: "gpt-5",
    PromptingModels.GPT_OSS_20B: "openai/gpt-oss-20b",
    PromptingModels.GPT_OSS_120B: "openai/gpt-oss-120b",
    PromptingModels.QWEN_CODER_30B: "qwen/qwen-coder-30b",
    PromptingModels.GEMMA_4_31B_QAT: "google/gemma-4-31b-qat",
    PromptingModels.GEMMA_4_26B_A4B_QAT: "google/gemma-4-26b-a4b-qat",
    PromptingModels.QWEN_3_6_35B_A3B: "qwen/qwen3.6-35b-a3b",
    PromptingModels.QWEN_3_6_27B: "qwen/qwen3.6-27b",
    PromptingModels.NEMOTRON_3_SUPER: "nemotron/nemotron-3-super",
}



def log_token_usage(logger: logging.Logger, response: object, provider_name: str) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        logger.debug(f"{provider_name} token usage unavailable")
        return

    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    input_details = getattr(usage, "input_tokens_details", None)
    cached_tokens = getattr(input_details, "cached_tokens", None) if input_details is not None else None

    logger.info(
        f"{provider_name} token usage: input={input_tokens}, output={output_tokens}, total={total_tokens}, cached_input={cached_tokens}"
    )


def extract_token_usage(response: object) -> Dict[str, int | None]:
    '''
    Extract token usage details from an LLM response object.
    '''
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "cached_input_tokens": None,
        }

    input_details = getattr(usage, "input_tokens_details", None)
    cached_tokens = getattr(input_details, "cached_tokens", None) if input_details is not None else None

    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "cached_input_tokens": cached_tokens,
    }


def _extract_response_text(resp: Any) -> str | None:
    '''Extracts plain text from OpenAI Responses/ChatCompletions style outputs.'''
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    if isinstance(resp, dict):
        data = resp
    else:
        try:
            data = resp.to_dict() if hasattr(resp, "to_dict") else None
        except Exception:
            data = None

    if not isinstance(data, dict):
        return None

    text = data.get("output_text")
    if isinstance(text, str) and text.strip():
        return text

    output = data.get("output")
    if isinstance(output, list):
        chunks: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_text = part.get("text") or part.get("output_text")
                    if isinstance(part_text, str) and part_text.strip():
                        chunks.append(part_text)
            item_text = item.get("text")
            if isinstance(item_text, str) and item_text.strip():
                chunks.append(item_text)
        if chunks:
            return "\n".join(chunks).strip()

    choices = data.get("choices")
    if isinstance(choices, list):
        chunks: List[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                chunks.append(content)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_text = part.get("text")
                    if isinstance(part_text, str) and part_text.strip():
                        chunks.append(part_text)
        if chunks:
            return "\n".join(chunks).strip()

    return None

class LLMAction(BaseModel):
    '''
    Represents a single change the LLM suggests to adjust the game configuration.
    Args:
        target (Target): The game parameter to adjust (e.g., "snake_speed").
        mode (Mode): The type of adjustment ("relative", "absolute", "delta", "set_enum").
        value (Union[float, int, FoodPos, WallPattern, str]): The value to apply for the chosen mode.
    Returns:
        LLMAction: A validated Pydantic model representing one configuration change.
    '''
    target: Target
    mode: Mode
    value: Union[float, int, FoodPos, WallPattern, str]


class LLMPlan(BaseModel):
    '''
    Represents the full adjustment plan proposed by the LLM.
    Args:
        objective (Objective): The difficulty direction ("HARDER", "EASIER", "TUNE").
        actions (List[LLMAction]): A list of one to three suggested actions.
        rationale (str): A short explanation for the chosen actions (up to 500 characters).
    Returns:
        LLMPlan: A validated Pydantic model with the overall objective and actions.
    '''
    objective: Objective
    actions: List[LLMAction] = Field(min_length=0, max_length=3)
    rationale: str = Field(max_length=500)


def _examples_block() -> List[Dict[str, Any]]:
    '''
    Provides example scenarios for the LLM prompt to demonstrate the expected format.
    Args:
        None
    Returns:
        List[Dict[str, Any]]: A list of example inputs and corresponding directives, narratives, and action plans.
    '''
    return [
        {
            "when": "fast completions with high food-per-minute and few collisions → room to raise challenge",
            "DIRECTIVE": "Increase snake speed by 10% and add one obstacle to raise spatial pressure.",
            "NARRATIVE": [
                "High food/min and short completion time show strong control.",
                "Small speed bump punishes late turns without spikes.",
                "One extra obstacle increases routing complexity slightly.",
            ],
            "ACTIONS_JSON": {
                "objective": "HARDER",
                "actions": [
                    {"target": "snake_speed", "mode": "relative", "value": 1.1},
                    {"target": "obstacles_count", "mode": "delta", "value": 1},
                ],
                "rationale": "Incremental pressure via speed and space shaping.",
            },
        },
        {
            "when": "short sessions with low food; many edge/self collisions → navigation/timing issues",
            "DIRECTIVE": "Reduce base speed by 10% and place food away from walls to minimize wall impacts.",
            "NARRATIVE": [
                "Frequent wall collisions indicate poor edge control.",
                "Lower speed improves reaction timing.",
                "Far-wall food placement reduces direct risk zones."
            ],
            "ACTIONS_JSON": {
                "objective": "EASIER",
                "actions": [
                    {"target": "snake_speed", "mode": "relative", "value": 0.9},
                    {"target": "food_position", "mode": "set_enum", "value": "far_from_wall"}
                ],
                "rationale": "Lower speed and central food reduce wall collision frequency."
            },
        },
        {
            "when": "steady completions at moderate pace; gameplay feels monotonous",
            "DIRECTIVE": "Add an 8-block random wall; keep speed unchanged",
            "NARRATIVE": [
                "Stable performance allows lateral variety.",
                "Short random wall adds fresh routing decisions.",
                "Maintaining speed preserves overall pacing.",
            ],
            "ACTIONS_JSON": {
                "objective": "TUNE",
                "actions": [
                    {"target": "wall_pattern", "mode": "set_enum", "value": "random"},
                    {"target": "wall_blocks", "mode": "absolute", "value": 8},
                ],
                "rationale": "Variety injection with minimal pace change.",
            },
        },
        {
            "when": "timeouts with medium food collected; collisions are rare → encourage decisive routing",
            "DIRECTIVE": "Keep speed; add a short wall (6 blocks) to funnel routes and increase intent.",
            "NARRATIVE": [
                "Low collision rate means control is fine.",
                "Timeouts indicate indecision or long paths.",
                "A small wall nudges more direct routes.",
            ],
            "ACTIONS_JSON": {
                "objective": "TUNE",
                "actions": [
                    {"target": "wall_pattern", "mode": "set_enum", "value": "random"},
                    {"target": "wall_blocks", "mode": "absolute", "value": 6},
                ],
                "rationale": "Spatial guidance without speed stress.",
            },
        },
    ]


def build_prompt(
    telemetry_summary: Dict[str, Any],
    current_config: Dict[str, Any],
    limits: Dict[str, Dict[str, float]],
) -> str:
    '''
    Builds a structured prompt for the LLM to generate a new game configuration.
    Args:
        telemetry_summary (Dict[str, Any]): Summarized telemetry data from the last game session.
        current_config (Dict[str, Any]): Current configuration values of the game.
        limits (Dict[str, Dict[str, float]]): Allowed ranges or constraints for each adjustable parameter.
    Returns:
        str: A fully formatted prompt combining input data, constraints, and examples for the LLM.
    '''
    constraints = {
        "targets": {
            "snake_speed": {
                "modes": {
                    "relative": f"multiplier in [{limits['snake_speed']['rel_min']}, {limits['snake_speed']['rel_max']}]",
                    "absolute": f"value in [{limits['snake_speed']['min']}, {limits['snake_speed']['max']}]",
                }
            },
            "obstacles_count": {
                "modes": {
                    "delta": f"integer step in [{int(limits['obstacles_count']['delta_min'])}, {int(limits['obstacles_count']['delta_max'])}]",
                    "absolute": f"integer in [{int(limits['obstacles_count']['min'])}, {int(limits['obstacles_count']['max'])}]",
                }
            },
            "food_position": {
                "modes": {"set_enum": "one of ['normal','near_wall','far_from_wall']"}
            },
            "wall_pattern": {
                "modes": {"set_enum": "one of ['random','letter']"}
            },
            "wall_blocks": {
                "modes": {
                    "delta": f"integer step in [{int(limits['wall_blocks']['delta_min'])}, {int(limits['wall_blocks']['delta_max'])}]",
                    "absolute": f"integer in [{int(limits['wall_blocks']['min'])}, {int(limits['wall_blocks']['max'])}]",
                }
            },
        },
        "global": [
           "Decide objective (HARDER/EASIER/TUNE) solely from telemetry; do NOT ask.",
            "Return EXACTLY three sections in order: DIRECTIVE, NARRATIVE, ACTIONS_JSON.",
            "Use at most 3 actions.",
            "Stay within all limits; prefer incremental changes (no big spikes).",
            "DIRECTIVE: 1–2 sentences, English, programmer-friendly.",
            "NARRATIVE: 3–6 bullet points, <= 600 chars total, reference telemetry (e.g., duration, food/min, collisions, death_reason=wall/self/timeout).",
            "Allowed targets only: snake_speed, obstacles_count, food_position, wall_pattern, wall_blocks.",
            "Judge difficulty using derived metrics if present (food_per_min high/low, completion near 1.0).",
            "When objective=HARDER (and optionally when TUNE), pick 1–2 keys from engagement_strategies.",
            "Do not repeat engagement strategies listed in current_config.current_engagement_strategies.",
            "If current_config.current_engagement_strategies is not empty, pick a different one for variety.",
            "Never use 'none' as engagement strategy.",
            "Include them in DIRECTIVE as [ENGAGEMENT_STRATEGY:KEY,...].",
            "Numbers in ACTIONS_JSON must be numbers (no quotes); enums are strings; no markdown fences.",
        ],
    }

    header = (
        "You are a creative but bounded level designer for a Snake game.\n"
        "There are no 'lives'-a session ends by success (time/food target) or failure (wall, self, timeout).\n"
        "Infer difficulty direction from telemetry (HARDER/EASIER/TUNE).\n"
        "You may modify ONLY: snake_speed, obstacles_count, food_position, wall_pattern, wall_blocks.\n"
        "Choose engagement strategies from the provided list if raising difficulty is required.\n"
        "Output format (exactly these sections):\n\n"
        "DIRECTIVE:\n"
        "- A single, actionable instruction (1–2 sentences, English) suitable for a programmer.\n\n"
        "NARRATIVE:\n"
        "- 3–6 bullet points (<= 600 chars total) explaining what/why using telemetry.\n\n"
        "ACTIONS_JSON:\n"
        + json.dumps(
            {
                "objective": "HARDER|EASIER|TUNE",
                "actions": [{"target": "...", "mode": "...", "value": "..."}],
                "rationale": "<=200 chars",
            },
            ensure_ascii=False,
        )
        + "\n\n"
    )

    input_block = {
        "telemetry": telemetry_summary,
        "derived": {
            "food_per_min": telemetry_summary.get("food_per_min"),
            "food_completion_ratio": telemetry_summary.get("food_completion_ratio"), 
            "max_food_available": telemetry_summary.get("max_food_available"),
        },
        "current_config": current_config,
        "limits": limits,
        "allowed_targets": list(constraints["targets"].keys()),
        "engagement_strategies": ENGAGEMENT_STRATEGIES,
        "notes": [
            "Decide objective from telemetry.",
            "Use derived metrics if present (food_per_min, completion).",
            "At most 3 actions.",
            "Stay within limits; prefer smaller steps first.",
            "Choose an engagement strategy in the directive if raising the difficulty is needed.",
            "The level tile map symbols are: '#'=wall, '.'=empty, 'F'=food, 'H'=snake head, 'B'=snake body.",
        ],
    }

    prompt = (
        header
        + f"INPUT:\n{json.dumps(input_block, ensure_ascii=False)}\n\n"
        + f"CONSTRAINTS:\n{json.dumps(constraints, ensure_ascii=False)}\n\n"
        + f"EXAMPLES:\n{json.dumps(_examples_block(), ensure_ascii=False)}\n"
    )
    return prompt


def parse_llm_triplet(text: str) -> Tuple[str, str, dict]:
    '''
    Extracts the DIRECTIVE, NARRATIVE, and ACTIONS_JSON sections from the LLM’s raw text output.
    Args:
        text (str): The raw string response from the LLM.
    Returns:
        Tuple[str, str, dict]: A tuple containing:
            - directive (str): Concise instruction for the developer.
            - narrative (str): Explanation of decisions in bullet points.
            - plan_dict (dict): The JSON-parsed action plan.
    Raises:
        ValueError: If one of the required sections cannot be found or parsed.
    '''
    m = re.search(r"ACTIONS_JSON:\s*({.*})\s*$", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        raise ValueError("ACTIONS_JSON block not found in LLM output.")
    actions_json_str = m.group(1).strip()

    pre_json = text[: m.start()].strip()

    dir_m = re.search(r"^\s*DIRECTIVE:\s*(.+?)\s*(?:\n\s*\n|$)", pre_json, flags=re.DOTALL | re.IGNORECASE)
    if not dir_m:
        raise ValueError("DIRECTIVE block not found.")
    directive = dir_m.group(1).strip()

    narrative_part = pre_json[dir_m.end():].strip()
    narrative = re.sub(r"^\s*NARRATIVE:\s*", "", narrative_part, flags=re.IGNORECASE).strip()

    plan_dict = json.loads(actions_json_str)
    return directive, narrative, plan_dict


async def call_llm(
    prompt: str,
    model: PromptingModels,
    url: str|None = None,
    api_key: str|None = None,
    headers: Dict[str, str]|None = None,
    temperature: float = 0.5,
    timeout_seconds: float = 120.0,
    request_id: str | None = None,
) -> Result[tuple[LLMPlan, str, str, Dict[str, Any]]]:
    '''
    Sends the prepared prompt to the LLM via the OpenAI API and returns the parsed result.
    Args:
        prompt (str): The formatted prompt containing telemetry, constraints, and examples.
        api_key (str): The OpenAI API key for authentication.
        model (PromptingModels): The name of the LLM model to use (default: "gpt-4.1").
        temperature (float, optional): Sampling temperature to control output randomness (default: 0.5).
        request_id (str | None, optional): An optional identifier for logging purposes.
    Returns:
        tuple[LLMPlan, str, str, Dict[str, Any]]: A tuple containing:
            - plan (LLMPlan): The validated action plan with difficulty objective and actions.
            - narrative (str): The explanatory narrative section.
            - directive (str): A concise instruction summarizing the adjustment goal.
            - metadata (Dict[str, Any]): Raw response text and token usage.
    Raises:
        HTTPException: If API key is missing, SDK is unavailable, or response parsing fails.
    '''

    model_name = model.get_model_name()

    timeout_seconds = max(1.0, float(timeout_seconds))
    client = OpenAI(api_key=api_key, base_url=url, default_headers=headers, timeout=timeout_seconds)

    #log before LLM call
    prompt_len = len(prompt) if isinstance(prompt, str) else 0
    log.info("[req=%s] LLM request start model=%s temp=%.2f timeout=%.1fs prompt_len=%d",
             request_id, model_name, temperature, timeout_seconds, prompt_len)
    t0 = time.perf_counter()

    try:
        resp = client.responses.create(
            model=model_name,
            input=prompt,
            temperature=temperature,
        )
        log_token_usage(log, resp, provider_name=f"model={model.value} ({model_name})")
        token_usage = extract_token_usage(resp)
        
        text = _extract_response_text(resp)

        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            dt = (time.perf_counter() - t0) * 1000
            log.error("[req=%s] LLM empty output after %.1f ms", request_id, dt)
            return Result.err("LLM returned empty text.")
            #raise HTTPException(status_code=502, detail="LLM returned empty text.")

        directive, narrative, plan_dict = parse_llm_triplet(text)
        plan = LLMPlan.model_validate(plan_dict)  
        dt = (time.perf_counter() - t0) * 1000
        out_len = len(text)
        log.info("[req=%s] LLM ok objective=%s actions=%d out_len=%d (%.1f ms)",
                 request_id, plan.objective, len(plan.actions), out_len, dt)
        
        metadata = {
            "response_text": text,
            "tokens": token_usage,
        }
        return Result.ok((plan, narrative, directive, metadata))

    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        dt = (time.perf_counter() - t0) * 1000
        log.error("[req=%s] LLM invalid output: %s (%.1f ms)", request_id, e, dt)
        return Result.err(f"Invalid LLM output: {e}")

    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000
        log.exception("[req=%s] LLM request failed after %.1f ms: %s", request_id, dt, e)
        return Result.err(f"LLM request failed: {e}")