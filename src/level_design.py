# --------------------------------------------
# LLM SAM odlučuje treba li HARDER/EASIER/TUNE isključivo iz telemetrije.
# Vraća tri odjeljka: DIRECTIVE (1–2 rečenice), NARRATIVE (3–6 bulletsa), ACTIONS_JSON (strogi JSON).
# Dozvoljene mete: snake_speed, obstacles_count, poison_count, food_position, wall_pattern, wall_blocks.
# Napomene o završetku runde: nema “života”; runda završava uspjehom (time/food target) ili
# neuspjehom (wall, self, poison-at-min-size, timeout).
# --------------------------------------------

from __future__ import annotations
from typing import Any, Dict, List, Literal, Tuple, Union
import json
import re

from pydantic import BaseModel, Field, ValidationError
from fastapi import HTTPException

try:
    from openai import OpenAI  # OpenAI Python SDK v1.x
except Exception:  # pragma: no cover
    OpenAI = None


'''
Allowed targets that the LLM is permitted to adjust
'''
Target = Literal[
    "snake_speed",
    "obstacles_count",
    "poison_count",
    "food_position",
    "wall_pattern",
    "wall_blocks",
]

Mode = Literal["relative", "absolute", "delta", "set_enum"]  #Available adjustment modes for each target
Objective = Literal["HARDER", "EASIER", "TUNE"] #Possible overall difficulty directions inferred by the LLM
FoodPos = Literal["normal", "near_wall", "far_from_wall"] #Possible food placement strategies
WallPattern = Literal["random", "letter"] #Allowed styles for wall placement


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
    actions: List[LLMAction] = Field(min_length=1, max_length=3)
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
            "DIRECTIVE": "Add an 8-block random wall; keep speed unchanged; add one poison.",
            "NARRATIVE": [
                "Stable performance allows lateral variety.",
                "Short random wall adds fresh routing decisions.",
                "A single poison adds zoning without big spike.",
            ],
            "ACTIONS_JSON": {
                "objective": "TUNE",
                "actions": [
                    {"target": "wall_pattern", "mode": "set_enum", "value": "random"},
                    {"target": "wall_blocks", "mode": "absolute", "value": 8},
                    {"target": "poison_count", "mode": "delta", "value": 1},
                ],
                "rationale": "Variety injection with minimal pace change.",
            },
        },
        {
            "when": "poison spiral: snake often at minimal length then hits poison again; food/min is low",
            "DIRECTIVE": "Decrease poison count by 1 and reduce obstacle count by 1 to open recovery space.",
            "NARRATIVE": [
                "Repeated shrink events stall progress.",
                "Fewer poisons lower accidental relapse.",
                "Less clutter opens safer recovery routes.",
            ],
            "ACTIONS_JSON": {
                "objective": "EASIER",
                "actions": [
                    {"target": "poison_count", "mode": "delta", "value": -1},
                    {"target": "obstacles_count", "mode": "delta", "value": -1},
                ],
                "rationale": "Reduce relapse traps and enable rebuild.",
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
            "poison_count": {
                "modes": {
                    "delta": f"integer step in [{int(limits['poison_count']['delta_min'])}, {int(limits['poison_count']['delta_max'])}]",
                    "absolute": f"integer in [{int(limits['poison_count']['min'])}, {int(limits['poison_count']['max'])}]",
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
            "NARRATIVE: 3–6 bullet points, <= 600 chars total, reference telemetry (e.g., duration, food/min, collisions, death_reason=wall/self/poison/timeout).",
            "Allowed targets only: snake_speed, obstacles_count, poison_count, food_position, wall_pattern, wall_blocks.",
            "Judge difficulty using derived metrics if present (food_per_min high/low, completion near 1.0).",
            "Numbers in ACTIONS_JSON must be numbers (no quotes); enums are strings; no markdown fences.",
        ],
    }

    header = (
        "You are a creative but bounded level designer for a Snake game.\n"
        "There are no 'lives'-a session ends by success (time/food target) or failure (wall, self, poison-at-min-size, timeout).\n"
        "Infer difficulty direction from telemetry (HARDER/EASIER/TUNE).\n"
        "You may modify ONLY: snake_speed, obstacles_count, poison_count, food_position, wall_pattern, wall_blocks.\n"
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
        "notes": [
            "Decide objective from telemetry.",
            "Use derived metrics if present (food_per_min, completion).",
            "At most 3 actions.",
            "Stay within limits; prefer smaller steps first.",
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
    api_key: str,
    model: str = "gpt-4.1",
    temperature: float = 0.5,
) -> tuple[LLMPlan, str, str]:
    '''
    Sends the prepared prompt to the LLM via the OpenAI API and returns the parsed result.
    Args:
        prompt (str): The formatted prompt containing telemetry, constraints, and examples.
        api_key (str): The OpenAI API key for authentication.
        model (str, optional): The name of the LLM model to use (default: "gpt-4.1").
        temperature (float, optional): Sampling temperature to control output randomness (default: 0.5).
    Returns:
        tuple[LLMPlan, str, str]: A tuple containing:
            - plan (LLMPlan): The validated action plan with difficulty objective and actions.
            - narrative (str): The explanatory narrative section.
            - directive (str): A concise instruction summarizing the adjustment goal.
    Raises:
        HTTPException: If API key is missing, SDK is unavailable, or response parsing fails.
    '''
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set.")
    if OpenAI is None:
        raise HTTPException(status_code=500, detail="OpenAI SDK is not installed. Run 'pip install openai'.")


    client = OpenAI(api_key=api_key)

    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
        )
        text = getattr(resp, "output_text", None)
        if not text:
            try:
                text = resp.to_dict().get("output_text")
            except Exception:
                text = None

        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            raise HTTPException(status_code=502, detail="LLM returned empty text.")

        directive, narrative, plan_dict = parse_llm_triplet(text)
        plan = LLMPlan.model_validate(plan_dict)  
        return plan, narrative, directive

    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        raise HTTPException(status_code=502, detail=f"Invalid LLM output: {e}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")
