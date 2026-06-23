from __future__ import annotations

ENGAGEMENT_STRATEGIES: dict[str, str] = {
    "MOVING_WALLS": "Add moving walls that shift to any neighbouring position every 2 seconds to increase unpredictability.",
    "POWER_UP_BLUE": "Introduce power-up tiles (colored blue) that gives 2 points in the score when collected and grows the snake by 2 body parts.",
    "POWER_UP_YELLOW": "Produce power-up tiles (colored yellow) that increase the snake speed temporarily.",
    "ADD_POISON": "Add poison tiles that spawn (colored red) and shrink the snake by 2 units and decrease the score by 2 upon contact, increasing risk.",
    "REVERSE_SNAKE": "Implement a special tile (colored purple) that reverses the snake's direction upon contact. Its head becomes its tail and vice versa.",
    "SOOTHING_FOOD": "Place soothing food items (colored light green) that slow down the snake speed temporarily when collected, allowing for more precise navigation.",
    "DISSAPEARING_WALLS": "Incorporate disappearing walls (colored cyan) that vanish for 30 seconds every 10 seconds, creating temporary safe zones or hazards.",
}