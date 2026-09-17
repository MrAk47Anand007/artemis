# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prompt construction for the Local profile's reactive loop.

Deliberately plain text, not a LangChain message/tool-declaration pipeline:
the Gallery on-device model has a small context window and is not expected
to support native tool-calling, so the whole contract (goal, available
actions, turn history) is spelled out as instructions the model must answer
with a single JSON object.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TurnRecord:
    """One completed turn, for rendering into the next prompt's history."""

    turn_number: int
    action: str
    args: dict
    outcome: str


_SYSTEM_TEMPLATE = """You are an Android device-control agent. Your ONLY job every turn is to \
decide the next action toward this goal, and output that decision as JSON. You are NOT an \
image-labeling or object-detection tool: never output a "box_2d", a list of detected UI \
elements, or any description of what is on screen. That is not what is being asked of you.

GOAL: {goal}

Each turn you are shown a screenshot and a numbered list of UI elements (given to you as text, \
already extracted -- you do not need to find elements yourself). Based on the goal and what you \
see, decide ONE action and respond with EXACTLY ONE JSON object and NOTHING else: no markdown \
fences, no explanation before or after it, no list, no second JSON object.

The two, and only two, valid response shapes are:

1. To take an action:
{{"action": "<one of: {actions}>", "args": {{...}}, "thought": "<brief reason>", "done": false}}

2. When the goal is fully accomplished:
{{"action": null, "args": {{}}, "done": true, "result": "<brief description of the outcome>"}}

Action argument shapes:
- click / long_press / input_text: {{"target": <element index>}} (input_text also needs "text")
- press_key: {{"key": "<key name, e.g. back, home>"}}
- swipe: {{"direction": "<up|down|left|right>"}}
- wait_for_delay: {{"seconds": <number>}}
- manage_app: {{"package_name": "<package>", "action": "<launch|close>"}}

Worked example. If shown element list:
1: [Icon] Settings
2: [Button] Wi-Fi toggle, currently on
and the goal were "turn Wi-Fi off", the correct entire response is exactly:
{{"action": "click", "args": {{"target": 2}}, "thought": "tapping the Wi-Fi toggle to turn it off", "done": false}}

Reminder: your entire reply must be that one JSON object -- nothing else, and never a "box_2d" \
or any other detection-style output."""


def build_system_prompt(goal: str, allowed_actions: frozenset[str]) -> str:
    """Renders the static system instructions for the Local profile's loop."""
    return _SYSTEM_TEMPLATE.format(goal=goal, actions=", ".join(sorted(allowed_actions)))


def build_turn_history_text(history: list[TurnRecord], max_turns: int = 8) -> str:
    """Renders the bounded recent turn history as plain text.

    Only the last ``max_turns`` entries are rendered: the on-device model's
    context window is small, so unlike Flash's chunked/compressed
    ``TranscriptLedger`` this simply drops older turns rather than
    summarizing them.
    """
    if not history:
        return "No actions taken yet."
    recent = history[-max_turns:]
    return "\n".join(
        f"Turn {t.turn_number}: {t.action}({t.args}) -> {t.outcome}" for t in recent
    )
