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


_SYSTEM_TEMPLATE = """You are controlling an Android device to accomplish this goal:

{goal}

Each turn you are shown a screenshot and a numbered list of UI elements. \
Respond with ONLY a single JSON object, no other text, no markdown fences, \
in this exact shape:

{{"action": "<one of: {actions}>", "args": {{...}}, "thought": "<brief reason>", "done": false}}

When the goal is fully accomplished, respond instead with:

{{"action": null, "args": {{}}, "done": true, "result": "<brief description of the outcome>"}}

Action argument shapes:
- click / long_press / input_text: {{"target": <element index>}} (input_text also needs "text")
- press_key: {{"key": "<key name, e.g. back, home>"}}
- swipe: {{"direction": "<up|down|left|right>"}}
- wait_for_delay: {{"seconds": <number>}}
- manage_app: {{"package_name": "<package>", "action": "<launch|close>"}}

Do not include any text outside the single JSON object."""


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
