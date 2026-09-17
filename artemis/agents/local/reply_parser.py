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

"""Parses the Gallery on-device model's reply into a single decided action.

The model is not expected to support OpenAI-style native tool-calling, so
the decision protocol is a plain JSON object in the reply text instead --
either bare or fenced in a ```json code block.
"""

from dataclasses import dataclass
import json
import re

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class ParsedAction:
    """One turn's decision, parsed out of the model's raw reply text."""

    action: str | None
    args: dict
    thought: str | None
    done: bool
    result: str | None


def _extract_json_text(text: str) -> str | None:
    fenced = _JSON_BLOCK_RE.search(text)
    if fenced:
        return fenced.group(1)
    bare = _BARE_OBJECT_RE.search(text)
    if bare:
        return bare.group(0)
    return None


def parse_model_reply(
    text: str, allowed_actions: frozenset[str] | None = None
) -> ParsedAction | None:
    """Parses ``text`` into a :class:`ParsedAction`, or ``None`` if it can't be.

    Returns ``None`` (rather than raising) on any of: no JSON object found,
    invalid JSON, missing the required ``done`` field, or (when
    ``allowed_actions`` is given) an ``action`` outside that vocabulary --
    the caller treats all of these as one uniform "bad reply, retry" case.
    """
    json_text = _extract_json_text(text)
    if json_text is None:
        return None

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict) or "done" not in parsed:
        return None

    action = parsed.get("action")
    if allowed_actions is not None and action is not None and action not in allowed_actions:
        return None

    return ParsedAction(
        action=action,
        args=parsed.get("args") or {},
        thought=parsed.get("thought"),
        done=bool(parsed["done"]),
        result=parsed.get("result"),
    )
