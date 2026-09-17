# mobile_run_task Local Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `mobile_run_task(model="Local", ...)` run an autonomous reactive loop backed entirely by the Google AI Edge Gallery's free on-device model, reusing the existing daemon/trace/inspection stack unchanged.

**Architecture:** A new `LocalRunner` (same external shape as `FlashRunner`) drives `UnifiedMobileController`/`McpActionExecutor` for device actions and `ctx.data_engine` for step recording, but replaces LangChain `ChatOpenAI` + tool-calling with a hand-rolled HTTP call to the Gallery's `/v1/chat/completions` and a JSON-in-prompt decision protocol. It plugs into `Agent.run_task()` as a third profile branch (`"local"`, alongside the existing `"flash"` special-case), so `mobile_run_task`'s daemon dispatch, background spawn, `trace_id`, `mobile_manage_task`, and `mobile_inspect_trace` all work against it with no changes.

**Tech Stack:** Python 3, `httpx` (already a dependency), `pydantic`, `pytest`/`pytest-asyncio`. No new external dependencies.

**Spec:** `docs/superpowers/specs/2026-09-17-local-runner-design.md`

**Repo root for all paths below:** repository root (`C:\Users\Anand\artemis`)

---

### Task 1: `LocalModelConfig` loader

**Files:**
- Create: `artemis/config/local_model.py`
- Test (new): `tests/unit/config/test_local_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/config/test_local_model.py`:

```python
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

import json

from artemis.config.local_model import LocalModelConfig, parse_local_model_config


def test_defaults_when_key_missing(tmp_path, monkeypatch):
    config_path = tmp_path / "artemis.jsonc"
    config_path.write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setenv("ARTEMIS_ARTEMIS_JSONC", str(config_path))

    config = parse_local_model_config()

    assert config == LocalModelConfig(
        api_base="http://127.0.0.1:8080/v1",
        token=None,
        model="local-model",
        max_turns=40,
    )


def test_overrides_from_file(tmp_path, monkeypatch):
    config_path = tmp_path / "artemis.jsonc"
    config_path.write_text(
        json.dumps(
            {
                "local_model": {
                    "api_base": "http://192.168.1.50:8080/v1",
                    "token": "secret-token",
                    "model": "Gemma-4-E2B-it",
                    "max_turns": 25,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARTEMIS_ARTEMIS_JSONC", str(config_path))

    config = parse_local_model_config()

    assert config == LocalModelConfig(
        api_base="http://192.168.1.50:8080/v1",
        token="secret-token",
        model="Gemma-4-E2B-it",
        max_turns=25,
    )


def test_partial_override_keeps_other_defaults(tmp_path, monkeypatch):
    config_path = tmp_path / "artemis.jsonc"
    config_path.write_text(
        json.dumps({"local_model": {"model": "Gemma-4-E4B-it"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARTEMIS_ARTEMIS_JSONC", str(config_path))

    config = parse_local_model_config()

    assert config.model == "Gemma-4-E4B-it"
    assert config.api_base == "http://127.0.0.1:8080/v1"
    assert config.max_turns == 40
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/config/test_local_model.py -v`
Expected: `ModuleNotFoundError: No module named 'artemis.config.local_model'`

- [ ] **Step 3: Implement `LocalModelConfig`**

Create `artemis/config/local_model.py`:

```python
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

"""Configuration for the Local profile's Gallery on-device model endpoint.

Deliberately separate from ``artemis.config.llm.LLMConfig``: that type is
shaped around constructing a LangChain chat model (provider dispatch,
fallback chains, per-node overrides), and the Local profile never constructs
one -- it talks to the Gallery's OpenAI-compatible HTTP server directly.
"""

from pydantic import BaseModel

from artemis.config.paths import ROOT_DIR, get_config_path
from artemis.utils.file import load_jsonc
from artemis.utils.logger import get_logger

logger = get_logger(__name__)


class LocalModelConfig(BaseModel):
    """Connection details for the Gallery local API server."""

    api_base: str = "http://127.0.0.1:8080/v1"
    token: str | None = None
    model: str = "local-model"
    max_turns: int = 40


def parse_local_model_config() -> LocalModelConfig:
    """Parses the ``local_model`` block out of ``artemis.jsonc``/``artemis.json``.

    Missing file or missing key both fall back to :class:`LocalModelConfig`'s
    defaults rather than raising -- unlike ``LLMConfig``, there is nothing to
    validate credentials for, so a fully-defaulted config is always usable
    against a Gallery server running on its default port.
    """
    config_path = None
    for candidate in ("artemis.jsonc", "artemis.json"):
        try:
            config_path = get_config_path(candidate)
            break
        except FileNotFoundError:
            continue

    if not config_path:
        config_path = get_config_path("artemis.jsonc", ROOT_DIR / "artemis.jsonc")

    try:
        with open(config_path, encoding="utf-8") as f:
            config_dict = load_jsonc(f)
    except Exception as e:
        logger.warning(f"Failed to load {config_path} for local_model config: {e}")
        return LocalModelConfig()

    return LocalModelConfig.model_validate(config_dict.get("local_model", {}))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/config/test_local_model.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add artemis/config/local_model.py tests/unit/config/test_local_model.py
git commit -m "feat(local): add LocalModelConfig loader for the Gallery endpoint"
```

---

### Task 2: Reply parser

**Files:**
- Create: `artemis/agents/local/__init__.py`
- Create: `artemis/agents/local/reply_parser.py`
- Test (new): `tests/unit/agents/local/__init__.py`
- Test (new): `tests/unit/agents/local/test_reply_parser.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/agents/local/__init__.py` (empty file).

Create `tests/unit/agents/local/test_reply_parser.py`:

```python
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

from artemis.agents.local.reply_parser import ParsedAction, parse_model_reply


def test_bare_json_object():
    result = parse_model_reply('{"action": "click", "args": {"target": 3}, "done": false}')
    assert result == ParsedAction(action="click", args={"target": 3}, thought=None, done=False, result=None)


def test_json_fenced_in_code_block():
    text = 'Here is my decision:\n```json\n{"action": "press_key", "args": {"key": "back"}, "done": false}\n```'
    result = parse_model_reply(text)
    assert result == ParsedAction(
        action="press_key", args={"key": "back"}, thought=None, done=False, result=None
    )


def test_done_with_result():
    text = '{"action": null, "args": {}, "done": true, "result": "Wi-Fi is now on"}'
    result = parse_model_reply(text)
    assert result.done is True
    assert result.result == "Wi-Fi is now on"


def test_thought_field_is_carried():
    text = '{"action": "click", "args": {"target": 1}, "thought": "tapping the toggle", "done": false}'
    result = parse_model_reply(text)
    assert result.thought == "tapping the toggle"


def test_malformed_json_returns_none():
    assert parse_model_reply("not json at all") is None


def test_json_missing_required_fields_returns_none():
    assert parse_model_reply('{"foo": "bar"}') is None


def test_action_outside_allowed_vocabulary_returns_none():
    result = parse_model_reply(
        '{"action": "ask_explorer", "args": {}, "done": false}',
        allowed_actions=frozenset({"click", "press_key"}),
    )
    assert result is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/agents/local/test_reply_parser.py -v`
Expected: `ModuleNotFoundError: No module named 'artemis.agents.local'`

- [ ] **Step 3: Implement the parser**

Create `artemis/agents/local/__init__.py` (empty file).

Create `artemis/agents/local/reply_parser.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/agents/local/test_reply_parser.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add artemis/agents/local/__init__.py artemis/agents/local/reply_parser.py \
        tests/unit/agents/local/__init__.py tests/unit/agents/local/test_reply_parser.py
git commit -m "feat(local): add JSON reply parser for the Local profile's decision loop"
```

---

### Task 3: Prompt builder

**Files:**
- Create: `artemis/agents/local/prompt.py`
- Test (new): `tests/unit/agents/local/test_prompt.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/agents/local/test_prompt.py`:

```python
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

from artemis.agents.local.prompt import TurnRecord, build_system_prompt, build_turn_history_text


def test_system_prompt_includes_goal_and_actions():
    prompt = build_system_prompt(
        goal="Open Settings and toggle Wi-Fi",
        allowed_actions=frozenset({"click", "press_key"}),
    )
    assert "Open Settings and toggle Wi-Fi" in prompt
    assert "click" in prompt
    assert "press_key" in prompt
    assert "JSON" in prompt


def test_turn_history_empty():
    assert build_turn_history_text([]) == "No actions taken yet."


def test_turn_history_renders_each_turn():
    history = [
        TurnRecord(turn_number=1, action="click", args={"target": 3}, outcome="success"),
        TurnRecord(turn_number=2, action="press_key", args={"key": "back"}, outcome="success"),
    ]
    text = build_turn_history_text(history)
    assert "Turn 1: click({'target': 3}) -> success" in text
    assert "Turn 2: press_key({'key': 'back'}) -> success" in text


def test_turn_history_caps_to_last_n_turns():
    history = [
        TurnRecord(turn_number=i, action="press_key", args={"key": "back"}, outcome="success")
        for i in range(1, 21)
    ]
    text = build_turn_history_text(history, max_turns=8)
    assert "Turn 13:" in text
    assert "Turn 20:" in text
    assert "Turn 12:" not in text
    assert "Turn 1:" not in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/agents/local/test_prompt.py -v`
Expected: `ModuleNotFoundError: No module named 'artemis.agents.local.prompt'`

- [ ] **Step 3: Implement the prompt builder**

Create `artemis/agents/local/prompt.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/agents/local/test_prompt.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add artemis/agents/local/prompt.py tests/unit/agents/local/test_prompt.py
git commit -m "feat(local): add prompt builder for the Local profile's decision loop"
```

---

### Task 4: `LocalRunner`

**Files:**
- Create: `artemis/agents/local/runner.py`
- Test (new): `tests/unit/agents/local/test_runner.py`

`LocalRunner` takes its device-observation, action-execution, and model-call
steps as injectable callables (defaulting to the real device/HTTP
implementations) specifically so this task's tests can drive the loop's
control flow — retry-on-bad-JSON, `done` handling, `max_turns` exhaustion —
with fakes, no real device or network needed. Real-device behavior is
verified manually in Task 8.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/agents/local/test_runner.py`:

```python
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

import pytest

from artemis.agents.local.runner import LocalRunner
from artemis.config.local_model import LocalModelConfig


class _FakeCtx:
    data_engine = None


def _make_runner(call_model_replies, max_turns=10, act_results=None):
    replies = iter(call_model_replies)
    acted: list[tuple[str, dict]] = []

    async def fake_observe():
        return b"fake-screenshot-bytes", "1: [Button] Wi-Fi toggle"

    async def fake_call_model(messages):
        return next(replies)

    async def fake_act(action, args):
        acted.append((action, args))
        if act_results:
            return act_results.pop(0)
        return "success"

    runner = LocalRunner(
        ctx=_FakeCtx(),
        goal="Toggle Wi-Fi",
        config=LocalModelConfig(max_turns=max_turns),
        observe_fn=fake_observe,
        call_model_fn=fake_call_model,
        act_fn=fake_act,
    )
    return runner, acted


@pytest.mark.asyncio
async def test_completes_when_model_reports_done():
    runner, acted = _make_runner(
        ['{"action": "click", "args": {"target": 1}, "done": false}',
         '{"action": null, "args": {}, "done": true, "result": "Wi-Fi is on"}']
    )

    result = await runner.run()

    assert result == {"status": "completed", "result": "Wi-Fi is on"}
    assert acted == [("click", {"target": 1})]


@pytest.mark.asyncio
async def test_retries_once_on_bad_json_then_succeeds():
    runner, acted = _make_runner(
        ["not json",
         '{"action": "click", "args": {"target": 1}, "done": false}',
         '{"action": null, "args": {}, "done": true, "result": "done"}']
    )

    result = await runner.run()

    assert result["status"] == "completed"
    assert acted == [("click", {"target": 1})]


@pytest.mark.asyncio
async def test_fails_after_two_consecutive_bad_json_replies():
    runner, acted = _make_runner(["not json", "still not json"])

    result = await runner.run()

    assert result["status"] == "failed"
    assert "non-JSON" in result["explanation"]
    assert acted == []


@pytest.mark.asyncio
async def test_fails_when_max_turns_exhausted():
    runner, acted = _make_runner(
        ['{"action": "press_key", "args": {"key": "back"}, "done": false}'] * 3,
        max_turns=3,
    )

    result = await runner.run()

    assert result["status"] == "failed"
    assert "Max turns reached" in result["explanation"]
    assert len(acted) == 3
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/unit/agents/local/test_runner.py -v`
Expected: `ModuleNotFoundError: No module named 'artemis.agents.local.runner'`

- [ ] **Step 3: Implement `LocalRunner`**

Create `artemis/agents/local/runner.py`:

```python
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

"""LocalRunner: a Flash-shaped reactive loop backed by the Gallery on-device model.

Same external shape as ``FlashRunner`` (``__init__(ctx, goal, max_turns)``,
``async def run(state)``) so it drops into ``Agent.run_task()``'s per-profile
branch unmodified, but replaces LangChain ``ChatOpenAI`` + tool-calling with a
plain HTTP call to the Gallery's ``/v1/chat/completions`` and a JSON-in-prompt
decision protocol (see ``reply_parser.py``) -- the on-device model is not
expected to support native tool-calling reliably.

``observe_fn``/``call_model_fn``/``act_fn`` default to the real device/HTTP
implementations; tests inject fakes for all three to drive the loop's control
flow without a device or network (see ``tests/unit/agents/local/test_runner.py``).
"""

import base64
from collections.abc import Awaitable, Callable

import httpx

from artemis.agents.local.prompt import TurnRecord, build_system_prompt, build_turn_history_text
from artemis.agents.local.reply_parser import parse_model_reply
from artemis.agents.validator.tool_declarations import capture_screenshot_and_parse_ui
from artemis.config.local_model import LocalModelConfig, parse_local_model_config
from artemis.context import ArtemisContext
from artemis.controllers.unified_controller import UnifiedMobileController
from artemis.mcp.action_executor import McpActionExecutor
from artemis.mcp.action_manifest import OPTIONAL_ACTIONS, REQUIRED_ACTIONS
from artemis.utils.logger import get_logger

logger = get_logger(__name__)

_ALLOWED_ACTIONS: frozenset[str] = REQUIRED_ACTIONS | OPTIONAL_ACTIONS
_MAX_CONSECUTIVE_PARSE_FAILURES = 2

ObserveFn = Callable[[], Awaitable[tuple[bytes | None, str | None]]]
CallModelFn = Callable[[list[dict]], Awaitable[str]]
ActFn = Callable[[str, dict], Awaitable[str]]


class LocalRunner:
    """Reactive agent loop driven by the Gallery on-device model over HTTP."""

    def __init__(
        self,
        ctx: ArtemisContext,
        goal: str,
        max_turns: int | None = None,
        config: LocalModelConfig | None = None,
        observe_fn: ObserveFn | None = None,
        call_model_fn: CallModelFn | None = None,
        act_fn: ActFn | None = None,
    ):
        self.ctx = ctx
        self.goal = goal
        self.config = config or parse_local_model_config()
        self.max_turns = max_turns if max_turns is not None else self.config.max_turns

        self.controller = UnifiedMobileController(ctx)
        self.executor = McpActionExecutor(ctx, self.controller, agent_name="local")

        self._observe_fn = observe_fn or self._default_observe
        self._call_model_fn = call_model_fn or self._default_call_model
        self._act_fn = act_fn or self._default_act

    async def _default_observe(self) -> tuple[bytes | None, str | None]:
        from artemis.graph.state import State

        state = State()
        _, img_bytes, xml_list = await capture_screenshot_and_parse_ui(
            self.ctx, state, self.controller
        )
        return img_bytes, xml_list

    async def _default_call_model(self, messages: list[dict]) -> str:
        headers = {}
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.config.api_base}/chat/completions",
                headers=headers,
                json={"model": self.config.model, "messages": messages, "temperature": 0},
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def _default_act(self, action: str, args: dict) -> str:
        result = await self.executor.execute(action, args, tool_call_id="local", state=None)
        return result.text_summary

    def _build_messages(self, elements_text: str | None, img_bytes: bytes | None, history: list[TurnRecord]) -> list[dict]:
        system_prompt = build_system_prompt(self.goal, _ALLOWED_ACTIONS)
        content: list[dict] = [
            {"type": "text", "text": build_turn_history_text(history)},
        ]
        if img_bytes:
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            )
        if elements_text:
            content.append({"type": "text", "text": elements_text})
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    async def run(self, state=None) -> dict:
        """Runs the reactive loop to completion; returns a Flash-shaped result dict."""
        history: list[TurnRecord] = []
        consecutive_parse_failures = 0
        turn = 0

        while turn < self.max_turns:
            turn += 1
            img_bytes, elements_text = await self._observe_fn()
            messages = self._build_messages(elements_text, img_bytes, history)

            reply_text = await self._call_model_fn(messages)
            parsed = parse_model_reply(reply_text, allowed_actions=_ALLOWED_ACTIONS)

            if parsed is None:
                consecutive_parse_failures += 1
                logger.warning(f"LocalRunner turn {turn}: could not parse model reply")
                if consecutive_parse_failures >= _MAX_CONSECUTIVE_PARSE_FAILURES:
                    return {
                        "status": "failed",
                        "explanation": "Model returned non-JSON output twice in a row.",
                    }
                continue

            consecutive_parse_failures = 0

            if parsed.done:
                if self.ctx.data_engine:
                    self.ctx.data_engine.end_session("completed")
                return {"status": "completed", "result": parsed.result}

            outcome = await self._act_fn(parsed.action, parsed.args)
            if self.ctx.data_engine:
                self.ctx.data_engine.allocate_step_id()
                self.ctx.data_engine.record_step(
                    action_taken={"action": parsed.action, "args": parsed.args},
                    operator_raw_thinking=parsed.thought,
                    last_execution_result={"result": outcome},
                )
            history.append(
                TurnRecord(turn_number=turn, action=parsed.action, args=parsed.args, outcome=outcome)
            )

        if self.ctx.data_engine:
            self.ctx.data_engine.end_session("failed")
        return {"status": "failed", "explanation": "Max turns reached without a final result."}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/agents/local/test_runner.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run the full local-profile unit test suite to check for regressions**

Run: `uv run pytest tests/unit/agents/local/ tests/unit/config/test_local_model.py -v`
Expected: all passed, no regressions.

- [ ] **Step 6: Commit**

```bash
git add artemis/agents/local/runner.py tests/unit/agents/local/test_runner.py
git commit -m "feat(local): add LocalRunner reactive loop for the Gallery on-device model"
```

---

### Task 5: Wire `LocalRunner` into `Agent.run_task()`

**Files:**
- Modify: `artemis/sdk/agent.py:685` (see exact context below)

- [ ] **Step 1: Add the import**

In `artemis/sdk/agent.py`, alongside the existing:

```python
from artemis.agents.flash.runner import FlashRunner
```

add:

```python
from artemis.agents.local.runner import LocalRunner
```

- [ ] **Step 2: Add the `"local"` branch**

In `artemis/sdk/agent.py`, the current code (around line 685) reads:

```python
                    try:
                        if request.profile and request.profile.lower() == "flash":
                            logger.info(f"[{task_name}] Invoking FlashRunner reactive loop...")
                            await task.set_status(
                                status="running",
                                message="Invoking FlashRunner...",
                            )

                            # An explicit max_steps caps the reactive loop; the
                            # default leaves the cap to agent.flash.max_turns
                            # (0 = unlimited).
                            max_turns = (
                                request.max_steps if request.max_steps != RECURSION_LIMIT else None
                            )
                            runner = FlashRunner(context, goal=request.goal, max_turns=max_turns)
                            flash_result = await runner.run(state)

                            output = flash_result
                            last_state_snapshot = state.model_dump()

                            status = flash_result.get("status")
                            if status == "completed":
                                logger.info(f"✅ Automation '{task_name}' is success ✅")
                                await task.finalize(content=output, state=last_state_snapshot)
                                if context.data_engine:
                                    context.data_engine.end_session("completed")
                            else:
                                err = (
                                    f"[{task_name}] FlashRunner failed:"
                                    f" {flash_result.get('explanation')}"
                                )
                                logger.warning(err)
                                await task.finalize(
                                    content=output,
                                    state=last_state_snapshot,
                                    error=err,
                                )
                                if context.data_engine:
                                    context.data_engine.end_session("failed")
                            return output
                        else:
```

Replace the `else:` on the last line with an `elif` for `"local"`, followed by the original `else:` unchanged:

```python
                        elif request.profile and request.profile.lower() == "local":
                            logger.info(f"[{task_name}] Invoking LocalRunner reactive loop...")
                            await task.set_status(
                                status="running",
                                message="Invoking LocalRunner...",
                            )

                            max_turns = (
                                request.max_steps if request.max_steps != RECURSION_LIMIT else None
                            )
                            runner = LocalRunner(context, goal=request.goal, max_turns=max_turns)
                            local_result = await runner.run(state)

                            output = local_result
                            last_state_snapshot = state.model_dump()

                            if local_result.get("status") == "completed":
                                logger.info(f"✅ Automation '{task_name}' is success ✅")
                                await task.finalize(content=output, state=last_state_snapshot)
                            else:
                                err = (
                                    f"[{task_name}] LocalRunner failed:"
                                    f" {local_result.get('explanation')}"
                                )
                                logger.warning(err)
                                await task.finalize(
                                    content=output,
                                    state=last_state_snapshot,
                                    error=err,
                                )
                            return output
                        else:
```

Note: `LocalRunner.run()` already calls `context.data_engine.end_session(...)` itself (Task 4), unlike the `FlashRunner` branch above which calls it here — this mirrors `LocalRunner` owning its own session lifecycle since it does not go through the same `flash_result`/`context.data_engine` coupling FlashRunner's call site assumes.

- [ ] **Step 3: Verify the module still imports and existing tests pass**

Run: `uv run pytest tests/unit/sdk/ -v`
Expected: all passed, no regressions (this is a pure branch addition — no existing behavior changed for `"flash"`/other profiles).

- [ ] **Step 4: Commit**

```bash
git add artemis/sdk/agent.py
git commit -m "feat(local): wire LocalRunner into Agent.run_task as the 'local' profile"
```

---

### Task 6: Accept `model="Local"` in `mobile_run_task`

**Files:**
- Modify: `mcp_server/tools/task_runner.py`

- [ ] **Step 1: Update model validation and the docstring**

In `mcp_server/tools/task_runner.py`, the current validation (inside `mobile_run_task`) reads:

```python
    # 0. Validate and normalize model
    if model.lower() not in ("flash", "pro"):
        raise ValueError(f"Invalid model '{model}'. Must be either 'Flash' or 'Pro'.")
    canonical_model = "Flash" if model.lower() == "flash" else "Pro"
```

Replace with:

```python
    # 0. Validate and normalize model
    if model.lower() not in ("flash", "pro", "local"):
        raise ValueError(f"Invalid model '{model}'. Must be 'Flash', 'Pro', or 'Local'.")
    canonical_model = {"flash": "Flash", "pro": "Pro", "local": "Local"}[model.lower()]
```

In the same function's docstring, the `### Model selection (routing)` section currently lists Flash and Pro. Add a third bullet immediately after the Pro bullet:

```
    - **Local**: same reactive loop shape as Flash, but backed by a free
      on-device model served by the Google AI Edge Gallery app's local API
      server instead of a paid LLM. No Artemis LLM credential required.
      Requires the Gallery app's "Expose local API server" setting to be on
      and reachable at the address configured in `artemis.jsonc`'s
      `local_model` block. Same limitations as Flash (no ADB shell, no
      persistent plan/notes, no verification/report) plus a smaller model
      context window — best for simple, short tasks.
```

Every other reference to `canonical_model != "Flash"` in this file (the
`notes_dir` handling for the standalone-spawn response, and the
`expected_output_desc` gating) should also exclude `"Local"`, since Local has
no notes/report support either. Find:

```python
        if expected_output_desc and canonical_model != "Flash":
            cmd.extend(["--expected-output-desc", expected_output_desc])
```

Replace with:

```python
        if expected_output_desc and canonical_model not in ("Flash", "Local"):
            cmd.extend(["--expected-output-desc", expected_output_desc])
```

Find:

```python
        if canonical_model != "Flash":
            response_dict["notes_dir"] = os.path.join(trace_dir, "notes")
```

Replace with:

```python
        if canonical_model not in ("Flash", "Local"):
            response_dict["notes_dir"] = os.path.join(trace_dir, "notes")
```

Find (in the daemon-dispatch branch):

```python
                    expected_output=expected_output_desc if canonical_model != "Flash" else None,
```

Replace with:

```python
                    expected_output=(
                        expected_output_desc if canonical_model not in ("Flash", "Local") else None
                    ),
```

- [ ] **Step 2: Run the existing MCP tool test suite**

Run: `uv run pytest tests/unit/mcp/test_mcp_tools.py -v`
Expected: all passed. If any test asserts the exact invalid-model error message
(`"Must be either 'Flash' or 'Pro'"`), update it to match the new message
(`"Must be 'Flash', 'Pro', or 'Local'."`) in that test file.

- [ ] **Step 3: Commit**

```bash
git add mcp_server/tools/task_runner.py
git commit -m "feat(local): accept model=Local in mobile_run_task"
```

---

### Task 7: Pass the `local` profile through the standalone background runner

**Files:**
- Modify: `mcp_server/background/task_runner.py`

- [ ] **Step 1: Update the profile selection**

In `mcp_server/background/task_runner.py`'s `run_task`, the current code reads:

```python
        if model.lower() == "flash":
            task_builder.using_profile("flash")
```

Replace with:

```python
        if model.lower() in ("flash", "local"):
            task_builder.using_profile(model.lower())
```

A few lines below, the result-formatting fallback reads:

```python
        if not result:
            if model.lower() == "flash":
                result = "Task executed successfully."
            else:
                result = (
                    "Task executed successfully. You can check the notes under "
                    "notes_dir to view more details."
                )
```

Replace with:

```python
        if not result:
            if model.lower() in ("flash", "local"):
                result = "Task executed successfully."
            else:
                result = (
                    "Task executed successfully. You can check the notes under "
                    "notes_dir to view more details."
                )
```

Also update the `--model` argparse help text at the bottom of the file:

```python
    parser.add_argument("--model", required=True, help="Model to use ('Flash' or 'Pro')")
```

Replace with:

```python
    parser.add_argument("--model", required=True, help="Model to use ('Flash', 'Pro', or 'Local')")
```

- [ ] **Step 2: Verify the module still imports cleanly**

Run: `uv run python -c "import mcp_server.background.task_runner"`
Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add mcp_server/background/task_runner.py
git commit -m "feat(local): pass the local profile through the standalone background runner"
```

---

### Task 8: Manual on-device verification

**Files:** none (verification only)

Prerequisite: the Gallery app's local API server (from
`docs/superpowers/specs/2026-09-17-local-api-server-design.md` and
`...-api-server-mode-design.md` in the `google-ai-edge-gallery` fork) must be
running and reachable, with a model selected and loaded.

- [ ] **Step 1: Point `artemis.jsonc` at the Gallery server**

Add (or confirm) a `local_model` block at the top level of `config/artemis.jsonc`:

```jsonc
"local_model": {
  "api_base": "http://127.0.0.1:8080/v1",
  "token": "<the token shown in the Gallery app's Settings>",
  "model": "<the model name shown in Gallery Settings, e.g. Gemma-4-E2B-it>",
  "max_turns": 40
}
```

(Use `adb forward tcp:8080 tcp:8080` if the phone is connected over USB and
the server binds to the device's own loopback; use the phone's LAN IP in
`api_base` instead if running over Wi-Fi.)

- [ ] **Step 2: Confirm the server responds directly**

Run: `curl -s http://127.0.0.1:8080/v1/models`
Expected: `{"data":[{"id":"<model name>"}]}`.

- [ ] **Step 3: Run a simple task via `mobile_run_task`**

Call the `mobile_run_task` MCP tool with:
- `task_desc`: `"Open Settings and toggle Wi-Fi off, then back on."`
- `model`: `"Local"`

Record the returned `trace_id`.

- [ ] **Step 4: Poll status until completion**

Call `mobile_manage_task(action="status", trace_id="<trace_id>")` every ~10
seconds. Expected: `status` moves from `"running"` to `"completed"` (or
`"failed"` with a legible `error`/`explanation` if the model's actions didn't
land — capture that message for follow-up if so).

- [ ] **Step 5: Inspect the trace**

Call `mobile_inspect_trace(action="view_summary", trace_id="<trace_id>")`.
Expected: a normal step-by-step trace showing each action taken, the same
shape Flash traces already produce.

- [ ] **Step 6: Record the outcome**

Add a "Verified on-device" section to
`docs/superpowers/specs/2026-09-17-local-runner-design.md` describing what
happened (success, or the specific failure and root cause if the on-device
model's output didn't parse reliably — this is the biggest real unknown in
this design, since it depends on how well a small on-device model follows
the JSON-only instruction).

```bash
git add docs/superpowers/specs/2026-09-17-local-runner-design.md
git commit -m "docs: record manual verification of the Local profile"
```

---

## Self-review notes

- **Spec coverage:** "Third profile, not a new tool" decision → Task 5, 6, 7.
  `LocalRunner` architecture (reused executor/DataEngine, custom HTTP+JSON
  loop, no ledger/chunking) → Tasks 1–4. Config block → Task 1 (and Task 8
  Step 1 for the on-device value). Error-handling table → Task 4's runner
  tests (`test_fails_after_two_consecutive_bad_json_replies`,
  `test_fails_when_max_turns_exhausted`) plus Task 8 Step 4 for the
  unreachable-server case (surfaced through `httpx`'s own exception,
  propagating out of `run_task`'s existing `except Exception` handler in
  `mcp_server/background/task_runner.py`, same as any other runner crash —
  no special-casing needed there). Testing section → Tasks 1–4's unit tests
  plus Task 8's manual script.
- **Type consistency:** `LocalRunner.__init__`'s `config`/`observe_fn`/
  `call_model_fn`/`act_fn` parameters (Task 4) match what Task 4's own tests
  construct. `ParsedAction`'s fields (`action`, `args`, `thought`, `done`,
  `result`) from Task 2 are exactly what `LocalRunner.run()` reads in Task 4.
  `TurnRecord`'s fields (Task 3) match what `LocalRunner.run()` constructs
  and appends to `history` in Task 4.
- **Known gap intentionally deferred:** the exact reliability of a small
  on-device model at following the "respond with ONLY JSON" instruction is
  unverified until Task 8 — this is called out explicitly there rather than
  assumed. If Task 8 finds the model frequently wraps JSON in prose the
  regex-based extraction in `reply_parser.py` can't handle, that is a
  follow-up fix to `_extract_json_text`, not a redesign.
