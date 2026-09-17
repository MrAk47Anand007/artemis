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
        # Lazily created on first _default_observe call and reused for the
        # whole run: McpActionExecutor.execute() resolves index-based click
        # targets against state.indexed_elements, so observe and act must
        # share the same State instance across turns, not a fresh one each
        # time. Tests inject observe_fn/act_fn directly and never touch this.
        self._state = None
        # Bumped to a nonzero value by run() after a parse failure, then
        # reset to 0 on the next successful parse. At temperature 0 a
        # malformed reply repeats byte-for-byte on retry against an
        # unchanged prompt (observed on-device: a stuck screen + a
        # deterministic JSON glitch is a guaranteed two-in-a-row failure);
        # a nonzero retry gives the decoder a chance to actually produce
        # something different.
        self._temperature = 0.0

    async def _default_observe(self) -> tuple[bytes | None, str | None]:
        from artemis.graph.state import State

        if self._state is None:
            self._state = State.initial(self.goal)
        _, img_bytes, xml_list = await capture_screenshot_and_parse_ui(
            self.ctx, self._state, self.controller
        )
        return img_bytes, xml_list

    async def _default_call_model(self, messages: list[dict]) -> str:
        headers = {}
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"
        # On-device inference latency is much more variable than a cloud LLM's
        # (observed 60s+ on a mid-range phone under load during manual
        # verification) -- a generous timeout here, not a network one.
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                f"{self.config.api_base}/chat/completions",
                headers=headers,
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self._temperature,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def _default_act(self, action: str, args: dict) -> str:
        result = await self.executor.execute(
            action, args, tool_call_id="local", state=self._state
        )
        return result.text_summary

    def _build_messages(
        self, elements_text: str | None, img_bytes: bytes | None, history: list[TurnRecord]
    ) -> list[dict]:
        # Folded into the user message rather than sent as a separate
        # role="system" message: manual on-device verification found the
        # Gallery server's local API doesn't surface a system message's
        # content to the model as an instruction (the model treated it as
        # something to acknowledge, not follow -- confirmed by direct
        # curl tests against /v1/chat/completions). A workaround on this
        # side, not a fix to the Gallery server itself.
        system_prompt = build_system_prompt(self.goal, _ALLOWED_ACTIONS)
        content: list[dict] = [
            {"type": "text", "text": system_prompt},
            {"type": "text", "text": build_turn_history_text(history)},
        ]
        if img_bytes:
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            )
        if elements_text:
            content.append({"type": "text", "text": elements_text})
        return [{"role": "user", "content": content}]

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
                # print(), not just logger.warning(): the background task
                # runner redirects sys.stdout/sys.stderr to per-trace log
                # files after loguru's sinks are already bound, so warnings
                # logged through loguru alone can silently miss those files.
                message = (
                    f"LocalRunner turn {turn}: could not parse model reply. "
                    f"Raw reply (first 500 chars): {reply_text[:500]!r}"
                )
                logger.warning(message)
                print(message)
                if consecutive_parse_failures >= _MAX_CONSECUTIVE_PARSE_FAILURES:
                    return {
                        "status": "failed",
                        "explanation": "Model returned non-JSON output twice in a row.",
                    }
                self._temperature = 0.4
                continue

            consecutive_parse_failures = 0
            self._temperature = 0.0

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
