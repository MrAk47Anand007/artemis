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

from unittest.mock import Mock, patch

import pytest

from artemis.agents.local.runner import LocalRunner
from artemis.config.local_model import LocalModelConfig
from artemis.context import ArtemisContext


def _fake_context():
    ctx = Mock(spec=ArtemisContext)
    ctx.device = Mock()
    ctx.data_engine = None
    ctx.adb_client = None
    ctx.driver = Mock()
    return ctx


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

    # Real device-driver construction is skipped: every device/network touch
    # point (observe/call-model/act) is injected as a fake, so no real driver
    # is ever needed -- same pattern as tests/unit/agents/flash/test_turn_index_snapshot.py.
    with patch("artemis.controllers.unified_controller.get_driver"):
        runner = LocalRunner(
            ctx=_fake_context(),
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
async def test_retry_after_parse_failure_uses_nonzero_temperature():
    # At temperature 0, a malformed reply against an unchanged prompt
    # repeats byte-for-byte on retry (observed on-device); bumping the
    # temperature on retry gives the decoder a chance to actually diverge.
    replies = iter(
        ["not json", '{"action": null, "args": {}, "done": true, "result": "ok"}']
    )
    temperatures_seen: list[float] = []
    runner_box: list[LocalRunner] = []

    async def fake_observe():
        return b"fake-screenshot-bytes", "1: [Button] Wi-Fi toggle"

    async def fake_call_model(messages):
        temperatures_seen.append(runner_box[0]._temperature)
        return next(replies)

    async def fake_act(action, args):
        return "success"

    with patch("artemis.controllers.unified_controller.get_driver"):
        runner = LocalRunner(
            ctx=_fake_context(),
            goal="Toggle Wi-Fi",
            config=LocalModelConfig(),
            observe_fn=fake_observe,
            call_model_fn=fake_call_model,
            act_fn=fake_act,
        )
    runner_box.append(runner)

    result = await runner.run()

    assert result["status"] == "completed"
    assert temperatures_seen == [0.0, 0.4]


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
