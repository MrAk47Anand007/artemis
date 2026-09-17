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
