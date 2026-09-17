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
