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
    # api_base uses escaped slashes (valid JSON `\/`): artemis.jsonc's comment
    # stripper (artemis/utils/file.py) naively regexes out `//...` to end of
    # line, which corrupts an unescaped URL. This is a known, separately
    # tracked bug (out of scope here) -- escaping is the established
    # workaround already used elsewhere in this repo's own artemis.jsonc.
    config_path = tmp_path / "artemis.jsonc"
    config_path.write_text(
        '{"local_model": {'
        '"api_base": "http:\\/\\/192.168.1.50:8080\\/v1", '
        '"token": "secret-token", '
        '"model": "Gemma-4-E2B-it", '
        '"max_turns": 25'
        "}}",
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
