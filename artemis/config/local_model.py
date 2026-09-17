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
    """Connection details for the Gallery local API server.

    Note: ``api_base`` in ``artemis.jsonc`` must escape its slashes
    (``http:\\/\\/...``) -- the file's comment stripper naively regexes out
    ``//...`` to end of line, which corrupts an unescaped URL. Pre-existing,
    separately tracked limitation of ``artemis.utils.file.strip_json_comments``.
    """

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
