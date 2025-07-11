# SPDX-License-Identifier: Apache-2.0

# Standard
import os

# Third Party
from test_utils import fix_granite_date, load_text_file

# Local
from granite_io import get_input_processor
from granite_io.types import (
    ChatCompletionInputs,
    UserMessage,
)

_GENERALE_MODEL_NAME = "Granite 3.2"
_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")


def test_run_processor_reasoning():
    input_processor = get_input_processor(_GENERALE_MODEL_NAME)
    question = (
        "Find the fastest way for a seller to visit all the cities in their region"
    )
    messages = [UserMessage(content=question)]
    prompt = input_processor.transform(
        ChatCompletionInputs(messages=messages, thinking=True)
    )

    expected_prompt = load_text_file(
        os.path.join(_TEST_DATA_DIR, "test_reasoning_prompt.txt")
    )
    assert isinstance(prompt, str)
    assert prompt == fix_granite_date(expected_prompt)


def test_remove_special_tokens():
    input_processor = get_input_processor(_GENERALE_MODEL_NAME)

    input_json_str = load_text_file(
        os.path.join(_TEST_DATA_DIR, "test_remove_special_tokens_input_json.txt")
    )

    inputs = ChatCompletionInputs.model_validate_json(input_json_str)
    prompt = input_processor.transform(inputs)

    expected_prompt = load_text_file(
        os.path.join(_TEST_DATA_DIR, "test_remove_special_tokens_expected_prompt.txt")
    )
    assert isinstance(prompt, str)
    assert prompt == fix_granite_date(expected_prompt)
