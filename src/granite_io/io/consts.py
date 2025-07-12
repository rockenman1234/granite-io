# SPDX-License-Identifier: Apache-2.0

"""
Common shared constants
"""

# Granite 3.2 constants
_GRANITE_3_2_COT_START = "Here is my thought process:"
_GRANITE_3_2_COT_END = "Here is my response:"

_GRANITE_3_2_2B_OLLAMA = "granite3.2:2b"
_GRANITE_3_2_2B_HF = "ibm-granite/granite-3.2-2b-instruct"
_GRANITE_3_2_MODEL_NAME = "Granite 3.2"

_GRANITE_3_2_SPECIAL_TOKENS = [
    "<|end_of_text|>",
    "<|start_of_role|>",
    "<|end_of_role|>",
    "<|tool_call|>",
]

# Granite 3.3 constants
_GRANITE_3_3_MODEL_NAME = "Granite 3.3"
_GRANITE_3_3_2B_OLLAMA = "granite3.3:2b"
_GRANITE_3_3_8B_OLLAMA = "granite3.3:8b"
_GRANITE_3_3_2B_HF = "ibm-granite/granite-3.3-2b-instruct"
_GRANITE_3_3_8B_HF = "ibm-granite/granite-3.3-8b-instruct"

_GRANITE_3_3_CITATIONS_START = '{"id": "citation"}'
_GRANITE_3_3_HALLUCINATIONS_START = '{"id": "hallucination"}'
_GRANITE_3_3_CITE_START = "<|start_of_cite|>"
_GRANITE_3_3_CITE_END = "<|end_of_cite|>"
_GRANITE_3_3_COT_START = "<think>"
_GRANITE_3_3_COT_END = "</think>"
_GRANITE_3_3_RESP_START = "<response>"
_GRANITE_3_3_RESP_END = "</response>"

_GRANITE_3_3_SPECIAL_TOKENS = [
    "<|end_of_text|>",
    "<|start_of_role|>",
    "<|end_of_role|>",
    "<|tool_call|>",
    "<|start_of_cite|>",
    "<|end_of_cite|>",
    "<|start_of_plugin|>",
    "<|end_of_plugin|>",
]
