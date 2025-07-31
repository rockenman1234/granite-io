# SPDX-License-Identifier: Apache-2.0

# Standard
import datetime
import json
import re

# Third Party
from pydantic_core import PydanticCustomError
from typing_extensions import Self
import pydantic

# Local
from granite_io.io.base import InputProcessor
from granite_io.io.consts import (
    _GRANITE_3_2_2B_HF,
    _GRANITE_3_2_2B_OLLAMA,
    _GRANITE_3_2_COT_END,
    _GRANITE_3_2_COT_START,
    _GRANITE_3_2_MODEL_NAME,
    _GRANITE_3_2_SPECIAL_TOKENS,
)
from granite_io.io.registry import input_processor
from granite_io.types import (
    AssistantMessage,
    ChatCompletionInputs,
    Document,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)

_TODAYS_DATE_STR = datetime.datetime.now().strftime("%B %d, %Y")


def override_date_for_testing(todays_date_str: str | None):
    """Override the date that methods in this file will use for today's date, in order
    to make test outputs consistent.

    :param todays_date_str: Date string to use for generating prompts until further
     notice, or ``None`` to revert to using the real date.
    """
    global _TODAYS_DATE_STR  # pylint: disable=global-statement
    if todays_date_str is None:
        _TODAYS_DATE_STR = datetime.datetime.now().strftime("%B %d, %Y")
    else:
        _TODAYS_DATE_STR = todays_date_str


# String that comes at the beginning of the system message that a Granite 3.2 model must
# receive at the beginning of the prompt for any completion request that does not
# provide a custom system message.
#
# Note that the original Jinja template tends to choose weird dates from the future for
# the "Today's date" part. Instead of replicating that behavior, we put today's actual
# date in that section of the prompt. This difference probably doesn't matter, since
# none of the supervised fine tuning data exercises knowledge cutoffs.
#
# As an additional wrinkle, we need to use a consistent date when testing, so we use a
# function to recreate this string every time we need it.
def _make_system_message_start():
    return f"""\
Knowledge Cutoff Date: April 2024.
Today's Date: {_TODAYS_DATE_STR}.
You are Granite, developed by IBM."""


# String that a Granite 3.2 model must receive immediately after _SYSTEM_MESSAGE_START
# if there are both tools and RAG documents in the current request.
_TOOLS_AND_DOCS_SYSTEM_MESSAGE_PART = """\
 You are a helpful AI assistant with access to the following tools. When a tool is \
required to answer the user's query, respond with <|tool_call|> followed by a JSON \
list of tools used. If a tool does not exist in the provided list of tools, notify the \
user that you do not have the ability to fulfill the request.

Write the response to the user's input by strictly aligning with the facts in the \
provided documents. If the information needed to answer the question is not available \
in the documents, inform the user that the question cannot be answered based on the \
available data."""

# String that a Granite 3.2 model must receive immediately after _SYSTEM_MESSAGE_START
# if there are documents in the current request but there are no tools in the current
# request.
_NO_TOOLS_AND_DOCS_SYSTEM_MESSAGE_PART = """\
Write the response to the user's input by strictly aligning with the facts in the \
provided documents. If the information needed to answer the question is not available \
in the documents, inform the user that the question cannot be answered based on the \
available data."""

# String that a Granite 3.2 model must receive immediately after _SYSTEM_MESSAGE_START
# if there are tools in the current request but there are no documents in the current
# request.
_TOOLS_AND_NO_DOCS_SYSTEM_MESSAGE_PART = """\
 You are a helpful AI assistant with access to the following tools. When a tool is \
required to answer the user's query, respond with <|tool_call|> followed by a JSON \
list of tools used. If a tool does not exist in the provided list of tools, notify the \
user that you do not have the ability to fulfill the request."""

# String that a Granite 3.2 model must receive immediately after _SYSTEM_MESSAGE_START
# if there are no tools or documents in the current request and the "thinking" flag is
# set to `True`.
_NO_TOOLS_AND_NO_DOCS_AND_THINKING_SYSTEM_MESSAGE_PART = f"""\
 You are a helpful AI assistant.
Respond to every user query in a comprehensive and detailed way. You can write down \
your thoughts and reasoning process before responding. In the thought process, engage \
in a comprehensive cycle of analysis, summarization, exploration, reassessment, \
reflection, backtracing, and iteration to develop well-considered thinking process. \
In the response section, based on various attempts, explorations, and reflections from \
the thoughts section, systematically present the final solution that you deem correct. \
The response should summarize the thought process. Write your thoughts after '\
{_GRANITE_3_2_COT_START}' and write your response after '{_GRANITE_3_2_COT_END}' \
for each user query."""

# String that a Granite 3.2 model must receive immediately after _SYSTEM_MESSAGE_START
# if there are no tools or documents in the current request and the "thinking" flag is
# set to `False`.
_NO_TOOLS_NO_DOCS_NO_THINKING_SYSTEM_MESSAGE_PART = """\
 You are a helpful AI assistant."""


# String that a Granite 3.2 model must receive immediately after either
# _TOOLS_AND_DOCS_SYSTEM_MESSAGE_MIDDLE  (if there are tools) or
# _NO_TOOLS_AND_DOCS_SYSTEM_MESSAGE_MIDDLE (if there are no tools) in the system prompt
# if the "citations" flag is `True` and there are documents.
_DOCS_AND_CITATIONS_SYSTEM_MESSAGE_PART = """\


In your response, use the symbols <co> and </co> to indicate when a fact comes from a \
document in the search result, e.g <co>0</co> for a fact from document 0. Afterwards, \
list all the citations with their corresponding documents in an ordered list."""

# String that a Granite 3.2 model must receive immediately after either
# _TOOLS_AND_DOCS_SYSTEM_MESSAGE_MIDDLE (if there are tools and no citations) or
# _NO_TOOLS_AND_DOCS_SYSTEM_MESSAGE_MIDDLE (if there are no tools or citations) or
# _DOCS_AND_CITATIONS_SYSTEM_MESSAGE_PART in the system prompt
# if the "hallucinations" flag is `True` and there are documents.
# Note that a list of zero documents counts as "having documents".
_DOCS_AND_HALLUCINATIONS_SYSTEM_MESSAGE_PART = """\


Finally, after the response is written, include a numbered list of sentences from the \
response that are potentially hallucinated and not based in the documents."""


class ControlsRecord(pydantic.BaseModel):
    citations: bool | None = None
    hallucinations: bool | None = None
    length: str | None = None  # Length output control variable
    originality: str | None = None

    @pydantic.field_validator("length", mode="after")
    @classmethod
    def _validate_length(cls, value: str | None) -> str | None:
        if value is None or value == "short" or value == "long":
            return value
        raise PydanticCustomError(
            "length field validator",
            'length ({length}) must be "short" or "long" or None',
            {"length": value},
        )

    @pydantic.field_validator("originality", mode="after")
    @classmethod
    def _validate_originality(cls, value: str | None) -> str | None:
        if value is None or value == "extractive" or value == "abstractive":
            return value
        raise PydanticCustomError(
            "originality field validator",
            'originality ({originality}) must be "extractive" or "abstractive" or None',
            {"originality": value},
        )


class Granite3Point2Inputs(ChatCompletionInputs):
    """
    Class that represents the inputs to a Granite 3.2 model generation call.

    Contains fields for all input functionality supported by the current version of
    Granite.

    This class will gain additional fields as new functionality is added to Granite.
    """

    documents: list[Document] = []
    controls: ControlsRecord | None = None

    thinking: bool = False
    sanitize: str | None = None

    @pydantic.field_validator("messages")
    @classmethod
    def _validate_inputs_messages(cls, messages: list) -> list:
        # Make a copy so the validation code below can mutate the messages list but pass
        # through the original value. The caller also might have a pointer to the list.
        original_messages = messages
        messages = messages.copy()

        # There is no supervised fine tuning data for the case of zero messages.
        # Models are not guaranteed to produce a valid response if there are zero
        # messages.
        if len(messages) == 0:
            raise ValueError(
                "No messages. Model behavior for this case is not defined."
            )

        # The first message, and only the first message, may be the system message.
        first_message_is_system_message = isinstance(messages[0], SystemMessage)
        if first_message_is_system_message:
            messages = messages[1:]
            # If there is a system message, there must be at least one more user or
            # assistant message.
            if len(messages) == 0:
                raise ValueError(
                    "Input contains only a system message. Model behavior for this "
                    "case is not defined."
                )

        # The first message that is not a system message must be
        # either a user or assistant message.
        if not isinstance(messages[0], UserMessage | AssistantMessage):
            if first_message_is_system_message:
                raise ValueError(
                    f"First message after system message must be a user or "
                    f"assistant message. Found type {type(messages[0])}"
                )
            raise ValueError(
                f"First message must be a system, user, or assistant "
                f"Found type {type(messages[0])}"
            )

        # Undocumented constraint: All other messages form a conversation that
        # alternates strictly between user and assistant, possibly with tool calls
        # after an assistant turn and before the next user turn.
        # TODO: Validate this invariant.

        # Pydantic will use the value that this validator returns as the value of the
        # messages field. Undo any changes that we made during validation and return
        # the original value.
        return original_messages

    @pydantic.field_validator("documents")
    @classmethod
    def _validate_documents(cls, documents: list[Document] | None) -> list | None:
        """
        Granite 3.2 documents should not have document IDs.
        """
        if documents is not None:
            for i, d in enumerate(documents):
                if not isinstance(d, Document):
                    raise TypeError(
                        f"Expected Document at position {i} but found "
                        f"{d} of type {type(d)}"
                    )
                if d.doc_id is not None:
                    raise ValueError(
                        f"Document at position {i} contains a `doc_id` "
                        f"field. This field is not allowed for Granite "
                        f"3.2."
                    )
        return documents

    @pydantic.field_validator("sanitize", mode="after")
    @classmethod
    def _validate_sanitize(cls, value: str | None) -> str | None:
        if value is None or value == "inputs" or value == "aggressive":
            return value
        raise PydanticCustomError(
            "sanitize field validator",
            'sanitize ({sanitize}) must be "inputs" or "aggressive" or None',
            {"sanitize": value},
        )

    def _remove_special_tokens(self, text) -> str:
        """
        Removes any special tokens from the text string.

        :param text: String for removal of special tokens.
        :returns: String with any special tokens removed.
        """

        regex_roles = r"<\|start_of_role\|>.*<\|end_of_role\|>.*<\|end_of_text\|>"
        regex_tool_call = r"<\|tool_call\|>\{.*\}"

        new_text = text
        new_text = re.sub(regex_roles, "", new_text)
        new_text = re.sub(regex_tool_call, "", new_text)

        # Replace any stray special tokens.
        for special_token in _GRANITE_3_2_SPECIAL_TOKENS:
            new_text = new_text.replace(special_token, "")
        return new_text

    @pydantic.model_validator(mode="after")
    def _sanitize_validator(self) -> Self:
        """
        Removes the special tokens from the inputs.
        - 'None': (default) keeps input as is
        - 'inputs': only the messages
        - 'aggressive': everything that's part of a chat completion request,
            e.g. documents, messages, tools
        """
        if self.sanitize and self.sanitize == "inputs":
            for message in self.messages:
                message.content = self._remove_special_tokens(message.content)

        if self.sanitize and self.sanitize == "aggressive":
            for message in self.messages:
                message.content = self._remove_special_tokens(message.content)
            for document in self.documents:
                document.text = self._remove_special_tokens(document.text)
            for tool in self.tools:
                tool.name = self._remove_special_tokens(tool.name)
                if tool.description:
                    tool.description = self._remove_special_tokens(tool.description)
                if tool.parameters:
                    new_params = {}
                    for k, v in tool.parameters.items():
                        kk = self._remove_special_tokens(k)
                        vv = self._remove_special_tokens(v)
                        if len(kk) > 0:
                            new_params[kk] = vv
                    tool.parameters = new_params
        return self


@input_processor(
    _GRANITE_3_2_MODEL_NAME,
    # Huggingface
    _GRANITE_3_2_2B_HF,
    "ibm-granite/granite-3.2-8b-instruct",
    # Ollama
    "granite3.2",
    "granite3.2:8b",
    _GRANITE_3_2_2B_OLLAMA,
    # RITS
    "ibm-granite/granite-8b-instruct-preview-4k",
)
class Granite3Point2InputProcessor(InputProcessor):
    """
    Input processor for version 3.2 of the main Granite models, all sizes.

    This input processor is based on the Jinja template that was used during
    supervised fine tuning of these models. This template is as follows:
    ```
    {%- if messages[0]['role'] == 'system' %}
        {%- set system_message = messages[0]['content'] %}
        {%- set loop_messages = messages[1:] %}
    {%- else %}
        {%- set system_message = \"Knowledge Cutoff Date: April 2024.\nToday's Date: \"
          + strftime_now('%B %d, %Y') + \".\nYou are Granite, developed by IBM.\" %}
        {%- if tools and documents %}
                {%- set system_message = system_message + \" You are a helpful AI
                assistant with access to the following tools.
                  When a tool is required to answer the user's query, respond with
                  <|tool_call|> followed by a JSON list of tools used. If a tool does
                  not exist in the provided list of tools, notify the user that you do
                  not have the ability to fulfill the request.\n\nWrite the response to
                  the user's input by strictly aligning with the facts in the provided
                  documents. If the information needed to answer the question is not
                  available in the documents, inform the user that the question cannot
                  be answered based on the available data.\" %}
        {%- elif tools %}
                {%- set system_message = system_message + \" You are a helpful AI
                assistant with access to the following tools. When a tool is required to
                answer the user's query, respond with <|tool_call|> followed by a JSON
                list of tools used. If a tool does not exist in the provided list of
                tools, notify the user that you do not have the ability to fulfill the
                request.\" %}
        {%- elif documents %}
                {%- set system_message = system_message + \" Write the response to the
                user's input by strictly aligning with the facts in the provided
                documents. If the information needed to answer the question is not
                available in the documents, inform the user that the question cannot be
                answered based on the available data.\" %}
        {%- elif thinking %}
                {%- set system_message = system_message + \" You are a helpful AI
                assistant.\nRespond to every user query in a comprehensive and detailed
                way. You can write down your thoughts and reasoning process before
                responding. In the thought process, engage in a comprehensive cycle of
                analysis, summarization, exploration, reassessment, reflection,
                backtracing, and iteration to develop well-considered thinking process.
                In the response section, based on various attempts, explorations, and
                reflections from the thoughts section, systematically present the final
                solution that you deem correct. The response should summarize the
                thought process. Write your thoughts after 'Here is my thought process:'
                and write your response after 'Here is my response:' for each user
                query.\" %}
        {%- else %}
                {%- set system_message = system_message + \" You are a helpful AI
                assistant.\" %}
        {%- endif %}
        {%- if 'citations' in controls and documents %}
            {%- set system_message = system_message + '\n\nIn your response, use the
            symbols <co> and </co> to indicate when a fact comes from a document in the
            search result, e.g <co>0</co> for a fact from document 0. Afterwards, list
            all the citations with their corresponding documents in an ordered list.' %}
        {%- endif %}
        {%- if 'hallucinations' in controls and documents %}
            {%- set system_message = system_message + '\n\nFinally, after the response
            is written, include a numbered list of sentences from the response that are
            potentially hallucinated and not based in the documents.' %}
        {%- endif %}
        {%- set loop_messages = messages %}
    {%- endif %}
    {{- '<|start_of_role|>system<|end_of_role|>' + system_message +
        '<|end_of_text|>\n' }}
    {%- if tools %}
        {{- '<|start_of_role|>tools<|end_of_role|>' }}
        {{- tools | tojson(indent=4) }}
        {{- '<|end_of_text|>\n' }}
    {%- endif %}
    {%- if documents %}
        {{- '<|start_of_role|>documents<|end_of_role|>' }}
        {%- for document in documents %}
            {{- 'Document ' + loop.index0 | string + '\n' }}
            {{- document['text'] }}
            {%- if not loop.last %}
                {{- '\n\n'}}
            {%- endif%}
        {%- endfor %}
        {{- '<|end_of_text|>\n' }}
    {%- endif %}
    {%- for message in loop_messages %}
        {{- '<|start_of_role|>' + message['role'] + '<|end_of_role|>' +
        message['content'] + '<|end_of_text|>\n' }}
        {%- if loop.last and add_generation_prompt %}
            {{- '<|start_of_role|>assistant' }}
            {%- if controls %}
                {{- ' ' + controls | tojson()}}
            {%- endif %}
            {{- '<|end_of_role|>' }}
        {%- endif %}
    {%- endfor %}
    ```
    """

    def _split_messages(
        self, inputs: Granite3Point2Inputs
    ) -> tuple[SystemMessage | None, list[UserMessage]]:
        """
        Separate the system message from other messages.

        :returns: Tuple of system message, if present, and remaining messages.
        """
        messages = inputs.messages

        # Validation code in the Inputs class should already have verified that there
        # are either zero or one system messages, and that the system message, if
        # present, occurs at position zero.
        if isinstance(messages[0], SystemMessage):
            # First message is a system message.
            return messages[0], messages[1:]
        return None, messages

    def _build_default_system_message(self, inputs: Granite3Point2Inputs) -> str:
        """
        :param inputs: All inputs to a completion request that does not include a custom
            system message.
        :returns: The standard system message portion of the prompt for the request,
            as a string suitable to feed to the model's tokenizer.
        """
        # Compute the predicates that determine exactly what default system message to
        # use.
        have_documents = inputs.documents is not None and len(inputs.documents) > 0
        have_tools = len(inputs.tools) > 0

        # Carefully hew to the policy that the original Jinja template's behavior
        # defines.
        # First, disallow the cases that the authors of the Jinja template did not
        # provide any code to handle.
        if inputs.thinking and have_documents:
            raise ValueError(
                f"'thinking' flag is set, but documents were provided. "
                f"{_GRANITE_3_2_MODEL_NAME} only supports the 'thinking' flag when "
                f"documents are not provided."
            )
        if inputs.thinking and have_tools:
            raise ValueError(
                f"'thinking' flag is set, but tools were provided. "
                f"{_GRANITE_3_2_MODEL_NAME} only supports the 'thinking' flag when "
                f"tools are not provided."
            )

        # The default system message starts with a header that includes the date and
        # knowledge cutoff.
        system_message = "<|start_of_role|>system<|end_of_role|>"
        system_message += _make_system_message_start()

        # Add a middle part that varies depending on tools, documents, and citations.
        if have_documents and have_tools:
            system_message += _TOOLS_AND_DOCS_SYSTEM_MESSAGE_PART
        elif have_documents:  # and not have_tools
            system_message += _NO_TOOLS_AND_DOCS_SYSTEM_MESSAGE_PART
        elif have_tools:  # and not have_documents
            system_message += _TOOLS_AND_NO_DOCS_SYSTEM_MESSAGE_PART
        elif inputs.thinking:  # if not have_documents and not have_tools
            system_message += _NO_TOOLS_AND_NO_DOCS_AND_THINKING_SYSTEM_MESSAGE_PART
        else:  # if not inputs.thinking and not have_documents and not have_tools
            system_message += _NO_TOOLS_NO_DOCS_NO_THINKING_SYSTEM_MESSAGE_PART

        # Next comes an optional section of instructions for citations.
        if inputs.controls and inputs.controls.citations:
            if not have_documents:
                # TODO: The template skips the citations instruction in this case.
                # Is this behavior an error? Should we raise an error if the caller
                # sets the citations flag but provides zero documents?
                pass
            else:  # if have_documents
                system_message += _DOCS_AND_CITATIONS_SYSTEM_MESSAGE_PART

        # Then comes an optional section of instructions for hallucinations.
        if inputs.controls and inputs.controls.hallucinations:
            if not have_documents:
                raise ValueError(
                    f"'hallucinations' flag is set, but the model input does not "
                    f"include documents. {_GRANITE_3_2_MODEL_NAME} only supports the "
                    f"'hallucinations' flag when documents are provided."
                )
            # if have_documents
            system_message += _DOCS_AND_HALLUCINATIONS_SYSTEM_MESSAGE_PART

        # Finish with an end of text
        system_message += "<|end_of_text|>\n"

        return system_message

    def _message_to_prompt_string(self, message: UserMessage | AssistantMessage) -> str:
        if isinstance(message, UserMessage):
            return (
                f"<|start_of_role|>user<|end_of_role|>{message.content}"
                f"<|end_of_text|>\n"
            )
        if isinstance(message, AssistantMessage):
            # Note that we discard any tool calls in the message, per the Jinja
            # template.
            return (
                f"<|start_of_role|>assistant<|end_of_role|>{message.content}"
                f"<|end_of_text|>\n"
            )
        if isinstance(message, ToolResultMessage):
            # Note that we discard the tool call ID, per the Jinja template.
            return (
                f"<|start_of_role|>tool<|end_of_role|>{message.content}"
                f"<|end_of_text|>\n"
            )
        raise TypeError(f"Unexpected message type {type(message)}")

    def _build_controls_record(self, inputs: Granite3Point2Inputs) -> dict | None:
        """
        Use the output control flags in ``inputs`` to build a version of the
        undocumented arbitrary JSON data regarding output controls that the Jinja
        template expected to see in the input for each chat completion request.

        :returns: A fake JSON record for "controls", or nothing of no output control
        flags were set.
        """
        if not inputs.controls:
            return None
        result = {}
        if inputs.controls.citations:
            # The following is a guess; we have no example data for this case.
            result["citations"] = True
        if inputs.controls.hallucinations:
            # The following is a guess; we have no example data for this case.
            result["hallucinations"] = True
        if inputs.controls.length is not None:
            result["length"] = inputs.controls.length
        if inputs.controls.originality is not None:
            result["originality"] = inputs.controls.originality

        if len(result) == 0:
            return None
        return result

    def transform(
        self, inputs: ChatCompletionInputs, add_generation_prompt: bool = True
    ) -> str:
        # Downcast to a Granite-specific request type with possible additional fields.
        # This operation also performs additional validation.
        inputs = Granite3Point2Inputs.model_validate(inputs.model_dump())

        # Check for a caller-provided system message
        system_message_json, loop_messages = self._split_messages(inputs)

        if system_message_json is not None:
            if inputs.thinking:
                raise ValueError(
                    f"'thinking' flag is set, but the model input includes a custom "
                    f"system message. {_GRANITE_3_2_MODEL_NAME} only supports the "
                    f"'thinking' flag when the default system message is used."
                )
            if len(inputs.documents) > 0:
                raise ValueError(
                    f"The model input includes documents and a custom system message. "
                    f"{_GRANITE_3_2_MODEL_NAME} only supports the documents list when "
                    f"the default system message is used."
                )
            if inputs.controls and inputs.controls.citations:
                raise ValueError(
                    f"'citations' flag is set, but the model input includes a custom "
                    f"system message. {_GRANITE_3_2_MODEL_NAME} only supports the "
                    f"'citations' flag when the default system message is used."
                )
            if inputs.controls and inputs.controls.hallucinations:
                raise ValueError(
                    f"'hallucinations' flag is set, but the model input includes a "
                    f"custom system message. {_GRANITE_3_2_MODEL_NAME} only supports "
                    f"the 'hallucinations' flag when the default system message is "
                    f"used."
                )
            system_message = (
                f"<|start_of_role|>system<|end_of_role|>"
                f"{system_message_json.content}<|end_of_text|>\n"
            )
        else:  # if system_message_json is None:
            # No caller-provided system message.
            # Create a default system message according to the rules implied by the
            # tokenizer's Jinja template.
            system_message = self._build_default_system_message(inputs)

        if len(inputs.tools) == 0:
            tools_part = ""
        else:
            tools_part = (
                "<|start_of_role|>tools<|end_of_role|>"
                + json.dumps([t.to_openai_json() for t in inputs.tools], indent=4)
                + "<|end_of_text|>\n"
            )

        if len(inputs.documents) == 0:
            documents_part = ""
        else:
            documents_body = "\n\n".join(
                [
                    f"Document {i}\n{inputs.documents[i].text}"
                    for i in range(len(inputs.documents))
                ]
            )
            documents_part = (
                "<|start_of_role|>documents<|end_of_role|>"
                + documents_body
                + "<|end_of_text|>\n"
            )

        messages_part = "".join(
            [self._message_to_prompt_string(message) for message in loop_messages]
        )

        # Jinja template expects arbitrary JSON, while our dataclass has specific
        # fields for supported controls.
        controls_record = self._build_controls_record(inputs)
        controls_str = (
            "" if controls_record is None else " " + json.dumps(controls_record)
        )

        generation_prompt_part = (
            ""
            if not add_generation_prompt
            else f"<|start_of_role|>assistant{controls_str}<|end_of_role|>"
        )

        return (
            system_message
            + tools_part
            + documents_part
            + messages_part
            + generation_prompt_part
        )
