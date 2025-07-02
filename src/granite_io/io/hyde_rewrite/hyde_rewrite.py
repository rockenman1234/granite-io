# SPDX-License-Identifier: Apache-2.0

"""
I/O processor for the Granite query rewrite intrinsic.
"""

# Local
from granite_io.io.base import (
    ModelDirectInputOutputProcessorWithGenerate,
)
from granite_io.io.granite_3_3.input_processors.granite_3_3_input_processor import (
    Granite3Point3InputProcessor,
    Granite3Point3Inputs,
)
from granite_io.types import (
    ChatCompletionInputs,
    ChatCompletionResult,
    ChatCompletionResults,
    GenerateInputs,
    GenerateResults,
)

HYDE_INSTRUCTION = (
    "Generate a short answer for the user query below. Do not use more than 50 words.\n"
)

HYDE_MAX_TOKENS = 256


class HyDERewriteIOProcessor(ModelDirectInputOutputProcessorWithGenerate):
    """
    I/O processor for Query rewrite using HyDE https://arxiv.org/abs/2212.10496

    Takes as input a chat completion and returns a completion with a rewrite of the
    most recent user turn (last in the conversation).

    Example raw input:
    ```
    <|start_of_role|>user<|end_of_role|>What items I need to keep in the safe room?<|end_of_text|>
    ```

    Example of corresponding raw output:
    ```
    <|start_of_role|>user<|end_of_role|>What items I need to keep in the safe room? A safe room should contain a supply of water, non-perishable food, first aid kit <|end_of_text|>
    ```

    Output string with the rewrite of the last user turn:
        'What items I need to keep in the safe room? A safe room should contain a supply of water, non-perishable food, first aid kit'
    """  # noqa: E501

    def __init__(self, backend):
        super().__init__(backend=backend)

        # Input processor for the base model, which does most of the input formatting.
        self.base_input_processor = Granite3Point3InputProcessor()

    def inputs_to_generate_inputs(
        self, inputs: ChatCompletionInputs, add_generation_prompt: bool = True
    ) -> GenerateInputs:
        # Validate the input and convert to Granite input
        inputs = Granite3Point3Inputs.model_validate(inputs.model_dump())

        # Check for the invariants that the model expects its input to satisfy
        if not inputs.messages[-1].role == "user":
            raise ValueError("Last message is not a user message")

        # To invoke the model, we add the rewrite prompt to the prompt prefix:
        rewritten_messages = [m.model_copy() for m in inputs.messages[-1:]]
        rewritten_messages[-1].content = (
            HYDE_INSTRUCTION + rewritten_messages[-1].content
        )
        rewritten_inputs = inputs.model_copy(update={"messages": rewritten_messages})
        prompt = self.base_input_processor.transform(
            rewritten_inputs, add_generation_prompt
        )
        result = inputs.generate_inputs.model_copy(
            update={"prompt": prompt, "max_tokens": HYDE_MAX_TOKENS}
        )
        return result

    def output_to_result(
        self,
        output: GenerateResults,
        inputs: ChatCompletionInputs | None = None,  # pylint: disable=unused-argument
    ) -> ChatCompletionResults:
        results = []
        for raw_result in output.results:
            completion_string = raw_result.completion_string

            orig_query = inputs.messages[-1].content
            # Change content but retain other properties of the message.
            rewritten_last_message = inputs.messages[-1].model_copy(
                update={"content": orig_query + " " + completion_string}
            )
            results.append(ChatCompletionResult(next_message=rewritten_last_message))

        return ChatCompletionResults(results=results)
