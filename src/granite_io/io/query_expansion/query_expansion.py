# SPDX-License-Identifier: Apache-2.0

"""
I/O processor for the Granite query expansion intrinsic.
"""

# Standard
import asyncio

# Local
from granite_io.io.base import (
    InputOutputProcessor,
)
from granite_io.types import (
    ChatCompletionInputs,
    ChatCompletionResult,
    ChatCompletionResults,
    GenerateInputs,
    UserMessage,
)


class QueryExpansionIOProcessor(InputOutputProcessor):
    def __init__(
        self,
        backend,
        io_processor,
        rewrite_request_proc,
    ):
        self.backend = backend
        self.io_proc = io_processor
        self.rewrite_request_proc = rewrite_request_proc

    @staticmethod
    def format_chat_history(messages):
        formatted = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    async def acreate_chat_completion(
        self, inputs: ChatCompletionInputs
    ) -> ChatCompletionResults:
        coroutines = []
        #################### Sample Answer from Granite ####################
        chat_input_ans_v1 = ChatCompletionInputs(
            messages=inputs.messages,
            generate_inputs={
                "temperature": inputs.generate_inputs.temperature,
                "top_p": 1,
                "max_tokens": 512,
            },
        )
        sampled_answer_coroutine = self.io_proc.acreate_chat_completion(
            chat_input_ans_v1
        )
        #################### Run QUERY REWRITE LoRA ####################
        coroutines.append(self.rewrite_request_proc.acreate_chat_completion(inputs))
        ####### Reformulate into Synonymous Query by prompting Granite #######
        input_conversation_string = self.format_chat_history(
            inputs.model_dump()["messages"]
        )
        # print(f"Input Conversation: {input_conversation_string}")

        generate_inputs = GenerateInputs(
            max_tokens=512,
            top_p=1,
            temperature=inputs.generate_inputs.temperature,
            stop="[[Input]]",
        )

        generate_inputs.prompt = (
            "You are given a multi-turn conversation between a user and an assistant. "
            "Reformulate the last-turn user query into a synonymous standalone query "
            "by replacing key terms with appropriate synonyms or closely related "
            "phrases, while preserving the original intent and meaning. "
            "This rewritten query will be used to retrieve relevant passages "
            "from a corpus, so it must remain faithful to the user's information need. "
            "Only output the rewritten query.\n\n[[Input]]\n"
            f"{input_conversation_string}\n\n[[Output]]\n"
        )

        coroutines.append(self.backend.pipeline(generate_inputs))
        #################### Await for Sampled Answer ####################
        query_answer_v1 = await sampled_answer_coroutine
        query_answer_v1 = query_answer_v1.results[0].next_message.content
        ####### Reverse-Engineer the Question from the Sampled Answer #######
        generate_inputs = GenerateInputs(
            max_tokens=512,
            top_p=1,
            temperature=inputs.generate_inputs.temperature,
            stop="[[Answer]]",
        )

        generate_inputs.prompt = f"""Generate a single question for the given answer.
        [[Answer]]
        Albert Einstein was born in Germany.
        [[Question]]
        Where was Albert Einstein born?
        [[Answer]]
        {query_answer_v1}
        [[Question]]
        """
        coroutines.append(self.backend.pipeline(generate_inputs))
        ######################################################################
        # Merge results from parallel invocations
        sub_results = await asyncio.gather(*coroutines)

        # print(
        #     f"QUERY REWRITE: {sub_results[0]}\nSYNONYMOUS QUERY: {sub_results[1]}",
        #     f"Reverse-Engineered Question: {sub_results[2]}",
        # )

        last_user_message = inputs.model_dump()["messages"][-1]
        query_str_list = [
            last_user_message["content"],  # 0
            sub_results[0].results[0].next_message.content,  # 1
            sub_results[1].results[0].completion_string,  # 2
            sub_results[2].results[0].completion_string,  # 3
            query_answer_v1,  # 4
        ]

        # print("\nList of Query Strings:", query_str_list)

        results = []
        for cur_query in query_str_list:
            results.append(
                ChatCompletionResult(next_message=UserMessage(content=cur_query))
            )
        # print("results", results)

        return ChatCompletionResults(results=results)
