# Standard
import os

# Local
from granite_io import make_backend, make_io_processor
from granite_io.backend.vllm_server import LocalVLLMServer
from granite_io.io.granite_3_3.input_processors.granite_3_3_input_processor import (
    Granite3Point3Inputs,
)
from granite_io.io.llmrerank import RerankRequestProcessor
from granite_io.io.retrieval import InMemoryRetriever, RetrievalRequestProcessor
from granite_io.io.retrieval.util import download_mtrag_embeddings


def main():
    # Constants go here
    temp_data_dir = "data/test_retrieval"
    corpus_name = "govt"
    embedding_model_name = "multi-qa-mpnet-base-dot-v1"
    # embedding_model_name = "ibm-granite/granite-embedding-30m-english"
    model_name = "ibm-granite/granite-3.3-8b-instruct"

    run_server = True

    if run_server:
        # Start by firing up a local vLLM server and connecting a backend instance to it
        server = LocalVLLMServer(model_name)
        server.wait_for_startup(200)
        backend = server.make_backend()
    else:  # if not run_server
        # Use an existing server.
        # The constants here are for the server that local_vllm_server.ipynb starts.
        # Modify as needed.
        openai_base_url = "http://localhost:55555/v1"
        openai_api_key = ""
        backend = make_backend(
            "openai",
            {
                "model_name": model_name,
                "openai_base_url": openai_base_url,
                "openai_api_key": openai_api_key,
            },
        )
        # backend=make_backend("openai", {"model_name": model_name})

    # Download the indexed corpus if it hasn't already been downloaded.
    # This example uses a subset of the government corpus from the MTRAG benchmark.
    embeddings_location = f"{temp_data_dir}/{corpus_name}_embeds.parquet"
    if not os.path.exists(embeddings_location):
        download_mtrag_embeddings(
            embedding_model_name, corpus_name, embeddings_location
        )

    # Spin up an IO processor for the base model
    io_proc = make_io_processor(model_name, backend=backend)

    # Create an example chat completions request
    chat_input = Granite3Point3Inputs.model_validate(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Welcome to the California Appellate Courts help desk.",
                },
                {
                    "role": "user",
                    "content": "I need to do some legal research to be prepared for my "
                    "oral argument. Can I visit the law library?",
                },
            ],
            "generate_inputs": {
                "temperature": 0.0,
                "max_tokens": 4096,
            },
        }
    )

    # Spin up an in-memory vector database
    retriever = InMemoryRetriever(embeddings_location, embedding_model_name)

    # Use a RetrievalRequestProcessor to augment the chat completion request with
    # documents.
    rag_processor = RetrievalRequestProcessor(retriever, top_k=32)
    rag_chat_input = rag_processor.process(chat_input)[0]

    # rerank with llm
    rerank_processor = RerankRequestProcessor(
        io_proc, rerank_top_k=32, return_top_k=5, verbose=True
    )
    rerank_chat_input = rerank_processor.process(rag_chat_input)

    rag_output = io_proc.create_chat_completion(rerank_chat_input)
    print(rag_output.results[0].next_message.content)


if __name__ == "__main__":
    main()
