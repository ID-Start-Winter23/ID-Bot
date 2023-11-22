import os
import openai
import gradio as gr
from llama_index import StorageContext, load_index_from_storage

from theme import CustomTheme


def response(message, history):
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="modulhandbuch")
    # load index
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine()
    answer = query_engine.query(message)

    return str(answer)


def main():
    openai.api_key = os.environ["OPENAI_API_KEY"]

    custom_theme = CustomTheme()

    chatbot = gr.ChatInterface(
        fn=response,
        retry_btn=None,
        undo_btn="Name",
        theme=custom_theme,
    )

    chatbot.launch(inbrowser=True)


if __name__ == "__main__":
    main()
