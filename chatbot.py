import gradio as gr
import time
import asyncio
import tiktoken
from llama_index import set_global_service_context, ServiceContext
from llama_index import Document, VectorStoreIndex
from llama_index.node_parser import TokenTextSplitter
from llama_index.llms import MockLLM
from llama_index import MockEmbedding
from llama_index.callbacks import CallbackManager, TokenCountingHandler
from llama_index.memory import ChatMemoryBuffer

llm = MockLLM(max_tokens=128)
embed_model = MockEmbedding(embed_dim=1536)
memory = ChatMemoryBuffer.from_defaults(token_limit=1500)


token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)

callback_manager = CallbackManager([token_counter])

set_global_service_context(
    ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, callback_manager=callback_manager
    )
)

text = """
What I Worked On

February 2021

Before college the two main things I worked on, outside of school, were writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, and probably still are: short stories. My stories were awful. They had hardly any plot, just characters with strong feelings, which I imagined made them deep.

The first programs I tried writing were on the IBM 1401 that our school district used for what was then called "data processing." This was in 9th grade, so I was 13 or 14. The school district's 1401 happened to be in the basement of our junior high school, and my friend Rich Draves and I got permission to use it. It was like a mini Bond villain's lair down there, with all these alien-looking machines — CPU, disk drives, printer, card reader — sitting up on a raised floor under bright fluorescent lights.

The language we used was an early version of Fortran. You had to type programs on punch cards, then stack them in the card reader and press a button to load the program into memory and run it. The result would ordinarily be to print something on the spectacularly loud printer.
"""

document = Document(
    text=text,
    metadata={
        "Topic": "Paul Graham",
    },
)
node_parser = TokenTextSplitter(
    chunk_size=512,
    chunk_overlap=128,
    separator=" ")

nodes = node_parser.get_nodes_from_documents([document])
index = VectorStoreIndex(nodes)

chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=memory,
)

user_message = ""

print(
    "Embedding Tokens Index: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens Index: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens Index: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count Index: ",
    token_counter.total_llm_token_count,
    "\n",
)
token_counter.reset_counts()


def add_text(message, history):
    global user_message
    user_message = message
    #history = history + [(message, None)]
    return "", history


def response(history):
    # check if storage already exists
    loop = asyncio.new_event_loop()
    # Set the event loop as the current event loop
    asyncio.set_event_loop(loop)

    #shorten remove chat_history
    answer = chat_engine.stream_chat(user_message, [])

    streamed_output = ""

    for token in answer.response_gen:
        streamed_output += token
        time.sleep(0.05)

        yield history

    print(
        "Embedding Tokens Response: ",
        token_counter.total_embedding_token_count,
        "\n",
        "LLM Prompt Tokens Response: ",
        token_counter.prompt_llm_token_count,
        "\n",
        "LLM Completion Tokens Response: ",
        token_counter.completion_llm_token_count,
        "\n",
        "Total LLM Token Count Response: ",
        token_counter.total_llm_token_count,
        "\n",
    )


with gr.Blocks() as chatbot:
    output = gr.Chatbot(show_label=False, value=[[None, "Hallo"]])
    message = gr.Textbox(placeholder="Message", show_label=False)
    clear = gr.ClearButton([message, chatbot])

    message.submit(add_text, [message, output], [message, output], queue=False).then(response, output, output)
    #reset after each message
    chat_engine.reset()

chatbot.launch(inbrowser=True)
