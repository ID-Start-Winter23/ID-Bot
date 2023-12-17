import os
import openai
import gradio as gr
from llama_index import ServiceContext, StorageContext, load_index_from_storage, set_global_service_context
from llama_index.llms import OpenAI
from llama_index.text_splitter import TokenTextSplitter

from theme import CustomTheme

# create storage context
storage_context = StorageContext.from_defaults(persist_dir="modulhandbuch")
# load index
index = load_index_from_storage(storage_context)

llm = OpenAI(temperature=0.1, model="gpt-4-1106-preview")
splitter =TokenTextSplitter(
     chunk_size=1024,
     chunk_overlap=128,
     separator=" "
)
service_context = ServiceContext.from_defaults(
     llm=llm, 
     text_splitter=splitter
)
set_global_service_context(service_context)

context = (
    "Context information is below.\n"
    "--------------\n"
    "{context_str}\n"
    "--------------\n"
    "Greet the user in a friendly way.\n"
    "Always keep the user on a first-name basis.\n"
    "Answer always in German and in a friendly, humorous matter.\n"
    "Keep the answers short and simple.\n"
    "Tell the user in a friendly way that you can only answer questions about the modules and courses in the study program Informatics and Design if they have questions about other topics.\n"
    "If the user asks a question that you cannot answer, tell them that you cannot answer the question and that they should contact the study program manager.\n"
    "Don't be afraid to ask the user to rephrase the question if you don't understand it.\n"
    "Don't repeat yourself.\n"
)

system_prompt =(
    "You are a study program manager."
)

query_engine = index.as_chat_engine(
    similarity_top_k = 5,
    chat_mode = "context",
    system_prompt = system_prompt,
    context_template = context,
    service_context = service_context,
)

default_text="Ich beantworte Fragen zum Modulhandbuch des Studiengangs Informatik und Design. Wie kann ich Dir helfen?"

bot_examples = [
    "Wer lehrt Mobile Anwendungen?",
    "Welche Prüfungsform hat das Modul Software Engineering?",
    "Wie viele Semesterwochenstunden hat das Modul Computational Thinking?",
]

submit_button = gr.Button(
        value="Ask MUC.DAI",
        elem_classes=["ask-button"],
)

def response(message, history):
    if message == "":
        answer = default_text
    else:
        answer = str(query_engine.chat(message, chat_history=query_engine.chat_history))
    print("message", message)
    print("answer", answer)
    print("history", history)
    return answer


def main():
    #openai.api_key="sk-..."
    openai.api_key = os.environ["OPENAI_API_KEY"]

    custom_theme = CustomTheme()

    chatbot = gr.Chatbot(
        avatar_images=["assets/smile.png", "assets/mucdai.png"],
        layout='bubble',
        height=600,
        value=[[None, default_text]]
    )

    chat_interface = gr.ChatInterface(
        fn=response,
        retry_btn=None,
        undo_btn=None,
        title="MUC.DAI Informatik und Design - frag alles was Du wissen willst!",
        submit_btn=submit_button,
        theme=custom_theme,
        chatbot=chatbot,
        css="style.css",
        examples=bot_examples,
    )

    chat_interface.launch(inbrowser=True, debug=True)


if __name__ == "__main__":
    main()
