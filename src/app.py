"""Task 4: same retrieve-and-stuff chain, behind a Gradio chat."""

from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

MENU_DIR = Path("data/menus")

docs = []
for path in sorted(MENU_DIR.iterdir()):
    if path.suffix == ".md":
        docs.extend(TextLoader(path, encoding="utf-8").load())
    elif path.suffix == ".pdf":
        docs.extend(PyPDFLoader(path).load())

chunks = RecursiveCharacterTextSplitter(
    chunk_size=250, chunk_overlap=50
).split_documents(docs)
store = InMemoryVectorStore.from_documents(
    chunks, OpenAIEmbeddings(model="text-embedding-3-small")
)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer from the menus only. If it is not in the menus, say you do not know.",
        ),
        (
            "human",
            "Menus:\n{context}\n\n{history}Question: {question}",
        ),
    ]
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def chat(message, history):
    hits = store.similarity_search(message, k=3)
    context = "\n\n".join(doc.page_content for doc in hits)
    prior = []
    for item in history[-6:]:
        role = "Customer" if item["role"] == "user" else "Assistant"
        prior.append(f"{role}: {item['content']}")
    history_text = ("\n".join(prior) + "\n\n") if prior else ""
    answer = (prompt | llm).invoke(
        {"context": context, "history": history_text, "question": message}
    ).content
    names = ", ".join(
        sorted({Path(doc.metadata["source"]).name for doc in hits})
    )
    return f"{answer}\n\nSources: {names}"


if __name__ == "__main__":
    gr.ChatInterface(
        chat,
        type="messages",
        title="Linden Street menus",
        examples=["Can I get cilantro rice on Wednesday?", "Do you offer Hummus?"],
    ).launch()
