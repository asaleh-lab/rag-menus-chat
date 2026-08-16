"""Task 5: same RAG flow in LlamaIndex. APIs differ; the menus do not."""

from pathlib import Path
import os
import re

import httpx
from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

LENA_MENU_URL = (
    "https://gist.githubusercontent.com/asaleh-lab/"
    "8a148ffc26b8278ba7c2fbc7664c240c/raw/lena-bakery.html"
)

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
splitter = SentenceSplitter(chunk_size=250, chunk_overlap=50)

folder_docs = SimpleDirectoryReader("data/menus").load_data()
print(
    f"SimpleDirectoryReader: {len(folder_docs)} docs "
    "(LangChain: TextLoader + PyPDFLoader)"
)

url = os.getenv("MENU_URL", "").strip() or LENA_MENU_URL
html = httpx.get(url, timeout=20, follow_redirects=True).text
text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()
web_doc = Document(text=text, metadata={"file_name": url})
print(
    f"httpx.get: {len(text)} chars from {url} "
    "(icebreaker fetched LinkedIn; we fetch a menu page)"
)

nodes = splitter.get_nodes_from_documents(folder_docs + [web_doc])
print(
    f"SentenceSplitter: {len(nodes)} nodes "
    "(LangChain: RecursiveCharacterTextSplitter; size here is tokens)"
)

index = VectorStoreIndex(nodes)
print("VectorStoreIndex (LangChain: InMemoryVectorStore.from_documents)")

qa_tmpl = ChatPromptTemplate(
    message_templates=[
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "Answer from the menus only. "
                "If it is not in the menus, say you do not know."
            ),
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="Menus:\n{context_str}\n\nQuestion: {query_str}",
        ),
    ]
)
engine = index.as_query_engine(
    similarity_top_k=3,
    text_qa_template=qa_tmpl,
    response_mode="compact",
)
print("as_query_engine (LangChain: stuff {context} into ChatPromptTemplate by hand)")

query = "Can I get cilantro rice on Wednesday?"
print(f"\nRetrieved for: {query}")
for node in index.as_retriever(similarity_top_k=3).retrieve(query):
    name = Path(str(node.metadata.get("file_name", "?"))).name
    print(f"--- {name} ---")
    print(node.text)
    print()

print(engine.query(query))
print()
print(engine.query("Does Lena sell sesame tahini rolls on Monday?"))
