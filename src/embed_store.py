"""Task 2: embed the chunks and keep them in memory. No LLM answer yet."""

from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
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

print(f"Stored {len(store.store)} vectors")
first = next(iter(store.store.values()))
print(f"Each vector has {len(first['vector'])} numbers")

query = "Can I get cilantro rice on Wednesday?"
print(f"\nNearest chunks for: {query}")
for doc in store.similarity_search(query, k=3):
    print(f"--- {Path(doc.metadata['source']).name} ---")
    print(doc.page_content)
    print()
