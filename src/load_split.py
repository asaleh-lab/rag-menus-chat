"""Task 1: load the menu folder, split into chunks. Nothing else yet."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

MENU_DIR = Path("data/menus")

docs = []
for path in sorted(MENU_DIR.iterdir()):
    if path.suffix == ".md":
        docs.extend(TextLoader(path, encoding="utf-8").load())
    elif path.suffix == ".pdf":
        docs.extend(PyPDFLoader(path).load())

print(f"Loaded {len(docs)} docs from {MENU_DIR}")
for doc in docs:
    print(f"  {Path(doc.metadata['source']).name}: {len(doc.page_content)} chars")

splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
chunks = splitter.split_documents(docs)

print(f"\n{len(chunks)} chunks (size 250, overlap 50)")
for i, chunk in enumerate(chunks, start=1):
    name = Path(chunk.metadata["source"]).name
    print(f"--- chunk {i} | {name} ---")
    print(chunk.page_content)
    print()
