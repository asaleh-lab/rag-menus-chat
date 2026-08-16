"""Task 3: retrieve k chunks, stuff them into a prompt, generate."""

from pathlib import Path

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

query = "Can I get cilantro rice on Wednesday?"
hits = store.similarity_search(query, k=3)

print("Retrieved:")
for doc in hits:
    print(f"--- {Path(doc.metadata['source']).name} ---")
    print(doc.page_content)
    print()

context = "\n\n".join(doc.page_content for doc in hits)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer from the menus only. If it is not in the menus, say you do not know.",
        ),
        ("human", "Menus:\n{context}\n\nQuestion: {question}"),
    ]
)
messages = prompt.invoke({"context": context, "question": query})
print("Filled prompt:")
print(messages.to_string())
print("---")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print(llm.invoke(messages).content)
