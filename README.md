# rag-menus-chat

This repo demonstrates how to create a chatbot on top of domain-specific files. The files are a few kitchen menus (markdown and a PDF). We will get our chat application answering questions from these resources as the source of truth. In other words, this is a RAG application.

**Article:** [RAG: Ask a folder of menus with LangChain and Gradio](https://wysiwygs.de/blog/ask-a-folder-of-menus-langchain-gradio/)

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Put your OpenAI API key in `.env`.

## Load the folder and split it

```powershell
python src/load_split.py
```

## Now we embed and store

```powershell
python src/embed_store.py
```

## Retrieve and stuff the prompt

```powershell
python src/retrieve_stuff.py
```

## Serve it with Gradio

```powershell
python src/app.py
```

Open the local URL Gradio prints.

## Same flow in LlamaIndex

```powershell
python src/llamaindex_qa.py
```

Same menus, same question. Only the library changes (LangChain then LlamaIndex). You should see Wednesday is still no. Wording can differ but the answer should not. The script also fetches a fourth menu from a gist. Set `MENU_URL` in `.env` if you want a different URL.
