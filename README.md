# rag-agentic-folder-qa

Ask a tiny folder of neighborhood menus. Continues [01 - messy notes to JSON](https://wysiwygs.de/blog/messy-notes-to-json-langchain-flask/). Same three names on Linden Street; these files are the menus, not the shift pad.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Task 1 does not call a model. From Task 2 on, put your OpenAI key in `.env`.

## Task 1 — load and split

```powershell
python src/load_split.py
```

You should see three files become a handful of chunks, each tagged with its source. If a chunk is mid-sentence, that is the splitter: size 250, overlap 50.

## Task 2 — embed and store

```powershell
python src/embed_store.py
```

Same chunks, now each one is a list of numbers in an in-memory store. You should see `9 vectors` and `1536 numbers`. Then the three nearest chunks for a cilantro-rice question. That is search, not an answer. There is still no chat model.

## Task 3 — retrieve and stuff

```powershell
python src/retrieve_stuff.py
```

Same three chunks as Task 2, then those strings pasted into `{context}`. You should see the filled prompt (the menus sitting in the human message) and then a short answer: cilantro rice is Tuesday and Friday, not Wednesday. Without the stuffing step the model would guess.
