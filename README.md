# rag-agentic-folder-qa

Ask a tiny folder of neighborhood menus. Continues [01 - messy notes to JSON](https://wysiwygs.de/blog/messy-notes-to-json-langchain-flask/). Same three names on Linden Street; these files are the menus, not the shift pad.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Task 1 does not call a model. The key is for later files.

## Task 1 — load and split

```powershell
python src/load_split.py
```

You should see three files become a handful of chunks, each tagged with its source. If a chunk is mid-sentence, that is the splitter: size 250, overlap 50.
