uv init hoopla
cd hoopla

uv venv

source .venv/bin/activate

uv run cli/keyword_search_cli.py search "Great"

Implementing stemming from scratch is a lot of work, so we'll use the nltk.stem library to handle it for us.
uv add nltk==3.9.1