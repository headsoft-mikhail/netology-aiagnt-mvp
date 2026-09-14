default: lock  install  lint  test

setup:
    brew update
    brew install pyenv uv
    brew upgrade pyenv uv

python version:
    if ! pyenv versions --bare | grep -qx "{{version}}"; then \
        pyenv install {{version}}; \
    fi
    pyenv local {{version}}
    pyenv rehash

    uv python pin {{version}}
    uv venv --python {{version}} --clear

lock:
    uv lock --upgrade

install:
    uv sync --frozen --all-extras --no-install-project --all-groups
    . ./.venv/bin/activate

lint:
    uv run auto-typing-final .
    uv run ruff format .
    uv run ruff check --fix .
    uv run ty check .

test: prepare_agent
    uv run pytest
    just clear_runtime

catalog_restore:
    uv run python -m ai_agent.catalog restore

catalog_clear:
    uv run python -m ai_agent.catalog clear

memory_clear:
    uv run python -m ai_agent.memory clear

rag_prepare:
    uv run python -m ai_agent.rag prepare

rag_chunk:
    uv run python -m ai_agent.rag chunk

rag_embedding:
    uv run python -m ai_agent.rag embedding

rag_vectorstore:
    uv run python -m ai_agent.rag vector_store

rag_clear:
    uv run python -m ai_agent.rag clear

rag_rebuild: rag_clear rag_prepare rag_chunk rag_embedding rag_vectorstore

prepare_agent: catalog_restore rag_rebuild

run_agent:
    uv run python -m ai_agent

clear_runtime: memory_clear catalog_clear rag_clear
