set dotenv-load

docker_image_name := env("DOCKER_IMAGE_NAME")
docker_runtime_volume_name := env("DOCKER_RUNTIME_VOLUME_NAME")
api_port := env("API_PORT")

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
    uv run --env-file .env pytest --cov=ai_agent --cov-report=term-missing --cov-report=json:tests/coverage_report.json
    just clear_runtime

catalog_restore:
    uv run --env-file .env python -m ai_agent.catalog restore

catalog_clear:
    uv run --env-file .env python -m ai_agent.catalog clear

memory_clear:
    uv run --env-file .env python -m ai_agent.memory clear

rag_prepare:
    uv run --env-file .env python -m ai_agent.rag prepare

rag_chunk:
    uv run --env-file .env python -m ai_agent.rag chunk

rag_embedding:
    uv run --env-file .env python -m ai_agent.rag embedding

rag_vectorstore:
    uv run --env-file .env python -m ai_agent.rag vector_store

rag_clear:
    uv run --env-file .env python -m ai_agent.rag clear

rag_rebuild: rag_clear rag_prepare rag_chunk rag_embedding rag_vectorstore

prepare_agent: catalog_restore rag_rebuild

run_agent *args:
    uv run --env-file .env python -m ai_agent {{args}}

run_agent_api:
    uv run --env-file .env python -m ai_agent.service

docker_build:
    docker build --tag {{docker_image_name}} .

docker_prepare:
    docker run --rm \
        --env-file .env \
        --volume {{docker_runtime_volume_name}}:/app/runtime \
        {{docker_image_name}} prepare

docker_run_agent_api:
    docker run --rm \
        --name netology-aiagnt \
        --env-file .env \
        --publish {{api_port}}:{{api_port}} \
        --volume {{docker_runtime_volume_name}}:/app/runtime \
        {{docker_image_name}}

clear_runtime: memory_clear catalog_clear rag_clear
