set dotenv-load

default: lock  install  lint  test

# Создать локальный .env из шаблона, не перезаписывая существующий файл.
env_init:
    @if [ -f .env ]; then \
        echo ".env уже существует, файл не изменён."; \
    else \
        cp .env.example .env; \
        echo ".env создан из .env.example."; \
    fi

# Остановить recipe с понятной ошибкой, если .env ещё не создан.
_require_env:
    @test -f .env || (echo ".env не найден. Выполните: just env_init" && exit 1)

# Установить или обновить pyenv и uv через Homebrew.
setup:
    brew update
    brew install pyenv uv
    brew upgrade pyenv uv

# Установить и выбрать указанную версию Python, затем пересоздать .venv.
python version:
    if ! pyenv versions --bare | grep -qx "{{version}}"; then \
        pyenv install {{version}}; \
    fi
    pyenv local {{version}}
    pyenv rehash

    uv python pin {{version}}
    uv venv --python {{version}} --clear

# Обновить uv.lock без ручной фиксации версий зависимостей.
lock:
    uv lock --upgrade

# Установить все основные и dev-зависимости из uv.lock.
install:
    uv sync --frozen --all-extras --no-install-project --all-groups
    . ./.venv/bin/activate

# Отформатировать код и выполнить статические проверки.
lint:
    uv run auto-typing-final .
    uv run ruff format .
    uv run ruff check --fix .
    uv run ty check .

# Подготовить runtime, выполнить тесты с coverage и очистить runtime.
test: _require_env prepare_agent
    uv run --env-file .env pytest --cov=ai_agent --cov-report=term-missing --cov-report=json:tests/coverage_report.json
    just clear_runtime

# Восстановить SQLite-каталог из версионируемого CSV-слепка.
catalog_restore: _require_env
    uv run --env-file .env python -m ai_agent.catalog restore

# Удалить сгенерированную SQLite-базу каталога.
catalog_clear: _require_env
    uv run --env-file .env python -m ai_agent.catalog clear

# Удалить SQLite-базу долговременной памяти.
memory_clear: _require_env
    uv run --env-file .env python -m ai_agent.memory clear

# Подготовить и нормализовать исходные документы базы знаний.
rag_prepare: _require_env
    uv run --env-file .env python -m ai_agent.rag prepare

# Разбить подготовленные документы на фрагменты.
rag_chunk: _require_env
    uv run --env-file .env python -m ai_agent.rag chunk

# Рассчитать embeddings для фрагментов базы знаний.
rag_embedding: _require_env
    uv run --env-file .env python -m ai_agent.rag embedding

# Построить vector store из рассчитанных embeddings.
rag_vectorstore: _require_env
    uv run --env-file .env python -m ai_agent.rag vector_store

# Удалить все сгенерированные RAG-артефакты.
rag_clear: _require_env
    uv run --env-file .env python -m ai_agent.rag clear

# Полностью пересобрать vector store из исходных документов.
rag_rebuild: rag_clear rag_prepare rag_chunk rag_embedding rag_vectorstore

# Восстановить каталог и пересобрать vector store для локального запуска.
prepare_agent: catalog_restore rag_rebuild

# Запустить интерактивный CLI агента с дополнительными аргументами.
run_agent_cli *args: _require_env
    uv run --env-file .env python -m ai_agent {{args}}

# Запустить HTTP API агента локально.
run_agent_api: _require_env
    uv run --env-file .env python -m ai_agent.service

# Собрать Docker-образ с именем из DOCKER_IMAGE_NAME.
docker_build: _require_env
    docker build --tag "$DOCKER_IMAGE_NAME" .

# Выполнить команду приложения в одноразовом контейнере с runtime volume.
_docker_run *args: _require_env
    docker run --rm \
        --env-file .env \
        --volume "$DOCKER_RUNTIME_VOLUME_NAME:/app/runtime" \
        "$DOCKER_IMAGE_NAME" {{args}}

# Очистить память, каталог и RAG-артефакты в persistent Docker volume.
docker_clear_runtime: _require_env
    just _docker_run python -m ai_agent.memory clear
    just _docker_run python -m ai_agent.catalog clear
    just _docker_run python -m ai_agent.rag clear

# Подготовить каталог и RAG-артефакты в persistent Docker volume.
docker_prepare: _require_env
    just _docker_run python -m ai_agent.catalog restore
    just _docker_run python -m ai_agent.rag clear
    just _docker_run python -m ai_agent.rag prepare
    just _docker_run python -m ai_agent.rag chunk
    just _docker_run python -m ai_agent.rag embedding
    just _docker_run python -m ai_agent.rag vector_store

# Запустить HTTP API из Docker-образа с подготовленным volume.
docker_run_agent_api: _require_env
    docker run --rm \
        --name netology-aiagnt \
        --env-file .env \
        --publish "$API_PORT:$API_PORT" \
        --volume "$DOCKER_RUNTIME_VOLUME_NAME:/app/runtime" \
        "$DOCKER_IMAGE_NAME"

# Проверить, что HTTP-процесс запущен.
api_health: _require_env
    curl --fail --silent --show-error "http://localhost:$API_PORT/health"

# Проверить готовность агента, каталога и vector store.
api_ready: _require_env
    curl --fail --silent --show-error "http://localhost:$API_PORT/ready"

# Открыть Swagger UI локального API в браузере.
open_swagger: _require_env
    open "http://localhost:$API_PORT/docs"

# Удалить память, каталог и все сгенерированные RAG-артефакты.
clear_runtime: memory_clear catalog_clear rag_clear
