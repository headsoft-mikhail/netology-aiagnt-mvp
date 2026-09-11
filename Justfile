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

test:
    uv run pytest

run:
    uv run python ai_agent
