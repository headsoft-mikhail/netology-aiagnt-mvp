FROM ghcr.io/astral-sh/uv:0.12.17 AS uv

FROM python:3.14.5-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --no-install-project

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

RUN useradd --create-home --uid 10001 agent \
    && mkdir -p /app/runtime \
    && chown -R agent:agent /app

USER agent

VOLUME ["/app/runtime"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; port = os.environ['API_PORT']; urllib.request.urlopen(f'http://localhost:{port}/ready', timeout=3)" || exit 1

CMD ["python", "-m", "ai_agent.service"]
