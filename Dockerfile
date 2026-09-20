FROM ghcr.io/astral-sh/uv:0.12.17 AS uv

FROM python:3.14.5-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN useradd --create-home --uid 10001 agent \
    && mkdir -p /app/runtime \
    && chown -R agent:agent /app \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

USER agent

VOLUME ["/app/runtime"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; port = os.environ['API_PORT']; urllib.request.urlopen(f'http://127.0.0.1:{port}/ready', timeout=3)" || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "-m", "ai_agent.service"]
