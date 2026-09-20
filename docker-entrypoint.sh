#!/bin/sh
set -eu

if [ "${1:-}" = "prepare" ]; then
    python -m ai_agent.catalog restore
    python -m ai_agent.rag clear
    python -m ai_agent.rag prepare
    python -m ai_agent.rag chunk
    python -m ai_agent.rag embedding
    python -m ai_agent.rag vector_store
    exit 0
fi

exec "$@"
