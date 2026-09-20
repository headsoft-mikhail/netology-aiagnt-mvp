import typing

import fastapi
import modern_di_fastapi

from ai_agent import contracts
from ai_agent.catalog.config import catalog_config
from ai_agent.rag.retrieval import retrieval
from ai_agent.rag.retrieval.config import rag_retrieval_config
from ai_agent.service import dependencies, models

SERVICE_UNAVAILABLE_MESSAGE: typing.Final = "Агент ещё не готов. Подготовьте каталог и базу знаний."

router: typing.Final = fastapi.APIRouter()
type InjectedAgentRuntime = typing.Annotated[
    dependencies.AgentRuntime,
    modern_di_fastapi.FromDI(dependencies.AgentRuntime),
]


@router.get("/health", response_model=models.HealthResponse, tags=["service"])
def health() -> models.HealthResponse:
    """Confirm that the HTTP process is running."""
    return models.HealthResponse()


@router.get(
    "/ready",
    response_model=models.ReadinessResponse,
    responses={fastapi.status.HTTP_503_SERVICE_UNAVAILABLE: {"model": models.ReadinessResponse}},
    tags=["service"],
)
def ready(
    response: fastapi.Response,
    agent_runtime: InjectedAgentRuntime,
) -> models.ReadinessResponse:
    """Check runtime data and the initialized agent."""
    catalog_ready: typing.Final = catalog_config.database_path.is_file()
    knowledge_base_ready: typing.Final = retrieval.is_vector_store_ready(rag_retrieval_config)
    is_ready: typing.Final = agent_runtime.is_ready and catalog_ready and knowledge_base_ready
    if not is_ready:
        response.status_code = fastapi.status.HTTP_503_SERVICE_UNAVAILABLE
    return models.ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        agent_ready=agent_runtime.is_ready,
        catalog_ready=catalog_ready,
        knowledge_base_ready=knowledge_base_ready,
    )


@router.post("/chat", response_model=contracts.AgentResponse, tags=["agent"])
def chat(
    payload: models.ChatRequest,
    agent_runtime: InjectedAgentRuntime,
) -> contracts.AgentResponse:
    """Run one turn while preserving server-side session memory."""
    agent_runner: typing.Final = agent_runtime.agent_runner
    if agent_runner is None:
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=SERVICE_UNAVAILABLE_MESSAGE,
        )
    return agent_runner.run(
        payload.user_id,
        payload.message,
        session_id=payload.session_id,
    )
