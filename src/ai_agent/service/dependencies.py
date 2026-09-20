import dataclasses
import logging
import typing

import modern_di
import modern_di.providers

from ai_agent.agent import AgentRunner
from ai_agent.factory import create_agent

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class AgentRuntime:
    agent_runner: AgentRunner | None

    @property
    def is_ready(self) -> bool:
        return self.agent_runner is not None


def create_agent_runtime() -> AgentRuntime:
    try:
        return AgentRuntime(agent_runner=create_agent())
    except Exception:
        LOGGER_OBJ.exception("Agent initialization failed")
        return AgentRuntime(agent_runner=None)


class ServiceDependencies(modern_di.Group):
    agent_runtime = modern_di.providers.Factory(
        create_agent_runtime,
        scope=modern_di.Scope.APP,
        cache=True,
    )


def create_container() -> modern_di.Container:
    return modern_di.Container(groups=[ServiceDependencies])
