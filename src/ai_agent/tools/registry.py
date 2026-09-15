import dataclasses
import json
import typing

from ai_agent import contracts
from ai_agent.tools import protocol


@dataclasses.dataclass(frozen=True, slots=True)
class ToolExecution:
    result: protocol.ToolResult
    trace: contracts.ToolCallTrace


@dataclasses.dataclass(kw_only=True, slots=True)
class ToolRegistry:
    tools: tuple[protocol.Tool, ...]
    _tools_by_name: dict[str, protocol.Tool] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self._tools_by_name = {}
        for tool in self.tools:
            if tool.name in self._tools_by_name:
                raise ValueError(f"Tool is already registered: {tool.name}")
            self._tools_by_name[tool.name] = tool

    def execute(self, tool_name: str, arguments: dict[str, object]) -> ToolExecution:
        try:
            tool: typing.Final = self._tools_by_name[tool_name]
        except KeyError as error:
            raise ValueError(f"Unknown Tool: {tool_name}") from error

        tool_input: typing.Final = tool.input_model.model_validate(arguments)
        result: typing.Final = tool.invoke(tool_input)
        trace: typing.Final = contracts.ToolCallTrace(
            tool_name=tool.name,
            status=result.status,
            input_json=json.dumps(arguments, ensure_ascii=False),
            output_json=result.model_dump_json(),
            error=result.error,
        )
        return ToolExecution(result=result, trace=trace)
