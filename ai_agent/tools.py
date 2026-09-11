import json
import typing

import constants
import pydantic

PERMISSIONS_TOOL_NAME: typing.Final = "get_engineer_permissions"
ERROR_STATUS: typing.Final = "error"
INEXISTENT_ENGINEER_ERROR: typing.Final = "Инженер с таким ID не найден в системе прав."


class PermissionCheckInput(pydantic.BaseModel):
    """Входные данные для проверки прав инженера."""

    engineer_id: int = pydantic.Field(
        strict=True,
        description="Числовой идентификатор инженера в системе прав доступа.",
    )


def get_engineer_permissions(args: PermissionCheckInput) -> str:
    """Проверить права инженера в корпоративной системе по его числовому ID.

    Вызывай этот инструмент, когда пользователь просит проверить доступы или
    права на серверы и указывает ID инженера. Не используй инструмент для
    вопросов, не связанных с правами доступа.

    Возвращает описание актуальных прав найденного инженера. Если инженер не
    найден или проверка завершилась ошибкой, возвращает JSON-строку со статусом
    ``error`` и понятным сообщением вместо возбуждения исключения.
    """
    try:
        # Имитация обращения к корпоративной системе доступов (остается неизменной)
        eid: typing.Final = args.engineer_id

        if eid == constants.EXISTING_ENGINEER_ID:
            return (
                f"Доступы инженера #{constants.EXISTING_ENGINEER_ID}: "
                f"[{constants.PRODENV_PERMISSION_R}, "
                f"{constants.TESTENV_PERMISSION_RW}]"
            )

        raise ValueError(INEXISTENT_ENGINEER_ERROR)

    # Tool является границей рантайма: по условию задания наружу не должны
    # выходить исключения реализации.
    except Exception as e:  # noqa: BLE001
        return json.dumps(
            {"status": ERROR_STATUS, "message": str(e)},
            ensure_ascii=False,
        )
