import typing

TEST_USER_ID: typing.Final = "tech_user_1"
TEST_USER_LANGUAGE: typing.Final = "Python"
TEST_USER_PROJECT: typing.Final = "Альфа"
TEST_USER_LEVEL: typing.Final = "Senior"
TEST_USER_FACTS: typing.Final = (
    f"Стек {TEST_USER_LANGUAGE}, Проект {TEST_USER_PROJECT}, Уровень {TEST_USER_LEVEL}"
)

EXISTING_ENGINEER_ID: typing.Final = 505
PRODENV_PERMISSION_R: typing.Final = "Production-сервер: Чтение"
TESTENV_PERMISSION_RW: typing.Final = "Тестовый контур: Полный доступ"
INEXISTENT_ENGINEER_ID: typing.Final = 999
