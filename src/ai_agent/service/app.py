import typing

import fastapi
import modern_di
import modern_di_fastapi

from ai_agent.service import dependencies, routes


def create_application(di_container: modern_di.Container | None = None) -> fastapi.FastAPI:
    application: typing.Final = fastapi.FastAPI(
        title="Network Equipment Consultant API",
        version="0.1.0",
    )
    application.include_router(routes.router)
    actual_container: typing.Final = di_container or dependencies.create_container()
    modern_di_fastapi.setup_di(application, actual_container)
    actual_container.validate()
    return application


app: typing.Final = create_application()
