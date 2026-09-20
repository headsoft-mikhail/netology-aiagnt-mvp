import uvicorn

from ai_agent.service.config import api_config


def main() -> None:
    uvicorn.run(
        "ai_agent.service.app:app",
        host=api_config.host,
        port=api_config.port,
        log_level=api_config.log_level,
    )


if __name__ == "__main__":
    main()
