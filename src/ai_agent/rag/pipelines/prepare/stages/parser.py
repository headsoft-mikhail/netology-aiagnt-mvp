import abc
import json
import logging
import typing
from typing import Any

from bs4 import BeautifulSoup

from ai_agent.rag.pipelines import models

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


class AbstractDocumentParser(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def parse(cls, document: models.SourceDocument) -> models.ParsedDocument:
        raise NotImplementedError


class TXTParser(AbstractDocumentParser):
    @classmethod
    def parse(cls, document: models.SourceDocument) -> models.ParsedDocument:
        return models.ParsedDocument(
            text=document.content,
            source=document.source,
            file_type=document.file_type,
        )


class JSONParser(AbstractDocumentParser):
    @classmethod
    def parse(cls, document: models.SourceDocument) -> models.ParsedDocument:
        data: typing.Final = json.loads(document.content)

        text: typing.Final = cls._json_to_text(data)

        return models.ParsedDocument(
            text=text,
            source=document.source,
            file_type=document.file_type,
        )

    @classmethod
    def _json_to_text(cls, data: Any, prefix: str = "") -> str:
        lines: typing.Final[list[str]] = []

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.extend(cls._json_to_text(value, prefix=f"{prefix}  ").splitlines())
                else:
                    lines.append(f"{prefix}{key}: {value}")

        elif isinstance(data, list):
            for index, item in enumerate(data, start=1):
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}Item {index}:")
                    lines.extend(cls._json_to_text(item, prefix=f"{prefix}  ").splitlines())
                else:
                    lines.append(f"{prefix}{item}")

        else:
            lines.append(f"{prefix}{data}")

        return "\n".join(lines)


class HTMLParser(AbstractDocumentParser):
    @classmethod
    def parse(cls, document: models.SourceDocument) -> models.ParsedDocument:
        soup: typing.Final = BeautifulSoup(document.content, "html.parser")

        sections: typing.Final = [heading.get_text(" ", strip=True) for heading in soup.find_all(["h1", "h2", "h3"])]

        for element in soup(["head", "script", "style", "nav", "footer"]):
            element.decompose()

        text: typing.Final = soup.get_text(separator="\n")

        return models.ParsedDocument(
            text=text,
            source=document.source,
            file_type=document.file_type,
            sections=sections,
        )


def parse_document(document: models.SourceDocument) -> models.ParsedDocument:
    LOGGER_OBJ.info(f"Parsing {document.source}...")

    if document.file_type == "txt":
        return TXTParser.parse(document)

    if document.file_type == "json":
        return JSONParser.parse(document)

    if document.file_type == "html":
        return HTMLParser.parse(document)

    raise ValueError(f"Unsupported file type: {document.file_type}")
