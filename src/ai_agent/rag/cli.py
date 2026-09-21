import argparse
import shutil
import typing

from ai_agent.rag.pipelines.config import rag_pipeline_config


def main() -> None:
    parser: typing.Final = argparse.ArgumentParser(
        description="RAG data preparation pipeline",
    )
    subparsers: typing.Final = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Prepare raw documents")
    subparsers.add_parser("chunk", help="Split prepared documents into chunks")
    subparsers.add_parser("embedding", help="Prepare embeddings")
    subparsers.add_parser("vector_store", help="Prepare vector store")
    subparsers.add_parser("clear", help="Remove generated RAG data")

    args: typing.Final = parser.parse_args()

    if args.command == "prepare":
        from ai_agent.rag.pipelines.prepare.pipeline import RAGPreparePipeline

        pipeline = RAGPreparePipeline(rag_pipeline_config)
    elif args.command == "chunk":
        from ai_agent.rag.pipelines.chunk.pipeline import RAGChunkPipeline

        pipeline = RAGChunkPipeline(rag_pipeline_config)
    elif args.command == "embedding":
        from ai_agent.rag.pipelines.embeddings.pipeline import RAGEmbeddingsPipeline

        pipeline = RAGEmbeddingsPipeline(rag_pipeline_config)
    elif args.command == "vector_store":
        from ai_agent.rag.pipelines.vector_store.pipeline import RAGVectorStorePipeline

        pipeline = RAGVectorStorePipeline(rag_pipeline_config)
    elif args.command == "clear":
        for directory in (
            rag_pipeline_config.paths.prepared,
            rag_pipeline_config.paths.chunks,
            rag_pipeline_config.paths.embeddings,
            rag_pipeline_config.paths.vector_store,
        ):
            if directory.exists():
                shutil.rmtree(directory)
        return
    else:
        raise ValueError(f"Unknown command: {args.command}")

    pipeline.run()
