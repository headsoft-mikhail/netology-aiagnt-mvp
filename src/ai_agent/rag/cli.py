import argparse
import shutil
import typing

from ai_agent.rag.pipelines import config


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

    pipeline_config: typing.Final = config.load_pipeline_config()

    if args.command == "prepare":
        from ai_agent.rag.pipelines.prepare.pipeline import RAGPreparePipeline

        pipeline = RAGPreparePipeline(pipeline_config)
    elif args.command == "chunk":
        from ai_agent.rag.pipelines.chunk.pipeline import RAGChunkPipeline

        pipeline = RAGChunkPipeline(pipeline_config)
    elif args.command == "embedding":
        from ai_agent.rag.pipelines.embeddings.pipeline import RAGEmbeddingsPipeline

        pipeline = RAGEmbeddingsPipeline(pipeline_config)
    elif args.command == "vector_store":
        from ai_agent.rag.pipelines.vector_store.pipeline import RAGVectorStorePipeline

        pipeline = RAGVectorStorePipeline(pipeline_config)
    elif args.command == "clear":
        for directory in (
            pipeline_config.paths.prepared,
            pipeline_config.paths.chunks,
            pipeline_config.paths.embeddings,
            pipeline_config.paths.vector_store,
        ):
            if directory.exists():
                shutil.rmtree(directory)
        return
    else:
        raise ValueError(f"Unknown command: {args.command}")

    pipeline.run()
