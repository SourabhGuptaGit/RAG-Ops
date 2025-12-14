from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import logging
import datetime

from data_loader import load_and_chunk_pdf
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult


load_dotenv()
inngest_client = inngest.Inngest(
    app_id="RAG-Ops",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(context: inngest.Context):
    return {"status": "Welcome!"}


app = FastAPI()

inngest.fast_api.serve(app=app, client=inngest_client, functions=[rag_ingest_pdf])
