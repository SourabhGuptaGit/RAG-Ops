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
from data_loader import load_and_chunk_pdf, embed_texts


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
    def _load(context: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = context.event.data["pdf_path"]
        source_id = context.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)
    
    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vectors = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, name=f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vectors, payloads)
        return RAGUpsertResult(ingested=len(chunks))
    
    chunks_and_src = await context.step.run(
        step_id="Load-and-Chunk", handler=lambda: _load(context=context), output_type=RAGChunkAndSrc
    )
    ingested = await context.step.run(
        step_id="Embed-and-Upsert", handler=lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult
    )
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf")
)
async def rag_query_pdf_ai(context: inngest.Context) -> RAGSearchResult:
    def _search(query: str, top_k: int = 5):
        query_vec = embed_texts([query])[0]
        results = QdrantStorage().search(query_vector=query_vec, top_k=top_k)
        return results
    
    question = context.event.data.get("question")
    top_k = context.event.data.get("top_k")
    top_k = int(top_k) if str(top_k).isdigit() else 5
    
    results = await context.step.run(
        step_id="Embed-and-Search", handler=lambda: _search(question, top_k), output_type=RAGSearchResult
    )
    
    context_block = "\n\n".join(f"- {c}" for c in results.contexts)
    user_content = (
        f"Use the following context to anwser the question.\n"
        f"Context:\n{context_block}\n"
        f"Question:\n{question}\n"
        f"Anwser concisevly using the conext above."
    )
    
    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPANAI_API_KEY"),
        model="gpt-4o-mini"
    )
    
    response = await context.step.ai.infer(
        step_id="LLM-Response",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages":[
                {"role": "system", "context": "You are an Q/A agent, anwser to questions using only the provided context."},
                {"role": "user", "context": user_content}
            ]
        }
    )
    anwser = response.get("choices")[0].get("message").get("context").strip()
    
    return {"anwser": anwser, "context": results.sources, "total_contexts": len(results.contexts)}


app = FastAPI()

inngest.fast_api.serve(app=app, client=inngest_client, functions=[rag_ingest_pdf, rag_query_pdf_ai])
