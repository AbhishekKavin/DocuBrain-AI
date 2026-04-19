import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.engine.rag import RAGEngine
from app.engine.ingestion import IngestionEngine
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

logger = logging.getLogger("MainAPI")

app = FastAPI(title="DocuBrain AI API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
rag_engine_instance = None

# Define Request/Response schema
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.on_event("startup")
async def startup_event():
    """
    Automates the ingestion check before the API starts accepting requests.
    This ensures the 'Production' system is always up to date.
    """
    global rag_engine_instance
    try:
        logger.info("Checking new documents in /data")
        #Trigger Ingestion
        ingestor = IngestionEngine(data_path="data/")
        ingestor.process_documents()

        #Initialize RAG Engine
        rag_engine_instance = RAGEngine()

        logger.info("Startup complete. API is ready to accept requests.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")

# Prediction endpoint
@app.post("/ask-stream", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Primary endpoint for querying the document intelligence system.
    """
    if rag_engine_instance is None:
        raise HTTPException(status_code=503, detail="RAG Engine not initialized yet. Please try again later.")
    
    return StreamingResponse(
        rag_engine_instance.stream_query(request.question),
        media_type="text/event-stream"
    )

@app.get("/health")
def health_check():
    """
    Standard health check endpoint for monitoring (MLOps)[cite: 78].
    """
    return {
        "status": "healthy", 
        "engine_loaded": rag_engine_instance is not None,
        }