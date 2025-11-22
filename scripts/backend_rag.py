from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import os
import requests
import numpy as np
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Config
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = int(os.getenv("QDRANT_PORT"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBED_MODEL = os.getenv("EMBED_MODEL")
TOP_K = int(os.getenv("TOP_K"))
LLM_SERVER_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL")  # Default Mistral model
HF_URL = os.getenv("HF_URL")


# MMR Configuration
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA"))  # Balance between relevance (0.7) and diversity (0.3)
# Higher lambda (0.8-0.9) = more relevant, less diverse
# Lower lambda (0.5-0.6) = more diverse, less relevant
MMR_FETCH_MULTIPLIER = int(os.getenv("MMR_FETCH_MULTIPLIER"))  # Fetch 3x more candidates for MMR selection

# FastAPI app
app = FastAPI(title="Medical RAG Backend")

# CORS Middleware
origins_str = os.getenv("CORS_ORIGINS", "")
if origins_str:
    # Split, strip whitespace, and remove trailing slashes
    origins = [origin.strip().rstrip('/') for origin in origins_str.split(",") if origin.strip()]
else:
    origins = [] # Default Vite dev server port

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",  # Allow all Vercel domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model and Qdrant client at startup
print("Loading embedding model...")
model = SentenceTransformer(EMBED_MODEL)
print("Connecting to Qdrant...")
if QDRANT_HOST and (QDRANT_HOST.startswith("http://") or QDRANT_HOST.startswith("https://")):
    # Qdrant Cloud - use URL parameter
    if QDRANT_API_KEY:
        client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
    else:
        client = QdrantClient(url=QDRANT_HOST)
else:
    # Local Qdrant - use host and port
    client = QdrantClient(host="localhost", port=QDRANT_PORT)
print("Startup complete.")

class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = TOP_K

class DocResult(BaseModel):
    id: str
    text: str
    score: float
    metadata: dict

class ChatResponse(BaseModel):
    query: str
    llm_answer: str
    results: List[DocResult]

PROMPT_TEMPLATE = """
You are a helpful, respectful and honest medical assistant. Answer the user's question based ONLY on the provided context.
If the context does not contain enough information to answer the question, state that you cannot answer based on the provided information.
Do not use any prior knowledge.

Context:
---
{context}
---

Question: {question}

Answer:
"""

def mmr_rerank(query_vec, candidates, top_k, lambda_param=MMR_LAMBDA):
    """
    Maximum Marginal Relevance reranking
    
    Args:
        query_vec: Query embedding vector
        candidates: List of candidate documents with embeddings and metadata
        top_k: Number of documents to return
        lambda_param: Trade-off between relevance (λ) and diversity (1-λ)
    
    Returns:
        List of selected documents ordered by MMR score
    """
    if len(candidates) <= top_k:
        return candidates
    
    selected = []
    remaining = candidates.copy()
    
    # Convert query_vec to numpy array if needed
    query_vec = np.array(query_vec).reshape(1, -1)
    
    # First document: highest relevance
    relevance_scores = []
    for candidate in remaining:
        doc_vec = np.array(candidate['embedding']).reshape(1, -1)
        # Cosine similarity (higher = more similar)
        query_norm = np.linalg.norm(query_vec)
        doc_norm = np.linalg.norm(doc_vec)
        if query_norm > 0 and doc_norm > 0:
            similarity = np.dot(query_vec, doc_vec.T) / (query_norm * doc_norm)
            relevance_scores.append(float(similarity[0, 0]))
        else:
            relevance_scores.append(0.0)
    
    # Select first document (highest relevance)
    first_idx = np.argmax(relevance_scores)
    selected.append(remaining.pop(first_idx))
    relevance_scores.pop(first_idx)
    
    # Iteratively select remaining documents using MMR
    while len(selected) < top_k and remaining:
        mmr_scores = []
        
        for i, candidate in enumerate(remaining):
            # Relevance to query
            doc_vec = np.array(candidate['embedding']).reshape(1, -1)
            query_norm = np.linalg.norm(query_vec)
            doc_norm = np.linalg.norm(doc_vec)
            if query_norm > 0 and doc_norm > 0:
                relevance = float(np.dot(query_vec, doc_vec.T)[0, 0] / (query_norm * doc_norm))
            else:
                relevance = 0.0
            
            # Maximum similarity to already selected documents
            max_similarity = 0.0
            for selected_doc in selected:
                selected_vec = np.array(selected_doc['embedding']).reshape(1, -1)
                selected_norm = np.linalg.norm(selected_vec)
                if doc_norm > 0 and selected_norm > 0:
                    similarity = float(np.dot(doc_vec, selected_vec.T)[0, 0] / (doc_norm * selected_norm))
                    max_similarity = max(max_similarity, similarity)
            
            # MMR score: λ * relevance - (1-λ) * max_similarity
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            mmr_scores.append(mmr_score)
        
        # Select document with highest MMR score
        if mmr_scores:
            best_idx = np.argmax(mmr_scores)
            selected.append(remaining.pop(best_idx))
        else:
            break
    
    return selected

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # 1. Embed the query
    query_vec = model.encode(request.query, convert_to_numpy=True)
    
    # 2. Fetch more candidates for MMR (fetch more, then rerank)
    fetch_count = request.top_k * MMR_FETCH_MULTIPLIER
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec.tolist(),
        limit=fetch_count,  # Fetch more candidates
        with_payload=True
    ).points
    
    # 3. Prepare candidates with embeddings for MMR
    candidates = []
    for hit in search_result:
        text = hit.payload.get("text", "") if hasattr(hit, 'payload') and hit.payload else ""
        if text:
            # Re-embed the text for MMR calculation
            doc_embedding = model.encode(text, convert_to_numpy=True)
            candidates.append({
                'hit': hit,
                'embedding': doc_embedding.tolist(),
                'text': text,
                'score': hit.score if hasattr(hit, 'score') else 0.0
            })
    
    # 4. Apply MMR reranking
    if len(candidates) > request.top_k:
        mmr_selected = mmr_rerank(
            query_vec.tolist(),
            candidates,
            request.top_k,
            lambda_param=MMR_LAMBDA
        )
        # Extract hits from MMR-selected candidates
        search_result = [item['hit'] for item in mmr_selected]
    else:
        # Not enough candidates, use all
        search_result = [item['hit'] for item in candidates]
    
    # 5. Format context for the LLM
    context = "\n\n---\n\n".join([
        hit.payload.get("text", "") if hasattr(hit, 'payload') and hit.payload else ""
        for hit in search_result
    ])
    
    # 6. Create the prompt
    prompt = PROMPT_TEMPLATE.format(context=context, question=request.query)

    # 7. Send prompt to the local LLM
    try:
        llm_response = requests.post(
            HF_URL,
            headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}", "Content-Type": "application/json"},
            json={
                "inputs": prompt,
                "parameters": {
                    "temperature": 0.1,
                    "max_new_tokens": 256,
                    "return_full_text": False
                }
            },
            timeout=30
        )
        llm_response.raise_for_status()
        response_data = llm_response.json()
        if isinstance(response_data, list) and len(response_data) > 0:
            llm_answer = response_data[0].get("generated_text", "").strip()
        else:
            llm_answer = str(response_data).strip()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"LLM server is unavailable: {e}")

    # 8. Format and return the final response
    results = [
        DocResult(
            id=str(hit.id),
            text=hit.payload.get("text", "") if hasattr(hit, 'payload') and hit.payload else "",
            score=hit.score if hasattr(hit, 'score') else 0.0,
            metadata=hit.payload if hasattr(hit, 'payload') and hit.payload else {}
        )
        for hit in search_result
    ]
    return ChatResponse(query=request.query, llm_answer=llm_answer, results=results)

@app.get("/test-ollama")
def test_ollama():
    """Test Ollama connection"""
    try:
        test_response = requests.post(
            LLM_SERVER_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": "Say hello in one word",
                "stream": False
            },
            timeout=30
        )
        test_response.raise_for_status()
        return {
            "status": "success",
            "ollama_response": test_response.json(),
            "model": OLLAMA_MODEL,
            "url": LLM_SERVER_URL
        }
    except Exception as e:
        import traceback
        error_info = {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "model": OLLAMA_MODEL,
            "url": LLM_SERVER_URL
        }
        if hasattr(e, 'response') and e.response is not None:
            error_info["response_status"] = e.response.status_code
            error_info["response_text"] = e.response.text
        error_info["traceback"] = traceback.format_exc()
        return error_info

@app.get("/")
def root():
    return {"message": "Medical RAG Backend is running."} 