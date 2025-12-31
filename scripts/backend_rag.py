from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import os
import requests
import numpy as np
import uuid
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
HF_URL = f"https://router.huggingface.co/v1/chat/completions"


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

# In-memory conversation storage (in production, use Redis or a database)
conversation_history: Dict[str, List[Dict[str, str]]] = {}

class ChatRequest(BaseModel):
    query: str
    chat_id: Optional[str] = None  # Add chat_id to request
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
    chat_id: str  # Add chat_id to response

PROMPT_TEMPLATE = """You are a helpful, respectful and honest medical assistant. Use the provided context to answer the user's question. You can also refer to previous conversation history above.

Context:
---
{context}
---

Question: {question}

Instructions:
- If the context contains relevant information, provide a clear and helpful answer based on that information.
- If the context does not contain enough information, briefly state what information is missing.
- Be concise but comprehensive.

Answer (cite sources when possible):"""

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
    # Generate chat_id if not provided
    chat_id = request.chat_id or str(uuid.uuid4())
    
    # Get or initialize conversation history for this chat
    if chat_id not in conversation_history:
        conversation_history[chat_id] = []
    
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

    # 7. Send prompt to Hugging Face Chat Completions API
    try:
        # Build messages array with conversation history
        messages = [
            {
                "role": "system",
                "content": "You are a helpful, respectful and honest medical assistant. You have access to both retrieved context from a knowledge base and conversation history. Use the conversation history to understand what the user previously asked, and use the retrieved context to provide accurate medical information, but DO NOT mention, summarize, or display the conversation history in your response. You can reference previous questions and answers in the conversation."
            }
        ]
        
        # Add conversation history (last 10 messages to avoid token limits)
        # This includes previous user questions and assistant answers
        for msg in conversation_history[chat_id][-10:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current query with context
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        llm_response = requests.post(
            HF_URL,
            headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": HUGGINGFACE_MODEL,
                "messages": messages,
                "stream": False,
                "temperature": 0.1,
                "max_tokens": 512  # Increased to allow longer responses with conversation context
            },
            timeout=60
        )
        llm_response.raise_for_status()
        response_data = llm_response.json()
        llm_answer = response_data["choices"][0]["message"]["content"].strip()
        
        # Update conversation history
        conversation_history[chat_id].append({
            "role": "user",
            "content": request.query
        })
        conversation_history[chat_id].append({
            "role": "assistant",
            "content": llm_answer
        })
            
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
    return ChatResponse(query=request.query, llm_answer=llm_answer, results=results, chat_id=chat_id)

# Add endpoint to create new chat
@app.post("/chat/new")
def new_chat():
    """Create a new chat session"""
    chat_id = str(uuid.uuid4())
    conversation_history[chat_id] = []
    return {"chat_id": chat_id}

# Add endpoint to clear chat history
@app.delete("/chat/{chat_id}")
def clear_chat(chat_id: str):
    """Clear conversation history for a chat"""
    if chat_id in conversation_history:
        del conversation_history[chat_id]
    return {"status": "cleared", "chat_id": chat_id}

# Add endpoint to get all queries and answers for a chat
@app.get("/chat/{chat_id}")
def get_chat_history(chat_id: str):
    """Get all queries and llm_answers for a specific chat_id"""
    if chat_id not in conversation_history:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Extract queries and answers from conversation history
    queries_and_answers = []
    history = conversation_history[chat_id]
    
    # Pair up user queries with assistant answers
    i = 0
    while i < len(history):
        if history[i]["role"] == "user":
            query = history[i]["content"]
            # Look for the corresponding assistant answer
            if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
                llm_answer = history[i + 1]["content"]
                queries_and_answers.append({
                    "query": query,
                    "llm_answer": llm_answer
                })
                i += 2
            else:
                # Query without answer yet
                queries_and_answers.append({
                    "query": query,
                    "llm_answer": None
                })
                i += 1
        else:
            i += 1
    
    return {
        "chat_id": chat_id,
        "queries_and_answers": queries_and_answers
    }

@app.get("/")
def root():
    return {"message": "Medical RAG Backend is running."} 
