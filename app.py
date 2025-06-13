# api/index.py
import os
import json
import sqlite3
import numpy as np
import re
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import aiohttp
import asyncio
import logging
from fastapi.responses import JSONResponse
import traceback
from mangum import Mangum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SIMILARITY_THRESHOLD = 0.4
MAX_RESULTS = 10
MAX_CONTEXT_CHUNKS = 4
API_KEY = os.getenv("API_KEY")

# Models
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

# Initialize FastAPI app
app = FastAPI(
    title="RAG Query API", 
    description="API for querying the RAG knowledge base",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database in /tmp directory for serverless
DB_PATH = "/tmp/knowledge_base.db"

def init_database():
    """Initialize database with tables"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Create discourse_chunks table
        c.execute('''
        CREATE TABLE IF NOT EXISTS discourse_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER,
            topic_id INTEGER,
            topic_title TEXT,
            post_number INTEGER,
            author TEXT,
            created_at TEXT,
            likes INTEGER,
            chunk_index INTEGER,
            content TEXT,
            url TEXT,
            embedding BLOB
        )
        ''')
        
        # Create markdown_chunks table
        c.execute('''
        CREATE TABLE IF NOT EXISTS markdown_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_title TEXT,
            original_url TEXT,
            downloaded_at TEXT,
            chunk_index INTEGER,
            content TEXT,
            embedding BLOB
        )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

def get_db_connection():
    """Get database connection with error handling"""
    try:
        if not os.path.exists(DB_PATH):
            init_database()
        
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    try:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
            
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
            
        return dot_product / (norm_vec1 * norm_vec2)
    except Exception as e:
        logger.error(f"Error in cosine_similarity: {e}")
        return 0.0

async def get_embedding(text, max_retries=3):
    """Get embedding from OpenAI API through aipipe proxy"""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY environment variable not set")
    
    for retry in range(max_retries):
        try:
            url = "https://aipipe.org/openai/v1/embeddings"
            headers = {
                "Authorization": API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "model": "text-embedding-3-small",
                "input": text
            }
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["data"][0]["embedding"]
                    elif response.status == 429:
                        wait_time = 2 ** retry
                        logger.warning(f"Rate limit, waiting {wait_time}s before retry {retry+1}")
                        await asyncio.sleep(wait_time)
                    else:
                        error_text = await response.text()
                        logger.error(f"Embedding API error (status {response.status}): {error_text}")
                        if retry == max_retries - 1:
                            raise HTTPException(status_code=response.status, detail=error_text)
        except asyncio.TimeoutError:
            logger.error(f"Timeout on embedding request, retry {retry+1}")
            if retry == max_retries - 1:
                raise HTTPException(status_code=408, detail="Request timeout")
        except Exception as e:
            logger.error(f"Exception getting embedding (attempt {retry+1}): {e}")
            if retry == max_retries - 1:
                raise HTTPException(status_code=500, detail=str(e))
            await asyncio.sleep(2 ** retry)

async def find_similar_content(query_embedding, conn):
    """Find similar content in database"""
    try:
        cursor = conn.cursor()
        results = []
        
        # Search discourse chunks
        cursor.execute("""
        SELECT id, post_id, topic_id, topic_title, post_number, author, created_at, 
               likes, chunk_index, content, url, embedding 
        FROM discourse_chunks 
        WHERE embedding IS NOT NULL
        LIMIT 1000
        """)
        
        discourse_chunks = cursor.fetchall()
        
        for chunk in discourse_chunks:
            try:
                embedding = json.loads(chunk["embedding"])
                similarity = cosine_similarity(query_embedding, embedding)
                
                if similarity >= SIMILARITY_THRESHOLD:
                    url = chunk["url"]
                    if not url.startswith("http"):
                        url = f"https://discourse.onlinedegree.iitm.ac.in/t/{url}"
                    
                    results.append({
                        "source": "discourse",
                        "id": chunk["id"],
                        "post_id": chunk["post_id"],
                        "topic_id": chunk["topic_id"],
                        "title": chunk["topic_title"],
                        "url": url,
                        "content": chunk["content"],
                        "author": chunk["author"],
                        "created_at": chunk["created_at"],
                        "chunk_index": chunk["chunk_index"],
                        "similarity": float(similarity)
                    })
            except Exception as e:
                logger.error(f"Error processing discourse chunk {chunk['id']}: {e}")
        
        # Search markdown chunks
        cursor.execute("""
        SELECT id, doc_title, original_url, downloaded_at, chunk_index, content, embedding 
        FROM markdown_chunks 
        WHERE embedding IS NOT NULL
        LIMIT 1000
        """)
        
        markdown_chunks = cursor.fetchall()
        
        for chunk in markdown_chunks:
            try:
                embedding = json.loads(chunk["embedding"])
                similarity = cosine_similarity(query_embedding, embedding)
                
                if similarity >= SIMILARITY_THRESHOLD:
                    url = chunk["original_url"]
                    if not url or not url.startswith("http"):
                        url = f"https://docs.onlinedegree.iitm.ac.in/{chunk['doc_title']}"
                    
                    results.append({
                        "source": "markdown",
                        "id": chunk["id"],
                        "title": chunk["doc_title"],
                        "url": url,
                        "content": chunk["content"],
                        "chunk_index": chunk["chunk_index"],
                        "similarity": float(similarity)
                    })
            except Exception as e:
                logger.error(f"Error processing markdown chunk {chunk['id']}: {e}")
        
        # Sort by similarity and group results
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Group by source and limit chunks per source
        grouped_results = {}
        for result in results:
            key = f"{result['source']}_{result.get('post_id', result.get('title'))}"
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        final_results = []
        for chunks in grouped_results.values():
            chunks.sort(key=lambda x: x["similarity"], reverse=True)
            final_results.extend(chunks[:MAX_CONTEXT_CHUNKS])
        
        final_results.sort(key=lambda x: x["similarity"], reverse=True)
        return final_results[:MAX_RESULTS]
        
    except Exception as e:
        logger.error(f"Error in find_similar_content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_answer(question, relevant_results, max_retries=2):
    """Generate answer using LLM"""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY environment variable not set")
    
    for retry in range(max_retries):
        try:
            context = ""
            for result in relevant_results:
                source_type = "Discourse post" if result["source"] == "discourse" else "Documentation"
                context += f"\n\n{source_type} (URL: {result['url']}):\n{result['content'][:1500]}"
            
            prompt = f"""Answer the following question based ONLY on the provided context. 
            If you cannot answer the question based on the context, say "I don't have enough information to answer this question."
            
            Context:
            {context}
            
            Question: {question}
            
            Return your response in this exact format:
            1. A comprehensive yet concise answer
            2. A "Sources:" section that lists the URLs and relevant text snippets you used to answer
            
            Sources must be in this exact format:
            Sources:
            1. URL: [exact_url_1], Text: [brief quote or description]
            2. URL: [exact_url_2], Text: [brief quote or description]
            """
            
            url = "https://aipipe.org/openai/v1/chat/completions"
            headers = {
                "Authorization": API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that provides accurate answers based only on the provided context. Always include sources in your response with exact URLs."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }
            
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    elif response.status == 429:
                        wait_time = 3 * (retry + 1)
                        logger.warning(f"Rate limit, waiting {wait_time}s before retry {retry+1}")
                        await asyncio.sleep(wait_time)
                    else:
                        error_text = await response.text()
                        logger.error(f"LLM API error (status {response.status}): {error_text}")
                        if retry == max_retries - 1:
                            raise HTTPException(status_code=response.status, detail=error_text)
        except asyncio.TimeoutError:
            logger.error(f"Timeout on LLM request, retry {retry+1}")
            if retry == max_retries - 1:
                raise HTTPException(status_code=408, detail="Request timeout")
        except Exception as e:
            logger.error(f"Exception generating answer: {e}")
            if retry == max_retries - 1:
                raise HTTPException(status_code=500, detail=str(e))
            await asyncio.sleep(2)

def parse_llm_response(response):
    """Parse LLM response to extract answer and sources"""
    try:
        parts = response.split("Sources:", 1)
        
        if len(parts) == 1:
            for heading in ["Source:", "References:", "Reference:"]:
                if heading in response:
                    parts = response.split(heading, 1)
                    break
        
        answer = parts[0].strip()
        links = []
        
        if len(parts) > 1:
            sources_text = parts[1].strip()
            source_lines = sources_text.split("\n")
            
            for line in source_lines:
                line = line.strip()
                if not line:
                    continue
                    
                line = re.sub(r'^\d+\.\s*', '', line)
                line = re.sub(r'^-\s*', '', line)
                
                url_match = re.search(r'URL:\s*\[(.*?)\]|url:\s*\[(.*?)\]|\[(http[^\]]+)\]|URL:\s*(http\S+)|url:\s*(http\S+)|(http\S+)', line, re.IGNORECASE)
                text_match = re.search(r'Text:\s*\[(.*?)\]|text:\s*\[(.*?)\]|[""](.*?)[""]|Text:\s*"(.*?)"|text:\s*"(.*?)"', line, re.IGNORECASE)
                
                if url_match:
                    url = next((g for g in url_match.groups() if g), "").strip()
                    text = "Source reference"
                    
                    if text_match:
                        text_value = next((g for g in text_match.groups() if g), "")
                        if text_value:
                            text = text_value.strip()
                    
                    if url and url.startswith("http"):
                        links.append({"url": url, "text": text})
        
        return {"answer": answer, "links": links}
    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        return {
            "answer": "Error parsing the response from the language model.",
            "links": []
        }

# API Routes
@app.post("/api/")
async def query_knowledge_base(request: QueryRequest):
    """Main query endpoint"""
    try:
        logger.info(f"Received query: {request.question[:50]}...")
        
        if not API_KEY:
            return JSONResponse(
                status_code=500,
                content={"error": "API_KEY environment variable not set"}
            )
        
        # Get embedding for the question
        query_embedding = await get_embedding(request.question)
        
        # Find similar content
        conn = get_db_connection()
        try:
            relevant_results = await find_similar_content(query_embedding, conn)
            
            if not relevant_results:
                return {
                    "answer": "I couldn't find any relevant information in my knowledge base.",
                    "links": []
                }
            
            # Generate answer
            llm_response = await generate_answer(request.question, relevant_results)
            result = parse_llm_response(llm_response)
            
            # If no links extracted, create from results
            if not result["links"]:
                links = []
                unique_urls = set()
                
                for res in relevant_results[:5]:
                    url = res["url"]
                    if url not in unique_urls:
                        unique_urls.add(url)
                        snippet = res["content"][:100] + "..." if len(res["content"]) > 100 else res["content"]
                        links.append({"url": url, "text": snippet})
                
                result["links"] = links
            
            return result
            
        finally:
            conn.close()
            
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Basic health check
        return {
            "status": "healthy",
            "api_key_set": bool(API_KEY),
            "database_path": DB_PATH
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/api/")
async def root():
    """Root endpoint"""
    return {"message": "RAG Query API is running", "docs": "/api/docs"}

# Create the Mangum handler for Vercel
handler = Mangum(app)