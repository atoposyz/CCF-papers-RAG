"""
API 后端服务器 (FastAPI + OpenAI API / RAG)
提供基于构建好的基于 ChromaDB 的混合检索服务的 RESTful API。
同时提供 /chat 接口以给用户进行问答。
"""
import os
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Union
from openai import AsyncOpenAI
from hybrid_search import HybridSearcher

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="CCF Papers Hybrid Search & RAG API")

# 获取项目目录并指向静态文件夹与数据库目录
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
project_root = os.path.dirname(current_dir)
db_dir = os.path.join(project_root, "paper_db")
chroma_db_path = os.path.join(db_dir, "chroma_db")

# 全局存储搜索器实例
searcher = None

@app.on_event("startup")
async def startup_event():
    global searcher
    try:
        logging.info("Initializing Hybrid Searcher...")
        searcher = HybridSearcher(db_path=chroma_db_path)
    except Exception as e:
        logging.error(f"Failed to initialize searcher: {e}")

# ================================
# 数据模型定义
# ================================
class SearchRequest(BaseModel):
    query: str
    venue: Optional[Union[List[str], str]] = None
    year: Optional[str] = None
    top_k: int = 5
    vector_weight: float = 0.7
    bm25_weight: float = 0.3

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    context_papers: str  # 拼接好的相关论文上下文
    model: str = "deepseek-chat"  # 允许前端选择模型
    api_key: str  # 用户的 API Key，不保存在后端
    base_url: str = "https://api.deepseek.com/v1"  # 支持兼容 OpenAI 接口规范的模型

# ================================
# API 路由
# ================================
@app.post("/api/search")
async def api_search(request: SearchRequest):
    if not searcher:
        raise HTTPException(
            status_code=500,
            detail="Search engine not initialized. Please ensure the vector database is built."
        )

    try:
        # 处理 venue，如果前端发来逗号分隔字符串也可以兼容（虽然 Pydantic 会优先尝试转 List）
        venues = request.venue
        if isinstance(venues, str):
            venues = [v.strip() for v in venues.split(',') if v.strip()]
        
        raw_results = searcher.search_hybrid(
            query=request.query,
            target_year=request.year,
            target_venue=venues,
            top_k=request.top_k,
            vector_weight=request.vector_weight,
            bm25_weight=request.bm25_weight
        )

        formatted_results = []
        for idx, item in enumerate(raw_results):
            meta = item["meta"]
            title = meta.get('title', 'Unknown Title')
            year = meta.get('year', 'Unknown')
            venue_abbr = meta.get('venue_abbr', 'Unknown').upper()
            first_author = meta.get('first_author', 'Unknown')
            abstract_snippet = meta.get('abstract_snippet', '')
            doi_url = meta.get('doi_url', '')
            dblp_url = meta.get('dblp_url', '')

            if not abstract_snippet.strip():
                content = item["doc"]
                abstract_snippet = content.replace(title, '').strip()[:200] + "..."

            vector_dist = item.get("vector_dist")
            bm25_score = item.get("bm25_score", 0.0)
            hybrid_score = item.get("hybrid_score", 0.0)

            formatted_results.append({
                "id": idx + 1,
                "title": title,
                "year": year,
                "venue": venue_abbr,
                "author": first_author,
                "abstract": abstract_snippet,
                "hybrid_score": round(hybrid_score, 6),
                "vector_dist": round(vector_dist, 4) if vector_dist is not None else None,
                "bm25_score": round(bm25_score, 2),
                "doi_url": doi_url,
                "dblp_url": dblp_url
            })

        return {"results": formatted_results}

    except Exception as e:
        logging.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def api_chat(request: ChatRequest):
    if not request.api_key:
        raise HTTPException(status_code=401, detail="API Key is required for Chat functionality.")

    try:
        # 实例化基于本次请求的 OpenAI Async 客户端
        client = AsyncOpenAI(
            api_key=request.api_key,
            base_url=request.base_url
        )

        # 组装 RAG Prompt 注入上下文
        system_prompt = f"""你是一个专业的学术论文检索和阅读助手 (AI RAG Assistant)。
用户当前正在浏览以下学术论文信息，请基于这些论文内容，专业、严谨且有逻辑地回答用户的问题。
如果用户的提问超出了以下提供的论文范围，你可以结合自身知识回答，但必须标明哪些内容来源于检索到的论文。回答时尽量使用 Markdown 排版。

[检出的论文上下文]
{request.context_papers}
"""
        # 将 system prompt 放至队首
        messages_for_llm = [{"role": "system", "content": system_prompt}]
        for msg in request.messages:
            messages_for_llm.append({"role": msg.role, "content": msg.content})

        # 创建流式响应生成器
        async def stream_generator():
            try:
                stream = await client.chat.completions.create(
                    model=request.model,
                    messages=messages_for_llm,
                    stream=True,
                    temperature=0.7,
                    max_tokens=2048
                )
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        # 将数据按照 Server-Sent Events (SSE) 格式返回
                        yield f"data: {json.dumps({'content': content})}\n\n"

                yield "data: [DONE]\n\n"
            except Exception as e:
                logging.error(f"LLM Stream Error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    except Exception as e:
        logging.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ================================
# 静态文件托管
# ================================
# 确保目录存在
os.makedirs(static_dir, exist_ok=True)
# 挂载前端页面
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=57999, reload=True)
