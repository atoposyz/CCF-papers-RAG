"""
构建本地向量数据库脚本 (Vector DB Setup)
读取 cleaned_papers.jsonl，使用 sentence-transformers 和 ChromaDB 构建向量数据库。
"""
import os
import json
import logging
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_vectordb(cleaned_file: str, db_path: str):
    setup_logging()
    
    if not os.path.exists(cleaned_file):
        logging.error(f"未找到清洗后的数据文件: {cleaned_file}")
        logging.error("请先运行 src/clean_data.py 生成所需文件。")
        return
    
    logging.info("正在加载 SentenceTransformer 模型 (all-MiniLM-L6-v2) ...")
    # 选择轻量级、速度快的文本嵌入模型
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    logging.info(f"正在初始化 ChromaDB，路径: {db_path} ...")
    # 创建基于本地路径的持久化 Chroma 客户端
    client = chromadb.PersistentClient(path=db_path)
    collection_name = "ccf_papers"
    
    # 如果集合已经存在，则将其清除以防混用旧数据，再重新创建
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = client.create_collection(name=collection_name)
    
    # 用于批量插入的数据结构
    documents = []
    embeddings = []
    metadatas = []
    ids = []
    batch_size = 500  # 定义批量大小
    
    # 读取原始数据并处理
    logging.info("开始读取 cleaned_papers.jsonl 数据并进行向量化和入库...")
    with open(cleaned_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for idx, line in enumerate(tqdm(lines, desc="向量化进度")):
        if not line.strip():
            continue
            
        try:
            paper = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        # 提取相关字段
        search_content = paper.get('search_content', '')
        if not search_content.strip():
            continue
            
        title = paper.get('title', 'Unknown Title')
        year = paper.get('year', '')
        venue_abbr = paper.get('venue_abbr', 'unknown')
        abstract = paper.get('abstract', '')
        
        authors = paper.get('authors', [])
        first_author = authors[0] if authors else "Unknown"
        
        # 年份必须转换为整数或字符串类型用于精确过滤 (支持大于、等号)
        if year and str(year).isdigit():
            year_int = int(year)
        else:
            year_int = 0
            
        doc_id = f"doc_{idx}"
        
        # 根据需求定义 Metadata
        # 注意: ChromaDB元数据字典仅支持 string, int, float, bool 类型
        metadata = {
            "title": title[:200], # 防止过长
            "year": year_int,
            "venue_abbr": venue_abbr,
            "first_author": first_author,
            # 将摘要截断一部分存储，方便查询结果中的快速展示
            "abstract_snippet": abstract[:500]
        }
        
        documents.append(search_content)
        metadatas.append(metadata)
        ids.append(doc_id)
        
        # 满足批次大小后统一嵌入与插入，提升效率
        if len(documents) >= batch_size:
            # 获取 embeddings (使用 CPU/GPU 取决于系统环境)
            batch_embeddings = model.encode(documents, show_progress_bar=False).tolist()
            # 存入到 Chroma 集成
            collection.add(
                documents=documents,
                embeddings=batch_embeddings,
                metadatas=metadatas,
                ids=ids
            )
            # 清空缓存
            documents, embeddings, metadatas, ids = [], [], [], []

    # 将剩余的数据提交入库
    if documents:
        batch_embeddings = model.encode(documents, show_progress_bar=False).tolist()
        collection.add(
            documents=documents,
            embeddings=batch_embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
    logging.info(f"向量数据库构建完成！成功存入 {len(lines)} 条记录。")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    db_dir = os.path.join(project_root, "paper_db")
    
    cleaned_file = os.path.join(db_dir, "cleaned_papers.jsonl")
    chroma_db_path = os.path.join(db_dir, "chroma_db")
    
    build_vectordb(cleaned_file, chroma_db_path)
