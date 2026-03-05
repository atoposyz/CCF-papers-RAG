"""
混合查询函数和测试逻辑 (Hybrid Search Function & Testing)
封装查询函数 search_papers，并编写测试代码验证检索效果。
"""
import os
import logging
import chromadb
from sentence_transformers import SentenceTransformer

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class HybridSearcher:
    def __init__(self, db_path: str):
        self.db_path = db_path
        logging.info("正在加载 SentenceTransformer 模型 (all-MiniLM-L6-v2) ...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logging.info(f"正在连接 ChromaDB... 路径: {db_path}")
        self.client = chromadb.PersistentClient(path=self.db_path)
        try:
            self.collection = self.client.get_collection(name="ccf_papers")
        except ValueError:
            logging.error("未找到集合 'ccf_papers'，请先运行 src/build_vectordb.py 构建索引。")
            raise
            
        logging.info("初始化完成！\n")

    def search_papers(self, query: str, target_year: str = None, target_venue: str = None, top_k: int = 5):
        """
        检索函数: Hybrid Search 混合检索。
        先使用 ChromaDB 的元数据 (Metadata) 进行精准过滤，再使用向量模型进行相似度搜索。
        
        :param query: 用户的自然语言查询
        :param target_year: 目标年份 (如 "2024")。内部会转成整数用于过滤。
        :param target_venue: 目标会议简称 (如 "dac")
        :param top_k: 返回结果的最大数量
        :return: 格式化的文本结果字符串
        """
        # 1. 对 query 进行向量化嵌入 (Embedding)
        query_embedding = self.model.encode(query, show_progress_bar=False).tolist()
        
        # 2. 构造元数据过滤条件 (Metadata Filtering)
        conditions = []
        if target_year and str(target_year).isdigit():
            conditions.append({"year": int(target_year)})
        if target_venue:
            conditions.append({"venue_abbr": target_venue.lower()})  # 转为小写以防大小写不一致
            
        where_clause = None
        if len(conditions) == 1:
            where_clause = conditions[0]
        elif len(conditions) > 1:
            where_clause = {"$and": conditions}

        # 3. 混合查询核心层
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k
        }
        if where_clause is not None:
            kwargs["where"] = where_clause
            
        # 检索 top_k 高维相关的文档集合以及距离分数 (通常是 L2/Cosine)
        results = self.collection.query(**kwargs)
        
        # 4. 格式化输出最终数据
        output_lines = []
        filter_status = [
            f"Year: {target_year if target_year else 'Any'}",
            f"Venue: {target_venue if target_venue else 'Any'}"
        ]
        
        output_lines.append(f"🔍 检索查询 (Query): '{query}'")
        output_lines.append(f"📌 检索过滤 (Filters): {', '.join(filter_status)}")
        output_lines.append("-" * 70)
        
        # 处理异常: 没查出数据或集合是空的
        if not results.get('documents') or not results['documents'][0]:
            output_lines.append("未找到符合条件的结果。可能是过滤条件太严格或数据库为空。")
            return "\n".join(output_lines)
            
        # 依次打印前 top_k 条
        for idx in range(len(results['documents'][0])):
            meta = results['metadatas'][0][idx]
            dist = results['distances'][0][idx]
            
            title = meta.get('title', 'Unknown Title')
            year = meta.get('year', 'Unknown')
            venue_abbr = meta.get('venue_abbr', 'Unknown').upper()
            first_author = meta.get('first_author', 'Unknown')
            abstract_snippet = meta.get('abstract_snippet', '')
            
            # 由于 abstract_snippet 可能存在丢失，如果没有存 metadata 中，可以取 content 中切割
            if not abstract_snippet.strip():
                content = results['documents'][0][idx]
                abstract_snippet = content.replace(title, '').strip()[:100] + "..."
                
            output_lines.append(f"[ Top {idx+1} ] (距离分数 Dist: {dist:.4f})")
            output_lines.append(f"📖 标题: {title}")
            output_lines.append(f"📅 年份: {year}")
            output_lines.append(f"🏢 会议: {venue_abbr}")
            output_lines.append(f"🧑‍🔬 作者 (一作): {first_author}")
            output_lines.append(f"📝 摘要片段: {abstract_snippet.replace('\n', ' ')}")
            output_lines.append("-" * 70)
            
        return "\n".join(output_lines)

if __name__ == "__main__":
    setup_logging()
    
    import argparse
    parser = argparse.ArgumentParser(description="CCF Papers Hybrid Search")
    parser.add_argument("--query", "-q", type=str, help="Search query")
    parser.add_argument("--venue", "-v", type=str, help="Target venue abbreviation (e.g., dac)")
    parser.add_argument("--year", "-y", type=str, help="Target year")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    db_dir = os.path.join(project_root, "paper_db")
    chroma_db_path = os.path.join(db_dir, "chroma_db")
    
    if not os.path.exists(chroma_db_path):
        logging.error(f"未找到 ChromaDB 目录: {chroma_db_path}，请先运行 src/build_vectordb.py")
        exit(1)
        
    searcher = HybridSearcher(db_path=chroma_db_path)
    
    if args.interactive:
        print("\n" + "=" * 50)
        print("欢迎使用本地学术论文检索系统！")
        print("输入 'exit' 或 'quit' 退出。")
        print("=" * 50)
        while True:
            query = input("\n🔍 请输入检索词 (Query): ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            if not query:
                continue
            
            venue = input("🏢 目标会议/期刊 (如 dac，直接回车跳过): ").strip() or None
            year = input("📅 目标年份 (如 2024，直接回车跳过): ").strip() or None
            
            print("\n正在检索中...\n")
            res = searcher.search_papers(query=query, target_venue=venue, target_year=year, top_k=args.top_k)
            # 兼容 Windows CMD 编码打印报错
            try:
                print(res)
            except UnicodeEncodeError:
                print(res.encode('gbk', 'replace').decode('gbk'))
    elif args.query:
        print("\n正在检索中...\n")
        res = searcher.search_papers(query=args.query, target_venue=args.venue, target_year=args.year, top_k=args.top_k)
        try:
            print(res)
        except UnicodeEncodeError:
            print(res.encode('gbk', 'replace').decode('gbk'))
    else:
        parser.print_help()
        print("\n提示: 你可以使用 -i 进入交互式搜索模式，例如: uv run python src/hybrid_search.py -i")
