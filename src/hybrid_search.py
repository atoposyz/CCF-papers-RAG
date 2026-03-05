"""
混合查询函数和测试逻辑 (Hybrid Search Function & Testing)
实现真正的混合检索：向量语义检索 + BM25 关键字检索，并使用 RRF 融合排名。
"""
import os
import re
import logging
import chromadb
from sentence_transformers import SentenceTransformer


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def _tokenize(text: str) -> list:
    """简单分词：转小写，按非字母数字字符切分"""
    return re.findall(r'[a-z0-9]+', text.lower())


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion 单项分数"""
    return 1.0 / (k + rank)


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

        # BM25 相关属性（会在 _build_bm25_index 中被赋值）
        self._bm25 = None
        self._bm25_ids: list = []
        self._bm25_docs: list = []
        self._bm25_metas: list = []

        # 构建 BM25 索引（从 ChromaDB 加载所有文档）
        logging.info("正在构建 BM25 关键字索引（首次加载），请稍候...")
        self._build_bm25_index()
        logging.info("初始化完成！\n")

    def _build_bm25_index(self):
        """从 ChromaDB 中拉取所有文档，构建 BM25 倒排索引。"""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logging.warning(
                "未找到 rank_bm25 库，BM25 关键字检索将被禁用。"
                "请运行: uv add rank-bm25"
            )
            self._bm25 = None
            self._bm25_ids = []
            self._bm25_metas = []
            self._bm25_docs = []
            return

        # ChromaDB 分页获取所有记录（防止一次性拉取过多占内存）
        total = self.collection.count()
        batch_size = 5000
        all_ids = []
        all_docs = []
        all_metas = []

        for offset in range(0, total, batch_size):
            batch = self.collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"]
            )
            all_ids.extend(batch["ids"])
            all_docs.extend(batch["documents"])
            all_metas.extend(batch["metadatas"])

        self._bm25_ids = all_ids
        self._bm25_docs = all_docs
        self._bm25_metas = all_metas

        # 对每篇文档分词（使用 document 字段，即 search_content）
        tokenized_corpus = [_tokenize(doc) for doc in all_docs]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logging.info(f"BM25 索引构建完成，共 {total} 篇文档。")

    def _vector_search(self, query: str, where_clause, n_results: int):
        """纯向量检索，返回 {doc_id: {...}} 字典"""
        query_embedding = self.model.encode(query, show_progress_bar=False).tolist()
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["metadatas", "distances", "documents"]
        }
        if where_clause is not None:
            kwargs["where"] = where_clause

        results = self.collection.query(**kwargs)
        ranked = {}
        if results.get("ids") and results["ids"][0]:
            for rank, (doc_id, meta, dist, doc) in enumerate(zip(
                results["ids"][0],
                results["metadatas"][0],
                results["distances"][0],
                results["documents"][0]
            )):
                ranked[doc_id] = {
                    "vector_rank": rank + 1,
                    "vector_dist": dist,
                    "meta": meta,
                    "doc": doc
                }
        return ranked

    def _bm25_search(self, query: str, where_clause, n_results: int):
        """BM25 关键字检索，考虑元数据过滤，返回 {doc_id: {...}} 字典"""
        if self._bm25 is None:
            return {}

        # 提取过滤条件
        filter_year = None
        filter_venues = None
        if where_clause:
            if "$and" in where_clause:
                for cond in where_clause["$and"]:
                    if "year" in cond:
                        filter_year = cond["year"]
                    if "venue_abbr" in cond:
                        v_cond = cond["venue_abbr"]
                        if isinstance(v_cond, dict) and "$in" in v_cond:
                            filter_venues = v_cond["$in"]
                        else:
                            filter_venues = [v_cond]
            elif "year" in where_clause:
                filter_year = where_clause["year"]
            elif "venue_abbr" in where_clause:
                v_cond = where_clause["venue_abbr"]
                if isinstance(v_cond, dict) and "$in" in v_cond:
                    filter_venues = v_cond["$in"]
                else:
                    filter_venues = [v_cond]

        query_tokens = _tokenize(query)
        bm25_scores = self._bm25.get_scores(query_tokens)

        # 构建 (score, idx) 列表，并按过滤条件过滤
        candidates = []
        for idx, score in enumerate(bm25_scores):
            meta = self._bm25_metas[idx]
            if filter_year is not None and meta.get("year") != filter_year:
                continue
            if filter_venues is not None:
                if meta.get("venue_abbr", "").lower() not in filter_venues:
                    continue
            candidates.append((score, idx))

        # 按分数降序排列，只取 n_results 候选
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = candidates[:n_results]

        ranked = {}
        for rank, (score, idx) in enumerate(top_candidates):
            doc_id = self._bm25_ids[idx]
            ranked[doc_id] = {
                "bm25_rank": rank + 1,
                "bm25_score": score,
                "meta": self._bm25_metas[idx],
                "doc": self._bm25_docs[idx]
            }
        return ranked

    def search_hybrid(
        self,
        query: str,
        target_year: str = None,
        target_venue: list = None,
        top_k: int = 5,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        candidate_k: int = None
    ) -> list:
        """
        真正的混合检索: 向量检索 + BM25，通过 RRF 融合结果。

        :param query: 自然语言查询
        :param target_year: 年份过滤
        :param target_venue: 会议过滤列表 (如 ['dac', 'icse'])
        :param top_k: 最终返回数量
        :param vector_weight: 向量检索 RRF 权重 (0~1)
        :param bm25_weight:   BM25 检索 RRF 权重 (0~1)
        :param candidate_k: 每路候选数量，默认 max(top_k * 5, 50)
        :return: 结构化结果列表
        """
        if candidate_k is None:
            candidate_k = max(top_k * 5, 50)

        # 构造元数据过滤条件
        conditions = []
        if target_year and str(target_year).isdigit():
            conditions.append({"year": int(target_year)})
        if target_venue:
            if isinstance(target_venue, list):
                if len(target_venue) == 1:
                    conditions.append({"venue_abbr": target_venue[0].lower()})
                else:
                    conditions.append({"venue_abbr": {"$in": [v.lower() for v in target_venue]}})
            else: # Fallback for single string if not passed as list
                conditions.append({"venue_abbr": target_venue.lower()})

        where_clause = None
        if len(conditions) == 1:
            where_clause = conditions[0]
        elif len(conditions) > 1:
            where_clause = {"$and": conditions}

        # 两路检索
        vector_results = self._vector_search(query, where_clause, candidate_k)
        bm25_results = self._bm25_search(query, where_clause, candidate_k)

        # RRF 融合
        all_ids = set(vector_results.keys()) | set(bm25_results.keys())
        fused = {}
        for doc_id in all_ids:
            v_score = 0.0
            b_score = 0.0
            if doc_id in vector_results:
                v_score = _rrf_score(vector_results[doc_id]["vector_rank"]) * vector_weight
            if doc_id in bm25_results:
                b_score = _rrf_score(bm25_results[doc_id]["bm25_rank"]) * bm25_weight

            # 获取元数据（优先从向量结果取，否则从 BM25 结果取）
            src = vector_results.get(doc_id) or bm25_results.get(doc_id)
            meta = src["meta"]
            doc = src["doc"]

            fused[doc_id] = {
                "hybrid_score": v_score + b_score,
                "vector_dist": vector_results[doc_id]["vector_dist"] if doc_id in vector_results else None,
                "bm25_score": bm25_results[doc_id]["bm25_score"] if doc_id in bm25_results else 0.0,
                "meta": meta,
                "doc": doc
            }

        # 按混合分数降序排列，取前 top_k
        sorted_results = sorted(fused.values(), key=lambda x: x["hybrid_score"], reverse=True)[:top_k]
        return sorted_results

    def search_papers(self, query: str, target_year: str = None, target_venue: str = None, top_k: int = 5):
        """
        命令行友好的格式化检索函数（调用混合检索后格式化输出）。

        :param query: 用户的自然语言查询
        :param target_year: 目标年份 (如 "2024")
        :param target_venue: 目标会议简称 (如 "dac")
        :param top_k: 返回结果的最大数量
        :return: 格式化的文本结果字符串
        """
        results = self.search_hybrid(query, target_year, target_venue, top_k)

        output_lines = []
        filter_status = [
            f"Year: {target_year if target_year else 'Any'}",
            f"Venue: {target_venue if target_venue else 'Any'}"
        ]
        output_lines.append(f"🔍 检索查询 (Query): '{query}'")
        output_lines.append(f"📌 检索过滤 (Filters): {', '.join(filter_status)}")
        output_lines.append("-" * 70)

        if not results:
            output_lines.append("未找到符合条件的结果。可能是过滤条件太严格或数据库为空。")
            return "\n".join(output_lines)

        for idx, item in enumerate(results):
            meta = item["meta"]
            hybrid_score = item["hybrid_score"]
            vector_dist = item["vector_dist"]
            bm25_score = item["bm25_score"]

            title = meta.get('title', 'Unknown Title')
            year = meta.get('year', 'Unknown')
            venue_abbr = meta.get('venue_abbr', 'Unknown').upper()
            first_author = meta.get('first_author', 'Unknown')
            abstract_snippet = meta.get('abstract_snippet', '')

            if not abstract_snippet.strip():
                abstract_snippet = item["doc"].replace(title, '').strip()[:100] + "..."

            dist_str = f"{vector_dist:.4f}" if vector_dist is not None else "N/A"
            output_lines.append(
                f"[ Top {idx+1} ] HybridScore: {hybrid_score:.5f}  "
                f"(VecDist: {dist_str}  BM25: {bm25_score:.2f})"
            )
            output_lines.append(f"📖 标题: {title}")
            output_lines.append(f"📅 年份: {year}")
            output_lines.append(f"🏢 会议: {venue_abbr}")
            output_lines.append(f"🧑‍🔬 作者 (一作): {first_author}")
            output_lines.append(f"📝 摘要片段: {abstract_snippet.replace(chr(10), ' ')}")
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
    parser.add_argument("--vector-weight", type=float, default=0.6, help="Vector search weight (0~1)")
    parser.add_argument("--bm25-weight", type=float, default=0.4, help="BM25 search weight (0~1)")
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
        print("欢迎使用本地学术论文检索系统！（混合检索模式）")
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
            try:
                print(res)
            except UnicodeEncodeError:
                print(res.encode('gbk', 'replace').decode('gbk'))
    elif args.query:
        print("\n正在检索中...\n")
        res = searcher.search_papers(
            query=args.query,
            target_venue=args.venue,
            target_year=args.year,
            top_k=args.top_k
        )
        try:
            print(res)
        except UnicodeEncodeError:
            print(res.encode('gbk', 'replace').decode('gbk'))
    else:
        parser.print_help()
        print("\n提示: 你可以使用 -i 进入交互式搜索模式，例如: uv run python src/hybrid_search.py -i")
