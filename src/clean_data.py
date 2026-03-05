"""
数据清洗与预处理脚本 (Data Cleaning)
提取 paper_db 下的所有 jsonl 文件，进行清洗和预处理，然后保存为 cleaned_papers.jsonl。
"""
import os
import json
import logging
from tqdm import tqdm

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_all_jsonl(db_dir: str):
    """递归查找目录下的所有 .jsonl 文件，排除特定的输出文件"""
    exclude_files = {"missing_abstracts.jsonl", "cleaned_papers.jsonl"}
    for root, _, files in os.walk(db_dir):
        for f in sorted(files):
            if f.endswith(".jsonl") and f not in exclude_files:
                yield os.path.join(root, f)

def clean_data(db_dir: str, output_file: str):
    setup_logging()
    
    # 获取所有需要处理的原始数据文件
    jsonl_files = list(find_all_jsonl(db_dir))
    logging.info(f"查找到 {len(jsonl_files)} 个原始数据文件，开始数据清洗...")
    
    cleaned_count = 0
    with open(output_file, 'w', encoding='utf-8') as out_f:
        # 使用 tqdm 显示处理进度
        for file_path in tqdm(jsonl_files, desc="清洗文件进度"):
            with open(file_path, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                        
                    try:
                        paper = json.loads(line)
                    except json.JSONDecodeError:
                        logging.warning(f"跳过无法解析的 JSON 行: {line[:50]}...")
                        continue
                    
                    # 1. 丢弃无用字段：删除 keywords 字段
                    if 'keywords' in paper:
                        del paper['keywords']
                    
                    # 2. 处理缺失值：如果 abstract 为空或缺失，将 title 作为备用内容补充进去
                    abstract = paper.get('abstract', '').strip()
                    title = paper.get('title', '').strip()
                    if not abstract:
                        abstract = title
                        paper['abstract'] = abstract # 更新原始字段
                    
                    # 3. 构建检索目标内容：新增 search_content 字段，值为 title + " " + abstract
                    paper['search_content'] = f"{title} {abstract}"
                    
                    # 将清洗后的 JSON 写入新文件
                    out_f.write(json.dumps(paper, ensure_ascii=False) + '\n')
                    cleaned_count += 1

    logging.info(f"数据清洗完成。共清洗并保存了 {cleaned_count} 条数据至 {output_file}。")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    db_dir = os.path.join(project_root, "paper_db")
    output_file = os.path.join(db_dir, "cleaned_papers.jsonl")
    
    # 确保 paper_db 存在
    if not os.path.exists(db_dir):
        logging.error(f"未找到数据库目录: {db_dir}")
        exit(1)
        
    clean_data(db_dir, output_file)
