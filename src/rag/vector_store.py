# target_discovery/vector_store.py
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Tuple

print(faiss.__version__)

class TargetVectorStore:
    """基于FAISS的向量存储，用于语义搜索"""
    
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        print(f"⏳ 加载嵌入模型: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None 
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, documents: List[Dict]):
        """将文档添加至向量库"""
        texts = []
        for doc in documents:
            text = f"{doc.get('title', '')} {doc.get('abstract', '')}"
            texts.append(text)
            self.documents.append(doc)

    def add_documents(self, documents: List[Dict]):
        """将文档添加到向量库"""
        texts = []
        for doc in documents:
            # merge the title of the doc, and the abstract of the doc
            text = f"{doc.get('title', '')} {doc.get('abstract', '')}"
            texts.append(text)
            self.documents.append(doc)
        
        # 生成嵌入
        print(f"⏳ 为 {len(texts)} 篇文献生成嵌入...")
        batch_size = 32
        all_embeddings = []        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings = self.model.encode(batch, normalize_embeddings=True)
            all_embeddings.append(embeddings)
            
        self.embeddings = np.vstack(all_embeddings)
        
        # 构建FAISS索引
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # 内积相似度（归一化后等价于余弦相似度）
        self.index.add(self.embeddings.astype(np.float32))
        
        print(f"✅ 向量库构建完成，共 {len(self.documents)} 篇文献")
    

    def search(self, query: str, top_k: int=10):
        if self.index is None or len(self.documents) ==0:
            return []
        query_embedding = self.model.encode([query], normalize_embeddings=True)

        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """语义搜索相关文献"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        # 查询向量化
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # 搜索
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # return results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
                
        return results
    
    
    def save(self, path: str):
        """保存向量库到磁盘"""
        os.makedirs(path, exist_ok=True)
        
        # 保存FAISS索引
        if self.index is not None:
            faiss.write_index(self.index, f"{path}/index.faiss")
        
        # 保存文档和嵌入
        with open(f"{path}/documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)
        
        if len(self.embeddings) > 0:
            np.save(f"{path}/embeddings.npy", self.embeddings)
            
        print(f"✅ 向量库已保存到 {path}")
    
    def load(self, path: str):
        """从磁盘加载向量库"""
        # 加载FAISS索引
        index_path = f"{path}/index.faiss"
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        
        # 加载文档
        with open(f"{path}/documents.pkl", "rb") as f:
            self.documents = pickle.load(f)
        
        # 加载嵌入
        embeddings_path = f"{path}/embeddings.npy"
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
            
        print(f"✅ 从 {path} 加载向量库完成，共 {len(self.documents)} 篇文献")


# test_vector_store.py

def main():
    print("🚀 测试 FAISS 向量存储...")
    
    # 初始化向量库
    store = TargetVectorStore(model_name="BAAI/bge-small-en-v1.5")
    
    # 模拟一些文献数据（实际应用中这些应来自 PubMed）
    mock_docs = [
        {
            "title": "PD-1 blockade in breast cancer",
            "abstract": "PD-1 is a key immune checkpoint. Anti-PD-1 antibodies have shown efficacy in triple-negative breast cancer.",
            "year": "2023",
            "pmid": "12345678"
        },
        {
            "title": "HER2-targeted antibody-drug conjugates",
            "abstract": "Trastuzumab deruxtecan is a novel HER2-directed antibody-drug conjugate with promising activity in breast cancer.",
            "year": "2022",
            "pmid": "23456789"
        },
        {
            "title": "CTLA-4 immunotherapy in solid tumors",
            "abstract": "CTLA-4 blockade with ipilimumab has been explored in melanoma and other cancers, including breast cancer.",
            "year": "2021",
            "pmid": "34567890"
        }
    ]
    
    print(f"\n📚 添加 {len(mock_docs)} 篇模拟文献到向量库...")
    store.add_documents(mock_docs)
    
    # 测试语义搜索
    queries = [
        "immune checkpoint inhibitors in breast cancer",
        "HER2 therapy",
        "CTLA-4 antibodies"
    ]
    
    for query in queries:
        print(f"\n🔍 搜索: '{query}'")
        results = store.search(query, top_k=2)
        
        if results:
            for doc, score in results:
                print(f"  - {doc['title']} (相似度: {score:.4f})")
        else:
            print("  无结果")
    
    # 测试保存与加载
    save_path = "./test_vector_store"
    print(f"\n💾 保存向量库到 {save_path}")
    store.save(save_path)
    
    print("📂 重新加载向量库")
    new_store = TargetVectorStore()
    new_store.load(save_path)
    
    # 验证加载后的搜索功能
    print("\n🔍 重新加载后搜索 'PD-1'")
    results = new_store.search("PD-1", top_k=2)
    for doc, score in results:
        print(f"  - {doc['title']} (相似度: {score:.4f})")
    
    print("\n✅ FAISS 向量存储测试完成")

if __name__ == "__main__":
    main()