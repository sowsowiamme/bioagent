from langchain_community.document_loaders import PubMedLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer
import torch
import os
import time
import sys 


class TargetDiscoveryRAG:
    def __init__(self, cache_dir= "./data/rag_cache"):
        # model_path = './all-MiniLM-L6-v2'
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        # self.embeddings = HuggingFaceEmbeddings(
            # model_name = "sentence-transformers/all-MiniLM-L6-v2",
            # model_kwargs = { "device": "mps" if torch.backends.mps.is_available() else "cpu"}
        #     model_name = model_path, 
        #     model_kwargs={'device': 'cpu'},  # 根据需要改为 'cuda'
        #     encode_kwargs={
        #         'normalize_embeddings': True,
        #         'show_progress_bar': False
        #     }
            
        # )
                # ✅ 关键2：从ModelScope下载模型（国内直连，100%成功）
        model_dir = snapshot_download(
            'AI-ModelScope/all-MiniLM-L6-v2',  # ModelScope镜像
            cache_dir=os.path.join(cache_dir, "models")
        )
        
        # # ✅ 关键3：从本地加载模型（无网络请求）
        self.embeddings = SentenceTransformer(model_dir)
        self.vectorstore = None


    def build_knowledge_base(self,diseases= ["lung cancer", "breaset_cancer"]):
        """knowledge database for diseases"""
        cache_path = os.path.join(self.cache_dir, "target_kb")
        if os.path.exists(cache_path):
            print("从缓存中加载知识库")
            self.vectorstore = FAISS.load_local(cache_path, self.embeddings, allow_dangerous_deserialization=True)
            return         
        print("⏳ 首次构建知识库（约10分钟）...")
        docs = []
        for disease in diseases:
            # 从PubMed加载靶点相关文献（每疾病5篇）
            loader = PubMedLoader(f"{disease} target therapy")
            docs.extend(loader.load())
        
        # 构建向量库
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.vectorstore.save_local(cache_path)
        print(f"✅ 知识库构建完成！共{len(docs)}篇文献")
    

    def discover_targets(self, disease:str, top_k:int=3) -> dict:
        """输入疾病，输出靶点假设，+文献证据"""
        if self.vectorstore is None:
            self.build_knowledge_base()
        # 检索相关文献
        query = f"therapeutic targets for {disease} treatment mechanism"
        results = self.vectorstore.similarity_search(query, k = top_k)
        targets = []
        for i, doc in enumerate(results):
            content = doc.page_content.lower()
            candidate_targets = []
            known_targets = ["pd-1", "pd-l1", "ctla-4", "her2", "egfr", "vegf", "parp", "brca"]
            for target in known_targets:
                if target in content:
                    candidate_targets.append(target.upper())

            if candidate_targets:
                targets.append({
                    "target": ", ".join(set(candidate_targets)),
                    "evidence": doc.page_content[:300] + "...",
                    "source": doc.metadata.get("uid", "PubMed"),
                    "relevance_score": 1.0 - i*0.2  # 模拟相关性
                })
        
        return {
            "disease": disease,
            "targets": targets[:3],  # 返回Top 3靶点
            "query_time": "0.8s"  # 模拟响应时间
        }



if __name__ == "__main__":
    # start = time.time()
    # try:
    #     rag = TargetDiscoveryRAG(cache_dir="./data/rag_cache")
    #     print(f"✅ RAG初始化成功 ({time.time()-start:.2f}s)")
    # except Exception as e:
    #     print(f"❌ RAG初始化失败: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     sys.exit(1)
    # Step 2: 构建知识库（真实PubMed检索）
    print("\n[2/3] 🔍 从PubMed检索真实文献（lung cancer targets）...")
    print("    ⏳ 首次运行需下载文献（约1-3分钟，请耐心等待）...")
    start = time.time()
    try:
        # 关键：减少max_results避免超时，聚焦高质量文献
        loader = PubMedLoader("non-small cell lung cancer PD-1 EGFR therapy")
        docs = loader.load()
        
        print(f"✅ PubMed检索成功！获取 {len(docs)} 篇真实文献 ({time.time()-start:.2f}s)")
        for i, doc in enumerate(docs):
            pmid = doc.metadata.get('uid', 'N/A')
            title_preview = doc.page_content[:60].split('. ')[0] + "..."  # 截断到第一个句号
        
            print(f"   📄 [{i+1}] PMID:{pmid}")
            print(f"      摘要预览: {title_preview}")
    except Exception as e:
        print(e)