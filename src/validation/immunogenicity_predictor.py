import numpy as np
import pandas as pd
from mhcflurry import Class1AffinityPredictor
import time
import re

class MHCflurryPredictor:
    def __init__(self, alleles=None):
        """
        初始化MHCflurry预测器
        :param alleles: 待预测的等位基因列表，默认使用常见HLA类型
        """
        if alleles is None:
            # 常见HLA-A和HLA-B等位基因（可根据需要调整）
            self.alleles = [
                "HLA-A01:01", "HLA-A02:01", "HLA-A03:01", "HLA-A11:01",
                "HLA-A24:02", "HLA-B07:02", "HLA-B08:01", "HLA-B15:01",
                "HLA-B35:01", "HLA-B40:01"
            ]
        else:
            self.alleles = alleles
        
        # 加载预测器（第一次运行会自动下载模型）
        print("⏳ 加载MHCflurry预测器...")
        self.predictor = Class1AffinityPredictor.load()
        print("✅ MHCflurry加载完成")
    
    def predict_peptides(self, sequence, peptide_lengths=[9, 10, 11]):
        """
        将蛋白序列切分为所有可能的肽段，并预测每个肽段的IC50 (nM)
        返回DataFrame，包含肽段、等位基因、IC50等信息
        """
        chains = sequence.split('/')
        peptides = []
        for chain in chains:
            # 清洗单条链（移除可能残留的其他非字母）
            # 2. 将所有非标准氨基酸（不是 ACDEFGHIKLMNPQRSTVWY 的字母）替换为 A
            #   这里包括 X, B, Z, J, O, U 等
            clean_chain = re.sub(r'[^A-Z]', '', chain.upper())

            standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
            clean_chain = ''.join(aa if aa in standard_aa else 'A' for aa in clean_chain)
            if len(clean_chain) < min(peptide_lengths):
                continue
            for length in peptide_lengths:
                for i in range(len(clean_chain) - length + 1):
                    peptides.append(clean_chain[i:i+length])
            if len(sequence) < min(peptide_lengths):
                return pd.DataFrame()
        
    
        
        # MHCflurry可以同时预测多个肽段和多个等位基因
        results = []
        for allele in self.alleles:
            try:
                # 关键修正：使用predictor.predict方法
                ic50 = self.predictor.predict(peptides, allele=allele)
                for pep, ic in zip(peptides, ic50):
                    results.append({
                        "peptide": pep,
                        "allele": allele,
                        "ic50": ic,
                        "length": len(pep)
                    })
            except Exception as e:
                print(f"⚠️ 等位基因 {allele} 预测失败: {e}")
                continue
        
        df = pd.DataFrame(results)
        return df
    
    def aggregate_immunogenicity(self, df):
        """
        根据肽段预测结果汇总免疫原性风险
        """
        if df.empty:
            return {
                "success": True,
                "strong_binders": 0,
                "weak_binders": 0,
                "safety": "SAFE",
                "details": [],
                "job_url": "",
                "error": ""
            }
        
        strong = df[df["ic50"] < 50]
        weak = df[(df["ic50"] >= 50) & (df["ic50"] < 500)]
        
        # 安全评级
        if len(strong) > 0:
            safety = "RISK"
        elif len(weak) > 5:
            safety = "CAUTION"
        else:
            safety = "SAFE"
        
        # 按等位基因分组，列出强结合肽段
        details = []
        if len(strong) > 0:
            for allele, group in strong.groupby("allele"):
                details.append({
                    "allele": allele,
                    "peptides": group["peptide"].tolist()[:5]  # 只显示前5个
                })
        
        return {
            "success": True,
            "strong_binders": len(strong),
            "weak_binders": len(weak),
            "safety": safety,
            "details": details,
            "job_url": "",
            "error": ""
        }
    
    def predict(self, sequence, allele=None, peptide_length=9):
        """
        兼容原接口的predict方法
        allele和peptide_length参数被忽略（我们使用多等位基因和多长度）
        """
        df = self.predict_peptides(sequence)
        result = self.aggregate_immunogenicity(df)
        return result