import numpy as np
import pandas as pd
from mhcflurry import Class1AffinityPredictor
import time
import re

    


class MHCflurryPredictor:
    def __init__(self, alleles=None):
        """
        initialize MHCflurry predictor
        :param alleles: the potential alleles，HLA ttypes
        """
        if alleles is None:
            # common HLA-A and HLA-B alleles
            self.alleles = [
                "HLA-A02:01"
            ]
        else:
            self.alleles = alleles
        
        # 加载预测器（第一次运行会自动下载模型）
        print("Load MHCflurry Predictor...")
        self.predictor = Class1AffinityPredictor.load()
        print("✅ MHCflurry Loading finished")
    
    def predict_peptides(self, sequence, peptide_lengths=[9]):
        """
        to split the sequence into different length of peptides, and predict the IC50(nM) of each peptide
        return a DataFrame which contains peptides, alleles, and IC50
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
    

    def predict(self, sequence):
        """
        对给定序列进行全面免疫原性评估：
        - 使用预设的多个常见等位基因
        - 扫描所有肽段长度 (8-11)
        """
        df = self.predict_peptides(sequence)
        return self.aggregate_immunogenicity(df)