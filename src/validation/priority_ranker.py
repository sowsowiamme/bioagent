# src/validation/priority_ranker.py
"""
湿实验优先级排序器
输入: ProteinMPNN score + ESMFold pLDDT + 亲和力 + 免疫原性
输出: 湿实验送测优先级 (Top 3)
"""
from typing import List, Dict

class WetlabPriorityRanker:
    def __init__(self):
        # 权重设计（基于工业实践）
        self.weights = {
            "score": 0.3,      # 序列-结构兼容性
            "plddt": 0.3,      # 结构置信度
            "affinity": 0.25,  # 亲和力
            "immunogenicity": 0.15  # 免疫原性（负向）
        }
    
    def rank_designs(self, designs: List[Dict]) -> List[Dict]:
        """
        designs: [
            {
                "id": 1,
                "score": 0.87,
                "plddt": 88.2,
                "kd_nm": 8.2,
                "immuno_strong": 0,
                "immuno_weak": 2
            },
            ...
        ]
        """
        for d in designs:
            # 归一化各指标到0-1
            score_norm = 1.0 - min(d["score"] / 1.5, 1.0)  # score越低越好
            plddt_norm = min(d["plddt"] / 100, 1.0)        # pLDDT越高越好
            affinity_norm = min(50 / max(d["kd_nm"], 1), 1.0)  # KD越低越好
            immuno_penalty = 1.0 - min((d["immuno_strong"] * 2 + d["immuno_weak"]) / 10, 1.0)
            
            # 综合评分
            d["priority_score"] = (
                self.weights["score"] * score_norm +
                self.weights["plddt"] * plddt_norm +
                self.weights["affinity"] * affinity_norm +
                self.weights["immunogenicity"] * immuno_penalty
            )
        
        # 按优先级排序
        sorted_designs = sorted(designs, key=lambda x: x["priority_score"], reverse=True)
        
        # 添加湿实验建议
        for i, d in enumerate(sorted_designs):
            if i == 0:
                d["wetlab_recommendation"] = "✅ 首选送测"
            elif i < 3:
                d["wetlab_recommendation"] = "✅ 备选送测"
            else:
                d["wetlab_recommendation"] = "⚠️ 低优先级"
        
        return sorted_designs