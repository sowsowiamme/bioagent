"""
验证指标计算：pLDDT, TM-score, RMSD等
"""
from typing import List
import numpy as np

def calculate_plddt(b_factors: List[float]) -> float:
    """计算平均pLDDT"""
    return np.mean(b_factors) if b_factors else 0.0


def assess_foldability(plddt: float) -> str:
    """
    评估可折叠性
    
    Returns:
        "HIGH" | "MEDIUM" | "LOW" | "UNFOLDED"
    """
    if plddt >= 90:
        return "HIGH"
    elif plddt >= 70:
        return "MEDIUM"
    elif plddt >= 50:
        return "LOW"
    else:
        return "UNFOLDED"

def calculate_tm_score(plddt_values: List[float]) -> float:
    """
    估算TM-score（简化版，基于pLDDT分布）
    注: 精确TM-score需结构比对，此处为快速估算
    """
    # 经验公式: TM-score ≈ 0.2 + 0.8 * (pLDDT>70的比例)
    high_confidence_ratio = sum(1 for p in plddt_values if p > 70) / len(plddt_values)
    return 0.2 + 0.8 * high_confidence_ratio

def generate_validation_report(results: List[dict]) -> str:
    """生成Markdown格式验证报告"""
    report = "# ESMFold验证报告\n\n"
    report += "| Design | pLDDT | 可折叠性 | 状态 |\n"
    report += "|--------|-------|----------|------|\n"
    


    for r in results:
        if not r["success"]:
            status = "❌ 失败"
            plddt_str = "N/A"
            foldability = "N/A"
        else:
            plddt = r["plddt"]
            foldability = assess_foldability(plddt)
            status = "✅ 通过" if plddt > 80 else "⚠️ 边界"
            plddt_str = f"{plddt:.1f}"
        
        report += f"| {r['design_id']} | {plddt_str} | {foldability} | {status} |\n"
    
    # 添加总结
    passed = sum(1 for r in results if r["success"] and r["plddt"] > 80)
    total = len(results)
    report += f"\n## 总结\n- 通过率: {passed}/{total} ({passed/total*100:.0f}%)\n"
    report += "- 工业标准: pLDDT > 80 为可实验候选\n"
    
    return report