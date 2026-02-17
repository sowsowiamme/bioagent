"""
ProteinMPNN序列设计封装器
职责: 
  • 解析ProteinMPNN输出(.fa文件)
  • 提取关键指标(score/seq_recovery)
  • 与验证模块对接
"""
from typing import List, Dict
from Bio import SeqIO
import re

class ProteinMPNNDesign:
    """ProteinMPNN设计结果解析器"""
    
    def __init__(self, fasta_path: str):
        self.fasta_path = fasta_path
        self.records = list(SeqIO.parse(fasta_path, "fasta"))
        self.original_sequence = None
        self.designs = []
        self._parse()
    
    def _parse(self):
        """解析FASTA文件"""
        # 第1条记录 = 原始序列
        if len(self.records) > 0:
            self.original_sequence = str(self.records[0].seq)
        
        # 后续记录 = 生成的设计
        for i, rec in enumerate(self.records[1:], 1):
            desc = rec.description
            
            # 提取指标（正则解析）
            score_match = re.search(r"score=([\d\.]+)", desc)
            recovery_match = re.search(r"seq_recovery=([\d\.]+)", desc)
            
            design = {
                "id": i,
                "sequence": str(rec.seq),
                "score": float(score_match.group(1)) if score_match else None,
                "seq_recovery": float(recovery_match.group(1)) if recovery_match else None,
                "description": desc
            }
            self.designs.append(design)
    
    def get_top_designs(self, n: int = 3) -> List[Dict]:
        """按score排序，返回Top N设计"""
        valid_designs = [d for d in self.designs if d["score"] is not None]
        sorted_designs = sorted(valid_designs, key=lambda x: x["score"])
        return sorted_designs[:n]
    
    def get_sequences_for_validation(self) -> List[str]:
        """获取序列列表（供ESMFold验证）"""
        return [d["sequence"] for d in self.designs]
    
    
    def generate_design_report(self) -> str:
        """生成设计质量报告"""
        report = "# ProteinMPNN设计报告\n\n"
        report += "## 原始序列基准\n"
        report += f"- Score: {self._extract_original_score()}\n"
        report += f"- 长度: {len(self.original_sequence)} 残基\n\n"
        
        report += "## 生成设计\n"
        report += "| Design | Score | Seq Recovery | 改进 |\n"
        report += "|--------|-------|--------------|------|\n"
        
        orig_score = self._extract_original_score()
        for d in self.designs[:5]:  # Top 5
            improvement = ((orig_score - d["score"]) / orig_score * 100) if orig_score else 0
            report += f"| {d['id']} | {d['score']:.4f} | {d['seq_recovery']:.2%} | ↓{improvement:.1f}% |\n"
        
        return report
            
    def _extract_original_score(self) -> float:
        """从原始序列描述提取score"""
        if len(self.records) > 0:
            desc = self.records[0].description
            match = re.search(r"score=([\d\.]+)", desc)
            return float(match.group(1)) if match else 1.5
        return 1.5