# src/validation/affinity_predictor.py
"""
简化亲和力预测器：基于序列物理化学特征估算KD
原理: 抗体-抗原结合依赖:
  • 互补电荷 (静电相互作用)
  • 疏水补丁 (范德华力)
  • 关键残基保守性 (如CDR-H3)
"""
from typing import List, Dict
import numpy as np

class SimpleAffinityPredictor:
    def __init__(self):
        # 工业经验值：Fc-FcγR结合的关键特征
        self.key_residues = {
            "H268": "E",   # Glu268: 关键盐桥
            "H269": "L",   # Leu269 疏水接触
            "H309": "D",   # Asp309 电荷互补
        }
    

    def predict_kd(self, sequence: str, chain: str = "H") -> Dict:
        """
        预测KD (nM)
        返回: {"kd_nm": float, "confidence": "HIGH/MEDIUM/LOW", "rationale": str}
        """
        # Step 1: 提取重链序列（1HZH的H链约440残基）
        if len(sequence) > 800:  # 多链序列
            # 简化：取前440残基作为H链近似
            heavy_chain = sequence[:440]
        else:
            heavy_chain = sequence
        
        # Step 2: 关键残基保守性评分（0-1）
        conservation_score = self._check_key_residues(heavy_chain)
        
        # Step 3: 电荷互补性（FcγR结合区净电荷）
        binding_region = heavy_chain[260:310]  # FcγR结合区近似
        charge_score = self._calculate_charge_complementarity(binding_region)
        
        # Step 4: 综合评分 → KD预测
        # 经验公式: KD ≈ 50 * exp(-2.0 * (conservation + charge)/2)
        combined_score = (conservation_score + charge_score) / 2
        kd_nm = 50 * np.exp(-2.0 * combined_score)
        
        # 置信度
        confidence = "HIGH" if combined_score > 0.7 else "MEDIUM" if combined_score > 0.5 else "LOW"
        
        return {
            "kd_nm": round(kd_nm, 1),
            "confidence": confidence,
            "conservation_score": round(conservation_score, 2),
            "charge_score": round(charge_score, 2),
            "rationale": f"关键残基保守性{conservation_score:.0%} + 电荷互补性{charge_score:.0%}"
        }
    
    def _check_key_residues(self, seq: str) -> float:
        """检查关键残基保守性（简化版）"""
        # 1HZH关键残基位置（基于PDB 1HZH）
        key_positions = [267, 268, 308]  # 0-based索引
        conserved = 0
        for pos in key_positions:
            if pos < len(seq):
                aa = seq[pos]
                # 工业标准：Glu/Asp在268/309位置提升亲和力
                if pos == 267 and aa in "E": conserved += 1
                if pos == 268 and aa in "LIV": conserved += 1  # 疏水
                if pos == 308 and aa in "D": conserved += 1
        return conserved / len(key_positions)
    
    def _calculate_charge_complementarity(self, region: str) -> float:
        """计算结合区净电荷（FcγR带正电，Fc需负电互补）"""
        # 简化：负电荷残基比例（D/E）
        negative = sum(1 for aa in region if aa in "DE")
        positive = sum(1 for aa in region if aa in "KR")
        net_charge = (negative - positive) / len(region)
        # 归一化到0-1（理想净负电荷）
        return max(0, min(1, (net_charge + 0.3) / 0.6))  # 经验偏移

# 示例用法
if __name__ == "__main__":
    predictor = SimpleAffinityPredictor()
    seq = "GVSITQSADVYVKTGETVTISCTASGVDFTKHKVGWIREAPGKKPEWIGSVNPSDNSSTYNPDYKDRVTLTVDPSKNTAYLTISNLKAEDTATYYCYLSGPATEGLKPGDNDYPYVWGKGTYVEVSDTPVAKPNAYPLAPLPTPXXTSGGTAALGCLVKDYFPEPVTVXSWXXXXNSGALTSGXVHTFPAVLQSXSGLYSLSSVVTVPSSSLGTXXQXTYICNVNHKPSNTKVDKKXXAEPKSCXDXXKTHTCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVXXDGXXVEVHNAKTKPREEQYNXXSTYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTIXSKAKGXQPREPQVYTLPPSRDEXXLTKNQVSLTCLVKGFYPSDIAVXXEWESXNGXXQPENNYKTTPPVLDSXDXXGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"
    result = predictor.predict_kd(seq)
    print(f"KD预测: {result['kd_nm']} nM ({result['confidence']})")
    print(f"依据: {result['rationale']}")