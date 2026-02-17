"""
ESMFold API封装器 - 免GPU快速验证ProteinMPNN序列
特点: 
  • 无需本地GPU（调用Meta免费API）
  • 自动清理非标准氨基酸（X→移除）
  • 限流保护（避免429错误）
"""
import requests
import time
import os
from typing import Dict, List, Optional
from pathlib import Path

class ESMFoldValidator:
    """ESMFold结构预测验证器"""
    
    def __init__(self, api_url: str = "https://api.esmatlas.com/foldSequence/v1/pdb/"):
        self.api_url = api_url
        self.headers = {"Content-Type": "application/x-www-form-urlencoded"}
        self.last_request_time = 0
        self.min_interval = 30  # API限流：30秒/请求
    
    def _clean_sequence(self, sequence: str) -> str:
        """清理非标准氨基酸（X等）"""
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        return "".join([aa for aa in sequence if aa in valid_aas])
    
    def _enforce_rate_limit(self):
        """强制限流（避免429 Too Many Requests）"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            print(f"⏳ API限流保护：等待 {wait_time:.1f} 秒...")
            time.sleep(wait_time)
        self.last_request_time = time.time()
    
    def predict_structure(self, sequence: str, output_pdb: Optional[str] = None) -> Dict:
        """
        预测蛋白质结构
        
        Args:
            sequence: 氨基酸序列（可含X，自动清理）
            output_pdb: 可选，保存PDB文件路径
        
        Returns:
            {
                "success": bool,
                "plddt": float,        # 平均pLDDT
                "plddt_per_residue": List[float],  # 残基级pLDDT
                "pdb_content": str,    # PDB文本
                "error": str (if failed)
            }
        """
        # 清理序列
        clean_seq = self._clean_sequence(sequence)
        if len(clean_seq) == 0:
            return {"success": False, "error": "序列清理后为空"}
        
        # 限流保护
        self._enforce_rate_limit()
        
        try:
            # ESMFold API限制400残基，截断长序列
            seq_for_api = clean_seq[:400] if len(clean_seq) > 400 else clean_seq
            seq_for_api = clean_seq[:400] if len(clean_seq) > 400 else clean_seq
            response = requests.post(
                self.api_url,
                data=seq_for_api,
                headers=self.headers,
                timeout=60
            )
            response.raise_for_status()
            
            pdb_content = response.text
            
            # 解析pLDDT（从PDB B-factor列）
            plddt, plddt_per_residue = self._parse_plddt(pdb_content)
            
            # 保存PDB（可选）
            if output_pdb:
                os.makedirs(Path(output_pdb).parent, exist_ok=True)
                with open(output_pdb, "w") as f:
                    f.write(pdb_content)
            
            return {
                "success": True,
                "plddt": plddt,
                "plddt_per_residue": plddt_per_residue,
                "pdb_content": pdb_content,
                "sequence_length": len(clean_seq),
                "truncated": len(clean_seq) > 400
            }
        
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"API请求失败: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"未知错误: {str(e)}"}
    

    def _parse_plddt(self, pdb_text: str) -> tuple:
        """从PDB B-factor列提取pLDDT"""
        bfactors = []
        for line in pdb_text.split("\n"):
            if line.startswith("ATOM") and line[13:15] == "CA":  # 仅Cα原子
                try:
                    bfactor = float(line[60:66].strip())
                    bfactor *=100
                    bfactors.append(bfactor)
                except (ValueError, IndexError):
                    pass
        avg_plddt = sum(bfactors) / len(bfactors) if bfactors else 0.0
        return avg_plddt, bfactors
    

    def batch_validate(self, sequences: List[str], output_dir: str = "outputs/validation") -> List[Dict]:
        """
        批量验证多个序列
        
        Returns:
            List of validation results (same format as predict_structure)
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        print(f"🔬 批量验证 {len(sequences)} 个序列 (ESMFold API)...")
        for i, seq in enumerate(sequences, 1):
            print(f"   [{i}/{len(sequences)}] 正在验证...")
            result = self.predict_structure(
                seq, 
                output_pdb=f"{output_dir}/design_{i}.pdb"
            )
            result["design_id"] = i
            results.append(result)
        
        # 生成总结
        passed = sum(1 for r in results if r["success"] and r["plddt"] > 80)
        print(f"\n✅ 验证完成: {passed}/{len(sequences)} 通过 (pLDDT>80)")
        
        return results