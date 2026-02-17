# src/__init__.py
"""BioAgent: End-to-end AIDD pipeline"""

# 导出核心类（方便顶层脚本导入）
from .rag.target_discovery import TargetDiscoveryRAG
from .mpnn.sequence_design import ProteinMPNNDesign
from .validation.esmfold_validator import ESMFoldValidator
from .validation.metrics import generate_validation_report

__all__ = [
    "TargetDiscoveryRAG",
    "ProteinMPNNDesign",
    "ESMFoldValidator",
    "generate_validation_report"
]