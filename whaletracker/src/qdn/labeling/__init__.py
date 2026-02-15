"""
Labeling modules — López de Prado style.

triple_barrier.py  – Triple Barrier Method for realistic labels
purged_kfold.py    – Purged K-Fold CV with embargo
sample_weights.py  – Uniqueness-based sample weighting
"""

from .triple_barrier import TripleBarrierLabeler
from .purged_kfold import PurgedKFoldCV

__all__ = ["TripleBarrierLabeler", "PurgedKFoldCV"]
