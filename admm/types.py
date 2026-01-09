"""型定義（暫定）。

現時点では NumPy などの外部依存を最小化するため、ArrayLike を Any として定義している。
将来的に numpy.ndarray / scipy.sparse / pandas.DataFrame などを許容する場合は、
ここを Protocol や Union で具体化すると、型安全性と開発体験が向上する。
"""

from typing import Any

# ArrayLike:
# - 「配列のように扱える」入力を表す暫定型。
# - skeleton 段階では Any としておき、実装が固まったら具体型へ寄せる。
ArrayLike = Any
