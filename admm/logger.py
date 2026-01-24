"""WandB ロギング用のユーティリティ。

方針:
    - WandB は任意依存。未インストールでも学習自体は動作させる。
    - ロギングは推定器から分離し、外側（main 等）で利用する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional


def _import_wandb():
    try:
        import importlib

        return importlib.import_module("wandb")
    except Exception as exc:  # noqa: BLE001 - 任意依存のため広めに捕捉
        raise RuntimeError(
            "wandb がインストールされていません。"
            " `pip install wandb` を実行するか、ロギングを無効化してください。"
        ) from exc


def wandb_available() -> bool:
    """wandb が利用可能かを返す。"""

    try:
        _import_wandb()
        return True
    except RuntimeError:
        return False


@dataclass
class WandBLogger:
    """WandB へのロギングを行うクラス。"""

    project: str
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: Optional[Iterable[str]] = None
    enabled: bool = True
    _run: Any = field(default=None, init=False, repr=False)

    def start_run(
        self,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> None:
        """WandB run を開始する。"""

        if not self.enabled:
            return
        wandb = _import_wandb()
        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=name or self.name,
            tags=(
                list(tags)
                if tags is not None
                else (list(self.tags) if self.tags else None)
            ),
            config=config,
        )

    def log_fit(self, history_row: Dict[str, Any], step: int) -> None:
        """ADMM の 1 反復分の履歴を記録する。"""

        if not self.enabled:
            return
        wandb = _import_wandb()
        payload = {f"history/{key}": value for key, value in history_row.items()}
        wandb.log(payload, step=step)

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: Optional[str] = None,
    ) -> None:
        """評価指標をログに送る。"""

        if not self.enabled:
            return
        wandb = _import_wandb()
        if prefix:
            payload = {f"{prefix}/{key}": value for key, value in metrics.items()}
        else:
            payload = dict(metrics)
        wandb.log(payload, step=step)

    def log_history(self, history: Dict[str, Any], prefix: str = "history") -> None:
        """履歴 dict を時系列としてまとめて記録する。"""

        if not self.enabled:
            return
        wandb = _import_wandb()

        series_keys = [
            key for key, value in history.items() if isinstance(value, (list, tuple))
        ]
        if not series_keys:
            return
        n_steps = max(len(history[key]) for key in series_keys)
        for step in range(n_steps):
            payload: Dict[str, Any] = {}
            for key in series_keys:
                values = history.get(key)
                if values is None or step >= len(values):
                    continue
                payload[f"{prefix}/{key}"] = values[step]
            if payload:
                wandb.log(payload, step=step)

    def finish(self) -> None:
        """WandB run を終了する。"""

        if not self.enabled:
            return
        wandb = _import_wandb()
        wandb.finish()
        self._run = None
