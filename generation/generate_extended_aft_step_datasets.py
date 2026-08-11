import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from generation.extended_aft_step_generator import build_generator, load_config


def _write_dataset(generator, output_path: Path) -> None:
    """生成データと、その解析・真値グリッドのメタデータを保存する。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator.simulate().to_csv(output_path, index=False)
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta_path.write_text(
        json.dumps(generator.metadata(), indent=2),
        encoding="utf-8",
    )


def generate_datasets(
    cfg: Dict,
    output_dir: Path,
    seed_start: int,
    seed_end: int,
    prefix: str,
    overwrite: bool,
    baseline_alpha: float | None,
    eval_output_dir: Path | None = None,
    eval_seed_offset: int = 100_000,
    eval_n: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if baseline_alpha is not None:
        cfg.setdefault("baseline", {})
        cfg["baseline"]["alpha"] = float(baseline_alpha)
        print(f"override baseline.alpha = {cfg['baseline']['alpha']}")

    for seed in range(seed_start, seed_end + 1):
        cfg_seed = copy.deepcopy(cfg)
        cfg_seed["seed"] = seed

        output_path = output_dir / f"{prefix}{seed}.csv"
        if output_path.exists() and not overwrite:
            print(f"skip: {output_path}")
        else:
            _write_dataset(build_generator(cfg_seed), output_path)
            print(f"saved: {output_path}")

        if eval_output_dir is not None:
            eval_cfg = copy.deepcopy(cfg_seed)
            eval_cfg["seed"] = seed + eval_seed_offset
            if eval_n is not None:
                eval_cfg["n"] = eval_n
            eval_output_path = eval_output_dir / f"{prefix}{seed}.csv"
            if eval_output_path.exists() and not overwrite:
                print(f"skip: {eval_output_path}")
            else:
                _write_dataset(build_generator(eval_cfg), eval_output_path)
                print(f"saved: {eval_output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="段階的係数の拡張AFTデータをseed範囲で一括生成"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="generation/extended_aft_step_generator.config.json",
        help="設定ファイル（JSON）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/extended_aft_step",
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=42,
        help="開始seed（含む）",
    )
    parser.add_argument(
        "--seed-end",
        type=int,
        default=141,
        help="終了seed（含む）",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="extended_aft_step_seed_",
        help="出力ファイル名の接頭辞",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存ファイルを上書きする",
    )
    parser.add_argument(
        "--baseline-alpha",
        type=float,
        default=None,
        help="baseline.alpha を上書きする値（未指定なら設定ファイル値）",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=Path,
        default=Path("data/extended_aft_step_eval"),
        help="独立評価データの出力先。--skip-eval 指定時は使用しない。",
    )
    parser.add_argument(
        "--eval-seed-offset",
        type=int,
        default=100_000,
        help="評価データのseedに加える値。",
    )
    parser.add_argument(
        "--eval-n",
        type=int,
        default=None,
        help="評価データの標本サイズ（未指定なら学習データと同じ）。",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="独立評価データを生成しない。",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_datasets(
        cfg=cfg,
        output_dir=Path(args.output_dir),
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        prefix=args.prefix,
        overwrite=args.overwrite,
        baseline_alpha=args.baseline_alpha,
        eval_output_dir=None if args.skip_eval else args.eval_output_dir,
        eval_seed_offset=args.eval_seed_offset,
        eval_n=args.eval_n,
    )


if __name__ == "__main__":
    main()
