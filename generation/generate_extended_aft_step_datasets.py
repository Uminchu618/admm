import argparse
import copy
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from generation.extended_aft_step_generator import build_generator, load_config


def generate_datasets(
    cfg: Dict,
    output_dir: Path,
    seed_start: int,
    seed_end: int,
    prefix: str,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in range(seed_start, seed_end + 1):
        cfg_seed = copy.deepcopy(cfg)
        cfg_seed["seed"] = seed

        generator = build_generator(cfg_seed)
        df = generator.simulate()

        output_path = output_dir / f"{prefix}{seed}.csv"
        if output_path.exists() and not overwrite:
            print(f"skip: {output_path}")
            continue

        df.to_csv(output_path, index=False)
        print(f"saved: {output_path}")


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

    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_datasets(
        cfg=cfg,
        output_dir=Path(args.output_dir),
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        prefix=args.prefix,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
