from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.visualize_lambda_results import plot_lambda_vs_c_td_with_cox


def test_plot_lambda_vs_c_td_with_cox_creates_png(tmp_path: Path) -> None:
    lambda_df = pd.DataFrame(
        {
            "lambda_fuse": [0.1, 0.1, 1.0, 1.0, 10.0, 10.0],
            "c_td": [0.70, 0.71, 0.72, 0.73, 0.74, 0.75],
        }
    )
    cox_df = pd.DataFrame(
        {
            "data_name": ["seed_1", "seed_2", "seed_3"],
            "c_td_cox": [0.69, 0.71, 0.70],
            "c_index_harrell": [0.68, 0.69, 0.70],
        }
    )

    plot_lambda_vs_c_td_with_cox(lambda_df, tmp_path, cox_df)

    output_path = tmp_path / "lambda_vs_c_td_with_cox.png"
    assert output_path.exists()
