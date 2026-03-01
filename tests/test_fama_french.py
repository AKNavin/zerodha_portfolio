from pathlib import Path

import pandas as pd

from zerodha_portfolio.fama_french import build_factor_tilt_portfolio, load_ff_factors


def test_load_ff_factors_normalizes_columns_and_scale():
    csv_path = Path(__file__).parent / "_ff_test_input.csv"
    try:
        csv_path.write_text(
            "Date,Mkt-RF,RF,SMB,HML,WML\n"
            "2024-01-01,0.50,0.01,0.20,-0.10,0.30\n"
            "2024-01-02,-0.30,0.01,0.10,0.20,-0.05\n",
            encoding="ascii",
        )

        df = load_ff_factors(csv_path)
        assert list(df.columns) == ["MKT_RF", "RF", "SMB", "HML", "WML"]
        assert abs(float(df.iloc[0]["MKT_RF"]) - 0.005) < 1e-12
    finally:
        if csv_path.exists():
            csv_path.unlink()


def test_build_factor_tilt_portfolio_outputs_weights():
    betas = pd.DataFrame(
        {
            "beta_smb": [0.9, -0.1, 0.5, 0.2],
            "beta_hml": [0.7, -0.4, 0.1, 0.2],
            "beta_wml": [0.3, 0.2, 0.6, -0.2],
        },
        index=["A", "B", "C", "D"],
    )

    weights = build_factor_tilt_portfolio(
        betas=betas,
        preference={"beta_smb": 1.0, "beta_hml": 1.0, "beta_wml": 1.0},
        top_n=2,
    )

    assert len(weights) == 2
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    assert all(w > 0 for w in weights.values())
