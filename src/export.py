import pandas as pd
from pathlib import Path

def create_csv(data: list[dict], save_path: str | Path = "results", filename: str = "piyavskiy_results.xlsx") -> pd.DataFrame:
    """
      Create and save a CSV file from iteration results.

      param: save_path: Path where the CSV file will be saved.
      param data: List of iteration dictionaries (from piyavskiy_method).

      Returns: pd.DataFrame: The resulting DataFrame used to create the CSV file.
      """

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / filename

    df = pd.DataFrame(data)
    df.rename(columns={
        "iteration": "Итерация",
        "u": "uₙ",  # u с подстрочным n
        "p_{n-1}(u_n)": "pₙ₋₁(uₙ)",  # p с подстрочным n-1 и uₙ
        "W(u_n)": "f(uₙ)",  # f(uₙ) — u с подстрочным n
        "delta": "Δ"  # дельта — греческая буква
    }, inplace=True)
    df = df.round(4)
    # df.to_csv(file_path, index=False, encoding="utf-8-sig")
    df.to_excel(file_path, index=False)
    return df
