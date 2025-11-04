import pandas as pd
from pathlib import Path

def create_csv(data: list[dict], save_path: str | Path = "results", filename: str = "piyavskiy_results.csv") -> pd.DataFrame:
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
        '$p_{n - 1}(u_n)$': 'p_(n-1)(u_n)',
        '$f(u_n)$': 'f(u_n)'
    }, inplace=True)
    df = df.round(4)
    df.to_csv(file_path, index=False, encoding="utf-8")
    return df