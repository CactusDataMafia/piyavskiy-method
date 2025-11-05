import pandas as pd
from pathlib import Path


def get_next_run_dir(base_dir: str = "results") -> Path:

    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)

    counter_file = base_path / "run_counter.txt"

    if counter_file.exists():
        with open(counter_file, "r", encoding="utf-8") as file:
            try:
                run_number = int(file.read().strip()) + 1
            except (ValueError, OSError):
                run_number = 1
    else:
        run_number = 1

    with open(counter_file, "w", encoding="utf-8") as file:
        file.write(str(run_number))

    run_dir = base_path / f"run_{run_number:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


save_path = get_next_run_dir()


def create_csv(data: list[dict], save_path: Path = save_path, filename: str = "piyavskiy_results.xlsx") -> pd.DataFrame:
    """
      Create and save a CSV file from iteration results.

      param: save_path: Path where the CSV file will be saved.
      param data: List of iteration dictionaries (from piyavskiy_method).

      Returns: pd.DataFrame: The resulting DataFrame used to create the CSV file.
      """

    file_path = save_path / filename

    df = pd.DataFrame(data)
    df.rename(columns={
        "iteration": "Итерация",
        "u": "uₙ",  # u с подстрочным n
        "p_{n-1}(u_n)": "pₙ₋₁(uₙ)",  # p с подстрочным n-1 и uₙ
        "W(u_n)": "f(uₙ)",  # f(uₙ) — u с подстрочным n
        "delta": "Δ"  # дельта — греческая буква
    }, inplace=True)
    df = df.round(4)
    df.to_excel(file_path, index=False)
    return df
