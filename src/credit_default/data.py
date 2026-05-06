from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "default_of_credit_card_clients.xls"
)


def load_credit_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Carga el dataset crudo de crédito.
    """

    path = Path(path)

    df = pd.read_excel(path, skiprows=1)

    df.columns = df.columns.str.strip()

    return df