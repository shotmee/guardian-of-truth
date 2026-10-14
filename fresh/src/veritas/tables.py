"""Чтение входа платформы и запись предсказаний."""
from __future__ import annotations

import os

import pandas as pd

PARQUET_MAGIC = b"PAR1"


def read_examples(path: str) -> pd.DataFrame:
    """Вход приходит parquet или csv — определяем по сигнатуре файла, а не по расширению."""
    with open(path, "rb") as source:
        magic = source.read(len(PARQUET_MAGIC))
    return pd.read_parquet(path) if magic == PARQUET_MAGIC else pd.read_csv(path)


def write_predictions(example_ids, labels, path: str) -> None:
    """Колонки id (string) и label (int64). Платформа читает parquet при любом расширении пути."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"id": pa.array([str(x) for x in example_ids], type=pa.string()),
                      "label": pa.array([int(x) for x in labels], type=pa.int64())})
    temporary_path = f"{path}.tmp"
    pq.write_table(table, temporary_path)
    os.replace(temporary_path, path)
