# Reading the platform's test file and writing the predictions back.
import os as _os

import pandas as _pd

_PARQUET_SIG = b"PAR1"


def read_examples(path):
    # The platform may hand us either parquet or CSV. We don't trust the
    # suffix; we sniff the leading magic bytes instead.
    with open(path, "rb") as handle:
        head = handle.read(len(_PARQUET_SIG))
    if head == _PARQUET_SIG:
        return _pd.read_parquet(path)
    return _pd.read_csv(path)


def write_predictions(example_ids, labels, path):
    # Two columns: id (string), label (int64). Platform reads parquet
    # regardless of the output extension.
    import pyarrow
    import pyarrow.parquet

    id_column = pyarrow.array([str(x) for x in example_ids], type=pyarrow.string())
    label_column = pyarrow.array([int(x) for x in labels], type=pyarrow.int64())
    table = pyarrow.table({"id": id_column, "label": label_column})
    staging = path + ".tmp"
    pyarrow.parquet.write_table(table, staging)
    _os.replace(staging, path)
