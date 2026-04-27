"""
I/O helpers for CSV / Parquet tables.
"""

from pathlib import Path
import pandas as pd


def read_source_csv(path: str | Path) -> pd.DataFrame:
    """Read Brand Analytics-like CSV with semicolon separator and embedded newlines."""
    path = Path(path)
    return pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        engine="python",
        on_bad_lines="warn",
    )


def write_table(df: pd.DataFrame, output_dir: str | Path, name: str) -> Path:
    """
    Write table as parquet when pyarrow/fastparquet is available.
    Fallback to CSV otherwise.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{name}.parquet"
    csv_path = output_dir / f"{name}.csv"

    try:
        df.to_parquet(parquet_path, index=False)
        if csv_path.exists():
            csv_path.unlink()
        return parquet_path
    except Exception:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return csv_path


def read_table(data_dir: str | Path, name: str) -> pd.DataFrame:
    """Read table saved by write_table."""
    data_dir = Path(data_dir)
    parquet_path = data_dir / f"{name}.parquet"
    csv_path = data_dir / f"{name}.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    raise FileNotFoundError(f"Table {name!r} not found in {data_dir}")


def write_manifest(output_dir: str | Path, manifest: dict) -> Path:
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
