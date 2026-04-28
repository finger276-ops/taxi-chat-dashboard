"""
I/O helpers for source files and generated CSV / Parquet tables.
"""

from pathlib import Path
import pandas as pd

from import_adapters import read_source_table


def read_source_csv(path: str | Path) -> pd.DataFrame:
    """Backward-compatible source reader.

    Despite the historical name, this now supports CSV and Excel exports from
    Mediologia, Brand Analytics and generic tabular files.
    """
    return read_source_table(path)


def write_table(df: pd.DataFrame, output_dir: str | Path, name: str) -> Path:
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
