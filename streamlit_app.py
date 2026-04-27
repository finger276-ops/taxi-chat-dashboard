"""Cloud entrypoint for Streamlit Community Cloud.

Main dashboard code lives in src/app.py. This wrapper keeps deployment simple:
select streamlit_app.py as the entrypoint in Streamlit Cloud.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from app import main


if __name__ == "__main__":
    main()
