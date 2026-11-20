from pathlib import Path
from runpy import run_path

if __name__ == "__main__":
    entry = Path(__file__).resolve().parents[1] / 'src/splatforge/studio/native_source' / "train.py"
    run_path(str(entry), run_name="__main__")
