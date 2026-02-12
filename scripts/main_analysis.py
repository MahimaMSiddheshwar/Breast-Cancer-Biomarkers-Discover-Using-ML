import subprocess
import sys
from pathlib import Path


def run_notebook_analysis() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = root / "notebooks" / "ML_Analysis.ipynb"

    if not notebook.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook}")

    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        str(notebook),
    ]

    print("Running analysis notebook...")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=root)
    print("Analysis completed. Check the results folder for outputs.")


if __name__ == "__main__":
    run_notebook_analysis()
