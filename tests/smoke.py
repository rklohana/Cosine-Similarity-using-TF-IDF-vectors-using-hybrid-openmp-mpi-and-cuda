"""End-to-end CSV, TF-IDF, MPI, and optional CUDA smoke test."""

import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="program-cpu")
    parser.add_argument("--mpirun", default="mpirun")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "matches.csv"
        command = [
            args.mpirun,
            "-np",
            "2",
            str((ROOT / args.binary).resolve()),
            "--input",
            str(ROOT / "tests" / "fixture.csv"),
            "--stopwords",
            str(ROOT / "StopWords.txt"),
            "--output",
            str(output),
            "--limit",
            "4",
            "--vocab",
            "10",
            "--top",
            "6",
            "--verify",
        ]
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = "2"
        subprocess.run(command, check=True, cwd=ROOT, env=environment)
        with output.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

    assert len(rows) == 6, rows
    assert rows[0]["movie_a"] == "Alpha, One", rows[0]
    assert rows[0]["movie_b"] == 'Beta "Two"', rows[0]
    assert math.isclose(float(rows[0]["cosine_similarity"]), 3 / math.sqrt(10), abs_tol=1e-9)
    assert all(math.isclose(float(row["cosine_similarity"]), 0.0, abs_tol=1e-12)
               for row in rows[1:]), rows[1:]
    print("Smoke test passed")


if __name__ == "__main__":
    main()
