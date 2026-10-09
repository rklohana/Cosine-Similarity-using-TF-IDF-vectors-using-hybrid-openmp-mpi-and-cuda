# Movie Similarity with TF-IDF, MPI, OpenMP, and CUDA

A C++17 project that ranks pairs of movies by the similarity of their descriptions. MPI divides document preprocessing across processes, OpenMP tokenizes documents in parallel within each process, and an optional CUDA backend computes the cosine-similarity matrix. A CPU backend makes the same pipeline usable without a GPU.

## What it produces

The program writes a ranked CSV with `movie_a`, `movie_b`, and `cosine_similarity` columns. For example, the four-document test fixture yields:

```csv
movie_a,movie_b,cosine_similarity
"Alpha, One","Beta ""Two""",0.948683298051
```

The score is approximately `3 / sqrt(10)` for these two descriptions; the fixture's unrelated or stop-word-only descriptions have zero similarity. This is a correctness example, not a benchmark result.

## Pipeline

1. Read movie titles and descriptions from a CSV file, including quoted commas, doubled quotes, and embedded newlines. The default limit is 1,000 valid records.
2. Split document ranges across MPI ranks. Each rank uses OpenMP to count terms in its assigned descriptions. Non-ASCII bytes and punctuation separate tokens; tokens are converted to lowercase and filtered with `StopWords.txt`.
3. Gather document term counts on rank 0. Rank 0 computes document frequencies, selects a **shared vocabulary** by descending document frequency, and builds TF-IDF vectors. Ties in vocabulary selection are alphabetical.
4. Use `TF = term_count / retained_token_count` and smoothed `IDF = log((1 + N) / (1 + document_frequency)) + 1`. Normalize each document vector to unit length. A document with no selected terms remains a zero vector.
5. Compute all pairwise dot products of the normalized vectors on the CPU or GPU. The CUDA implementation copies results back in row tiles, avoiding a full similarity matrix allocation on the GPU. Rank 0 writes the highest-scoring distinct movie pairs.

## Requirements

- Linux with a C++17 compiler, OpenMP support, an MPI implementation providing `mpicxx` and `mpirun`, and GNU Make.
- For the GPU build: NVIDIA CUDA toolkit with `nvcc`, a compatible NVIDIA GPU, and CUDA runtime libraries.
- Python 3 for the smoke-test runner.

Set `CUDA_ARCH` and `CUDA_HOME` for your installation if the Makefile defaults (`sm_70`, `/usr/local/cuda`) do not apply.

## Build and run

From the repository root:

```sh
make cpu
OMP_NUM_THREADS=4 mpirun -np 4 ./program-cpu --limit 1000 --vocab 256 --top 20 --verify
```

With CUDA available:

```sh
make gpu CUDA_ARCH=sm_86
OMP_NUM_THREADS=4 mpirun -np 4 ./program-gpu --limit 1000 --vocab 256 --top 20 --verify
```

`--verify` checks the full matrix for finite, symmetric scores in range and compares the first 32 by 32 scores against a CPU reference with a `1e-9` tolerance. The output defaults to `results/top_matches.csv`. Only rank 0 writes the output or uses the GPU.

| Option | Default | Description |
| --- | --- | --- |
| `--input PATH` | `movies_metadata - Sheet1.csv` | Movie CSV file; first two columns are title and description |
| `--stopwords PATH` | `StopWords.txt` | Words to exclude during tokenization |
| `--output PATH` | `results/top_matches.csv` | Ranked result CSV |
| `--limit N` | `1000` | Maximum valid movie records to read |
| `--vocab N` | `256` | Maximum terms in the shared vocabulary |
| `--top N` | `20` | Maximum distinct pairs to write |
| `--verify` | Off | Compare a sample of scores with the CPU reference |

The program prints document count, MPI rank count, OpenMP thread count, selected vocabulary size, backend, and preprocessing/similarity times. These are runtime observations; no speedup claim is included.

## Correctness check

The fixture tests quoted CSV fields, a known TF-IDF cosine score, stop-word-only input, and MPI processing with two ranks.

```sh
make cpu
python3 tests/smoke.py

make gpu
python3 tests/smoke.py --binary program-gpu
```

Both tests require `mpirun`; the GPU test also requires an NVIDIA GPU. The test script accepts `--mpirun` if the launcher has a different name or path.

## Project layout

| File | Role |
| --- | --- |
| `tfidfstg.cpp` | CSV parsing, distributed term counts, TF-IDF, CPU reference, CLI, result writing |
| `CudaCosineSimilarity.cu` | Tiled CUDA cosine kernel and device-memory handling |
| `CudaCosineSimilarity.hpp` | CUDA backend interface |
| `tests/fixture.csv`, `tests/smoke.py` | Small end-to-end correctness fixture and runner |
| `movies_metadata - Sheet1.csv`, `StopWords.txt` | Bundled movie data and stop-word list |

## Scope and scaling

Similarity calculation and result storage are quadratic in the number of movies. The GPU stores only a tile of results at a time, while rank 0 still holds the full matrix and the set of movie pairs for sorting. For larger corpora, stream a per-document top-k selection rather than retaining all pairs. Tokenization is ASCII-oriented; multilingual descriptions would need Unicode-aware tokenization. The bundled dataset is used as a demonstration input, and results depend on the selected vocabulary size.
