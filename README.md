# Loss of plasticity in continually trained recurrent neural networks

Code and result files for the study of plasticity loss in vanilla RNN, LSTM and GRU
networks with six candidate activation functions (tanh, ReLU, Leaky ReLU, ELU, SiLU, GELU),
an MLP baseline, and three remedies (L2-Init, shrink-and-perturb, layer normalisation).

Authors: Gaurav Kumar Pandey, Sumit Srivastava, Munish Khanna.
The accompanying manuscript is in preparation.

## Contents

| File | Purpose |
|---|---|
| `rnn_plasticity.py` | Runs the experiments (single file, PyTorch). Resumes from checkpoints and writes one JSON file per run. |
| `make_assets.py` | Reads the result files and produces every table, figure and number used in the manuscript, including the statistical tests. |
| `results/results_main.zip` | Per-run result files (one JSON per run) and worker logs of the main design, the fresh-network references and the remedies. |
| `requirements.txt` | Python dependencies. |

## Requirements

Python 3.10 or later and the packages in `requirements.txt`. The experiments were run on
NVIDIA Tesla T4 GPUs with PyTorch 2.11. MNIST and Fashion-MNIST are downloaded
automatically by torchvision.

```
pip install -r requirements.txt
```

## Running the experiments

A quick check of the code on synthetic data (a few minutes):

```
python rnn_plasticity.py --smoke --plan final --parallel 2
```

The main design (3 learning rates x 4 seeds = 12 units; each unit contains both task
streams, four architectures and six activations). `--parallel` starts that many worker
processes; `--shard` selects the first unit of the session:

```
python rnn_plasticity.py --seeds 4 --seed_base 0 --num_shards 12 --parallel 4 --shard 0
python rnn_plasticity.py --seeds 4 --seed_base 0 --num_shards 12 --parallel 4 --shard 4
python rnn_plasticity.py --seeds 4 --seed_base 0 --num_shards 12 --parallel 4 --shard 8
```

Fresh-network references and remedies:

```
python rnn_plasticity.py --plan night2 --num_shards 12 --parallel 4 --shard 0   # also 4 and 8
```

Fashion-MNIST replication, extended task sequence, fifth seed, duplicate audit:

```
python rnn_plasticity.py --plan final --num_shards 4 --parallel 4 --shard 0
```

Every session stops cleanly after `--hours` (default 11) and writes a zip file with the
results; unfinished work continues with `--resume_zip <file>`.

## Reproducing the tables and figures

`results/results_main.zip` contains one folder per compute session (`session01` ...
`session06`), each with the result files and the worker logs. Unpack it and run:

```
unzip results/results_main.zip -d results_unpacked
python make_assets.py paper results_unpacked
```

Result zips of later sessions are unpacked into further sub-folders of `results_unpacked`
(for example `session07`). When a run appears in two sessions, the later one is used.

This writes `paper/tables/*.tex`, `paper/figures/*.pdf` and `paper/numbers.tex`, and prints
a check of the qualitative statements made in the manuscript.

## Result file format

Each JSON file holds the configuration of the run (`bench`, `mode`, `arch`, `act`, `lr`,
`seed`, `remedy`, all hyper-parameters, software and hardware versions) and a per-task
`log` with accuracy, loss, gradient norms, unit and gate saturation, dormant-unit
fraction, effective rank and weight norms.

## Citation

To be added when the manuscript is published.
