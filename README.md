# DMG-BEE — Run Instructions

> 中文版本 / Chinese version: **[README_zh.md](README_zh.md)**

Biomedical event extraction (BEE) experiments on **BioNLP Shared Task 2013 GE (GE13)**, implemented in Python + PyTorch.
Main pipeline: **SciBERT encoding → sentence-level span scoring with a global pointer (RoPE) → multi-label cross-entropy → event merging → official GE13 Perl evaluation**.
The repository also keeps a two-stage "trigger → argument" pipeline and an unused GCN graph-reasoning module (see Sections 8 and 9).

---

## Contents

1. [Requirements](#1-requirements)
2. [Repository layout](#2-repository-layout)
3. [Data](#3-data)
4. [Quick start (verify the evaluation chain in 3 minutes)](#4-quick-start-verify-the-evaluation-chain-in-3-minutes)
5. [Full training: `run_model.py`](#5-full-training-run_modelpy)
6. [Output files and re-evaluation](#6-output-files-and-re-evaluation)
7. [Test set and online submission](#7-test-set-and-online-submission)
8. [Two-stage pipeline: `main.py`](#8-two-stage-pipeline-mainpy)
9. [Event merging model (DualCL): `merge/GE13/`](#9-event-merging-model-dualcl-mergege13)
10. [End-to-end command list](#10-end-to-end-command-list)
11. [FAQ](#11-faq)
12. [Reference results](#12-reference-results)

---

## 1. Requirements

| Item | Requirement | Notes |
| --- | --- | --- |
| OS | Windows 10/11 or Linux | The official evaluation scripts are Perl scripts and are cross-platform |
| Python | **3.8** (3.6–3.8 was the original dev stack) | 3.10+ not verified; the shipped `__pycache__` contains cpython-36/37/38 bytecode |
| PyTorch | ≥ 1.8 (original: **1.10.0 + CUDA 10.2**) | Uses `torch.optim.AdamW` and `pack_padded_sequence(enforce_sorted=False)` |
| transformers | 4.x (original ≈ 4.6–4.11) | Only `AutoTokenizer / AutoModel / AutoConfig / get_linear_schedule_with_warmup` are used |
| GPU | ≥ 8 GB VRAM (11 GB+ recommended) | SciBERT-base, `max_len=512`, training `batch_size=8` |
| **Perl** | **Mandatory** | GE13 official `a2-normalize.pl` / `a2-evaluate.pl`; **called at the end of every training epoch**, otherwise training crashes (FAQ 11.3) |
| spacy + scispacy `en_core_sci_md` | Only for rebuilding the dataset | `sources/convert-doc-GE13.py` needs dependency parsing |
| dgl | Optional | Only for `model/graph_reason.py` (MixHop/GCN); not imported by any entry script |

### 1.1 Create the environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

```bash
# 1) PyTorch: pick the command for your CUDA version from https://pytorch.org, e.g. CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 2) Everything else
pip install "transformers>=4.6,<4.40" tqdm numpy scikit-learn

# 3) Only if you rebuild the dataset from raw BioNLP files
pip install spacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_md-0.5.3.tar.gz
```

### 1.2 Install Perl (mandatory)

| Platform | How |
| --- | --- |
| Windows | Install [Strawberry Perl](https://strawberryperl.com/) (or `choco install strawberryperl`); make sure `perl -v` works |
| Ubuntu / Debian | `sudo apt-get install -y perl` |
| CentOS | `sudo yum install -y perl` |

> Check with `perl -v`.

### 1.3 Put the pretrained language model in place (mandatory)

`bert/scibert_scivocab_uncased/` in this repository is an **empty directory**; download SciBERT weights into it:

```text
DMG-BEE/
└── bert/
    └── scibert_scivocab_uncased/
        ├── config.json
        ├── vocab.txt
        ├── pytorch_model.bin      (or model.safetensors)
        └── tokenizer_config.json / tokenizer.json
```

- HuggingFace: <https://huggingface.co/allenai/scibert_scivocab_uncased>
- Command line:

  ```bash
  pip install huggingface_hub
  huggingface-cli download allenai/scibert_scivocab_uncased --local-dir bert/scibert_scivocab_uncased
  ```

- In mainland China you may use the mirror: `export HF_ENDPOINT=https://hf-mirror.com` before running the command above.

> `--bert_path` defaults to `./bert/scibert_scivocab_uncased` in both `main.py` and `run_model.py`; any local BERT directory works
> (the directory name is used as a suffix of the preprocessed cache files, e.g. `train-scibert_scivocab_uncased.pt`).

---

## 2. Repository layout

```text
DMG-BEE/
├── run_model.py            ★ main entry: single-stage train / dev / merge / evaluate (runs with the shipped data)
├── main.py                   alternative entry: two-stage (trigger → argument) pipeline
├── model_base.py             model used by run_model.py: MentionPointer (BERT + BiGRU + RoPE global pointer)
├── dataloader.py             preprocessing for run_model.py (SpanDataloader / MyDataset / my_collate)
├── metrics.py                span decoding, trigger/argument/network scoring, event_merge
├── merge.py                  predictions → BioNLP *.a1/*.a2/*.txt, then runs the official Perl evaluation
├── logger.py                 logging into ./log/{task_name}.log
├── binding_clean.py          utility: Binding-event distance statistics (writes binding.pt, analysis only)
├── BioNLP在线测试文档.docx    screenshots of the GE13 online (tar.gz upload) evaluation procedure
├── filename.tar.gz           a previous test-set submission bundle (experiments/GE13/test a1/a2/txt)
│
├── model/                    (used by main.py)
│   ├── model_base.py         DMGModel: BERT + BiGRU + 3 heads (trigger/relation/sentence)
│   ├── graph_reason.py       GCN / Edgeupdate / MixHop (needs dgl, not referenced anywhere)
│   └── loss_func.py          CELoss / SupConLoss / DualLoss (not referenced anywhere)
├── dataloader/dataloader.py  (used by main.py) reads sources/GE13/std-*.json
├── metrics/metrics.py        (used by main.py) performance_report / generation
│
├── bert/                     SciBERT weights directory (download yourself, see 1.3)
├── log/GE13.log              historical training log (per-epoch metrics)
│
├── sources/                  data + conversion scripts
│   ├── GE13/                 ★ the dataset (train/dev/test + index files)
│   ├── MLEE/                 (empty) MLEE is not shipped
│   ├── CG13/                 another corpus directory (empty, unused)
│   ├── utils/                tokenisation, sentence splitting, UMLS filtering helpers
│   ├── convert-doc-GE13.py   raw *.txt/*.a1/*.a2 → GE13/*.json (+ span/prior/tag_limits/structure)
│   ├── convert-tag-GE13.py   *.json → std-*.json (for the two-stage pipeline)
│   ├── convert-merge-GE13*.py *.json → *.pt (main pipeline) / merge-*.json (merging model)
│   └── convert-*-MLEE.py     MLEE variants (hard-coded ../Joint-Task/datasets/MLEE, edit before use)
│
├── experiments/GE13/         training artifacts (cached tensors, schema, prediction json)
│   ├── train-scibert_scivocab_uncased.pt   preprocessed cache (13.8 MB)
│   ├── devel-scibert_scivocab_uncased.pt   preprocessed cache (14.2 MB)
│   ├── test-scibert_scivocab_uncased.pt    preprocessed cache (13.8 MB)
│   ├── schema.json          label spaces (15 trigger types, 28 relations, 49 POS, 47 dependencies)
│   ├── structure.json       structural constraints on event arguments
│   └── devel_*_42.json      one historical dev prediction (triggers/relations/networks)
│
├── merge/GE13/              event merging (DualCL)
│   ├── train_merge.py       trains the merging model, writes merge.bin
│   ├── merge.bin            already-trained merging model (~420 MB)
│   ├── merge_v1.py          rule-based merging: prediction json → *.a2
│   ├── data_merge.py        → predictions.json
│   ├── inference.py         placeholder inference script
│   └── pred-devel/          a sample of merged dev predictions (*.a2)
│
└── evaluation/              official evaluation tools
    ├── GE13/evaluation.py   calls the Perl scripts → end-to-end F1
    ├── GE13/tools/*.pl      a2-normalize.pl / a2-evaluate.pl / a2-decompose.pl / a2d-evaluate.pl
    └── MLEE/evaluation-MLEE.py + test/ (MLEE test a1/a2/txt)
```

---

## 3. Data

### 3.1 Contents of `sources/GE13/`

```text
sources/GE13/
├── train/  devel/  test/     raw corpus: *.txt (text), *.a1 (entities), *.a2 (events; absent in test), *.json (official)
├── train.json  devel.json  test.json    converted instances (JSON Lines, one sentence-level instance per line)
├── train.pt   devel.pt   test.pt       ★ pickled data read directly by run_model.py
├── std-train.json  std-devel.json  std-test.json   (must be generated by convert-tag-GE13.py, for main.py)
├── span-train.json  span-devel.json  span-test.json  document → sentence span index (drops cross-sentence spans)
├── prior.json               trigger dictionary (876 words → event-type prior)
├── tag_limits.json          POS tag → id (49 classes)
├── structure.json           event type → valid argument roles/types
├── tokens.json / dist.json  vocabulary / argument-distance statistics (not read by the current code)
└── segment/*.jsonl          paragraph-level intermediate files
```

| Split | Documents | Sentence-level instances (`*.pt`) | Gold labels |
| --- | --- | --- | --- |
| train | 222 | **1198** | yes |
| devel | 249 | **1318** | yes |
| test | 305 | **1508** | **no** (online evaluation only) |

> Instance counts measured with `pickle.load(sources/GE13/*.pt)`.

### 3.2 Label space (`experiments/GE13/schema.json`)

| Key | Size | Meaning |
| --- | --- | --- |
| `lab` | 15 | trigger / event types (including `Entity`, `Anaphora`) |
| `rel` | 28 | argument roles + nested-event relations (`Theme`, `Theme_rev`, `Theme_nested`, `Cause`, …) |
| `ent` | 3 | `O` / `B-Protein` / `I-Protein` |
| `tag` | 49 | POS tags |
| `dep` | 47 | dependency labels |

### 3.3 (Optional) rebuild the dataset from the raw BioNLP files

> The converted `*.json` / `*.pt` files are already shipped — **no rebuild is needed for normal use**;
> only do this when you change the corpus or the preprocessing logic.

```bash
# (1) raw *.txt/*.a1/*.a2 → GE13/*.json + span-*.json + prior.json + tag_limits.json + structure.json + tokens.json
python convert-doc-GE13.py

# (2) *.json → std-*.json (two-stage pipeline only)
python convert-tag-GE13.py

# (3) *.json → *.pt (needed by run_model.py)
python convert-merge-GE13-v5.py

# (4) *.json → merge-train.json / merge-devel.json (needed by train_merge.py only)
python convert-merge-GE13.py
```

**Important caveats**

1. All conversion scripts use `./GE13/...`, so they must be run **inside `sources/`**; but `convert-doc-GE13.py` line 9 is
   `from sources.utils.tokenization_bert import BasicTokenizer`, which requires the **repository root** on `sys.path`. Pick one:
   - Option A (recommended): change line 9 to `from utils.tokenization_bert import BasicTokenizer`, then run from `sources/`;
   - Option B: run from the repository root and move the output `./GE13/` back to `sources/GE13/`.
2. `convert-doc-GE13.py` requires `spacy.load("en_core_sci_md")` (scispaCy); otherwise you get an `OSError`.
3. These scripts have **no `if __name__ == '__main__'` guard** — importing them runs them.
4. Several versions exist (`convert-merge-GE13.py`, `-v1` … `-v5`, `convert-tag-*`). `v5` produces the current `*.pt` format;
   `convert-merge-GE13.py` produces `merge-*.json` for the merging model.
5. `sources/MLEE/` is empty and `convert-doc-MLEE.py` is hard-coded to `../Joint-Task/datasets/MLEE`:
   **the MLEE branch cannot be run as-is.**

---

## 4. Quick start (verify the evaluation chain in 3 minutes)

To check the environment and reproduce one evaluation **without a GPU and without training**, reuse the shipped dev predictions:

```bash
cd DMG-BEE
python - <<'PY'
import json
from merge import merge

root = "experiments/GE13"
aux = [
    json.load(open(f"{root}/devel_trigger_p_42.json", encoding="utf-8")),
    None,
    json.load(open(f"{root}/devel_relation_p_42.json", encoding="utf-8")),
    None,
    json.load(open(f"{root}/devel_network_p_42.json", encoding="utf-8")),
    None,
    json.load(open(f"{root}/devel_entity_g_42.json", encoding="utf-8")),
]
score = merge(aux=aux, source_path="sources/GE13", save_path="experiments/GE13", mode="devel")
print("end-to-end F1 =", score)
PY
```

PowerShell does not support heredocs: save the lines between `PY` into `check_eval.py` and run `python check_eval.py`.

Expected on success

- the full `a2-evaluate.pl` report is printed (including the `Approximate Span / Recursive` table);
- `experiments/GE13/devel/{a1,a2,txt}/` are created with the predicted standoff files;
- the script prints `end-to-end F1 = 0.5x–0.58` (the value reflects this particular set of shipped dev predictions).

> This step needs Perl. Without it, `merge()` raises `IndexError: list index out of range` while parsing the evaluation output.

---

## 5. Full training: `run_model.py`

### 5.1 One command

```bash
cd DMG-BEE
python run_model.py
```

With the defaults (`--task_name GE13`, `--seed 42`, `--gpu_id 0`) it will:

1. read `sources/GE13/{train,devel,test}.pt` (preprocessing and caching into `experiments/GE13/*.pt` if the cache is missing);
2. build or load the label space in `experiments/GE13/schema.json`;
3. train `MentionPointer` (SciBERT + BiGRU + RoPE global pointer + multi-label CE);
4. **every epoch**: decode dev → `merge()` writes `*.a2` → run the Perl evaluation for the official F1;
5. keep the best model w.r.t. **event-level F1** as `experiments/GE13/best_model_42.bin` and dump dev predictions as json;
6. stop early after `--early_stop` (default 50) epochs without improvement.

Logs go to both stdout and `log/GE13.log`; typical lines:

```text
devel metric
precision (Overall): 68.0096%
recall (Overall): ...
f1 (Overall): 68.0096%
f1 (Event): 56.8600%
current best f1: tri: 58.1900%
current best epoch: 11
```

### 5.2 Arguments (`run_model.py`)

| Argument | Default | Description |
| --- | --- | --- |
| `--task_name` | `GE13` | `GE13` or `MLEE`; **MLEE data is missing, so only GE13 works** |
| `--seed` | `42` | random seed; also used in `best_model_{seed}.bin` |
| `--train_path` / `--devel_path` / `--test_path` | `sources/GE13/{train,devel,test}.pt` | pickled instances |
| `--prior_path` | `sources/GE13/prior.json` | trigger prior dictionary |
| `--output_path` | `experiments/GE13` | cache and prediction output directory |
| `--bert_path` | `bert/scibert_scivocab_uncased` | pretrained model directory |
| `--max_len` | `512` | sub-token limit (truncation) |
| `--inner_dim` | `64` | per-label vector size of the global pointer |
| `--ent_dim` | `50` | entity-type embedding size |
| `--RoPE` | `True` | use rotary position embeddings |
| `--drop_rate` | `0.5` | dropout |
| `--lr` | `5e-5` | learning rate (AdamW) |
| `--epoch` | `300` | max epochs (early stopping usually kicks in first) |
| `--early_stop` | `50` | patience |
| `--max_grad_norm` | `1.0` | gradient clipping |
| `--train_batch_size` | `8` | lower this first if you hit OOM |
| `--devel_batch_size` / `--test_batch_size` | `32` | inference batch sizes |
| `--gpu_id` | `0` | which GPU to use |
| `--do_train` / `--do_devel` / `--do_test` | `True` | switches (**argparse bool pitfall, see FAQ 11.6**) |

> `--stage` and `--fix_bert_embedding` are still defined in `run_model.py` but unused by the main flow
> (`--stage` belongs to the two-stage pipeline).

### 5.3 Tuning / speed-up tips

| Goal | How |
| --- | --- |
| Sanity check | `python run_model.py --epoch 2 --early_stop 2` |
| Lower memory usage | `--train_batch_size 4 --max_len 384` |
| Faster epochs | comment out the `merge(...)` call on line 258 (select on dev loss instead), or evaluate every 5 epochs |
| Another GPU | `--gpu_id 1`, and change line 49 `os.environ['CUDA_VISIBLE_DEVICES'] = str(args.seed)` to `str(args.gpu_id)` (see FAQ 11.7) |

---

## 6. Output files and re-evaluation

Key files under `experiments/GE13/` after training/prediction:

| File | Content |
| --- | --- |
| `best_model_42.bin` | best model w.r.t. dev event-level F1 (`seed=42`) |
| `devel_trigger_p_42.json` / `_g_42.json` | dev trigger predictions / gold (document → span set) |
| `devel_relation_p_42.json` / `_g_42.json` | dev argument-relation predictions / gold |
| `devel_network_p_42.json` / `_g_42.json` | dev nested-event chain predictions / gold |
| `devel_entity_g_42.json` | dev entities (gold, used to build `a1`) |
| `devel/{a1,a2,txt}/` | merged BioNLP standoff files (input to the official evaluation) |
| `schema.json` / `structure.json` | label space and structural constraints |

**Merge + evaluate only (no retraining)**

```bash
# Option 1: copy the seed-suffixed files to the names expected by merge.py's __main__, then run
#   devel_trigger_p_42.json  → devel_trigger_p.json
#   devel_relation_p_42.json → devel_relation_p.json
#   devel_network_p_42.json  → devel_network_p.json
#   devel_entity_g_42.json   → devel_entity_g.json
python merge.py

# Option 2 (recommended): call merge() directly, as in the script of Section 4;
#           you can pass mode="devel" or mode="test".
```

What `merge()` does per document: copy `*.txt` and the gold `*.a2` into `a1/`, then write the predicted entity `*.a1`
and event `*.a2`, and finally execute:

```bash
./evaluation/GE13/tools/a2-normalize.pl -g ./sources/GE13/devel/ -u ./experiments/GE13/devel/a2/*.a2
./evaluation/GE13/tools/a2-evaluate.pl  -g ./sources/GE13/devel/ -t1 -sp ./experiments/GE13/devel/a2/*.a2
```

> Therefore **`merge.py` must be run from the repository root (`DMG-BEE/`)**, because it uses `./sources/...` and `./evaluation/...`.

---

## 7. Test set and online submission

The GE13 test set ships **without `*.a2` gold annotations**, so you can only produce predictions locally; scoring requires
the official BioNLP evaluation server (procedure documented with screenshots in `BioNLP在线测试文档.docx`).

1. **Generate test predictions.** Replace the tail of `run_model.py` with:

   ```python
   Pipeline = PipeLineTrainer(args=args, logger=exp_logger)
   # Pipeline.run_train()
   Pipeline.prediction(Pipeline.test_data, mode="test", source_path="./sources/GE13/test")
   ```

   This loads `experiments/GE13/best_model_42.bin`, decodes, writes `test_*_42.json`, runs `merge(mode="test")` to create
   `experiments/GE13/test/{a1,a2,txt}/`, and finally packs `filename.tar.gz`.

2. **Check the archive layout**: `*.a1/*.a2/*.txt` at the top level, no gold annotations, no extra nesting.

   ```bash
   tar -tzf filename.tar.gz | head
   # ./experiments/GE13/test/a1/PMC-....a1 ...
   ```

   ⚠ If a *flat* archive is required (some servers expect `tar -zcvf filename.tar.gz *` executed where the `a2` files are),
   pack from inside the directory:

   ```bash
   cd experiments/GE13/test/a2 && tar -zcvf ../../../../filename.tar.gz *
   ```

3. **Request evaluation access**: e-mail `majordomo@bionlp-st.org` with the body `subscribe bionlp-st`;
   confirm the reply that your address was added to the evaluation server.

4. **Upload**: open <http://bionlp-st.dbcls.jp/GE/2011/eval-development/>, enter your e-mail, upload the `tar.gz`, and read the scores.

5. **Which row to report**: the F1 in the second table, **Approximate Span / Recursive**, is the number used in the paper
   (it corresponds to the local `a2-evaluate.pl -t1 -sp` run).

---

## 8. Two-stage pipeline: `main.py`

`main.py` is the earlier **"extract triggers, then classify arguments"** implementation; the model is
`model/model_base.py::DMGModel` (SciBERT → BiGRU → LayerNorm → 3 heads: `tri` (15 classes), `rel` (28 classes),
`lab` (sentence-level, 2 classes)). It reads `sources/GE13/std-*.json`.

### 8.1 Prerequisite: generate `std-*.json`

```bash
cd sources
python convert-tag-GE13.py     # writes GE13/std-train.json / std-devel.json / std-test.json
```

### 8.2 Training commands

```bash
cd DMG-BEE

# Stage 1: trigger tagging
python main.py --stage trigger --task_name GE13 --lr 1e-5 --train_batch_size 16 --epoch 300
# → experiments/GE13/best_trigger_model.bin

# Stage 2: argument classification (loads stage-1 weights and dev/test trigger logits)
python main.py --stage argument --task_name GE13 --lr 1e-5 --train_batch_size 16 --epoch 300
# → experiments/GE13/best_argument_model.bin
```

> Note: `main.py` ends with `Pipeline.prediction()`, which requires `best_argument_model.bin`.
> Running `python main.py` before any training therefore fails with a missing-file error; for the first run replace the
> last lines with `Pipeline.run_train()`, or apply the fixes of 8.3.

### 8.3 Known gaps (this branch is unfinished)

The `main.py` pipeline is a **work in progress** in this repository. Three small fixes are needed:

| Location | Symptom | Fix |
| --- | --- | --- |
| `main.py:13` `from dataloader.dataloader import Dataloader` | `ImportError` (the module only defines `SpanDataloader`) | `from dataloader.dataloader import SpanDataloader as Dataloader` |
| `model/model_base.py:19` uses `args.edge_dim` | `AttributeError: 'Namespace' object has no attribute 'edge_dim'` | add `parser.add_argument('--edge_dim', type=int, default=50)` in `main.py` |
| `metrics/metrics.py:22` reads `output["lab"]["predicts"]` | `TypeError` ( `DMGModel` returns a plain tensor `logits`) | `torch.argmax(output["lab"], dim=-1).tolist()` |
| missing arguments | the 2022 log contained `--num_layer` / `--gnn_type`, no longer defined | only needed if you attach the graph modules |

After these fixes, train with `--stage trigger` and then `--stage argument`. Its outputs
(`devel_trigger_pred.json`, `devel_relation_pred.json`, without the seed suffix) are exactly what the merging model of Section 9 consumes.

---

## 9. Event merging model (DualCL): `merge/GE13/`

To merge trigger/argument predictions into complete events (important for nested events), the repository provides a
contrastive-learning based merging classifier:

```bash
cd merge/GE13

# (1) Train the merging model (data: sources/GE13/merge-train.json / merge-devel.json)
python train_merge.py --data_dir ../../sources/GE13 --model_name ../../bert/scibert_scivocab_uncased
# → ./merge.bin  (a trained 420 MB checkpoint ships with the repo, so you can skip this)

# (2) Post-processing: turn prediction json into BioNLP *.a1/*.a2
python merge_v1.py        # reads ../../experiments/GE13/devel_trigger_pred.json etc., writes devel-pred/
python data_merge.py      # aggregates into predictions.json
python analysis.py        # compares with ../../sources/GE13/merge-devel.json

# (3) Standalone evaluation
cd ../../evaluation/GE13
python evaluation.py      # compares ../../sources/GE13/devel with ../../experiments/GE13/devel-pred
```

| Script | Input | Output |
| --- | --- | --- |
| `train_merge.py` | `sources/GE13/merge-train.json`, `merge-devel.json`; `--method dualcl/ce/scl` | `merge/GE13/merge.bin`, `log/dualcl.log` |
| `merge_v1.py` | `sources/GE13/devel/`, `experiments/GE13/devel_{trigger,relation}_pred.json` | `experiments/GE13/devel-pred/*.a2` |
| `data_merge.py` | `experiments/GE13/devel_relation_pred.json`, `sources/GE13/devel.json` | `merge/GE13/predictions.json` |
| `inference.py` | `predictions.json` | placeholder demo script (12 lines) |

> ⚠ Working directory: these scripts assume **CWD = `merge/GE13/`** and **file names without the seed suffix**
> (`devel_trigger_pred.json`, produced by `main.py`). If you use `run_model.py`'s `*_p_42.json`, rename them or edit the paths.

---

## 10. End-to-end command list

Everything from a clean machine to a score (Linux/macOS bash; on Windows run the same lines in PowerShell):

```bash
# 0) environment
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install "transformers>=4.6,<4.40" tqdm numpy scikit-learn
sudo apt-get install -y perl
huggingface-cli download allenai/scibert_scivocab_uncased --local-dir bert/scibert_scivocab_uncased

# 1) repository root (all relative paths are anchored here)
cd DMG-BEE

# 2) self-check: CUDA and Perl
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
perl -v | head -2

# 3) train + validate (merging and the official evaluation run every epoch)
python run_model.py --gpu_id 0

# 4) evaluate again (first copy *_p_42.json to *_p.json, or use the script in Section 4)
python merge.py

# 5) test inference + packaging (switch the tail of run_model.py to Pipeline.prediction(...), see Section 7)
#    → experiments/GE13/test/{a1,a2,txt}/ and filename.tar.gz
```

---

## 11. FAQ

### 11.1 `ModuleNotFoundError: No module named 'transformers'`

The dependency is missing, or installed into a different interpreter. Check that `which python` / `where python`
and `pip -V` point to the same environment, then reinstall.

### 11.2 `OSError: Error no file named pytorch_model.bin ... scibert_scivocab_uncased`

`bert/scibert_scivocab_uncased/` is empty — download the weights as described in [1.3](#13-put-the-pretrained-language-model-in-place-mandatory).

### 11.3 `IndexError: list index out of range` right after the first epoch

`merge.py` parses the output of `a2-evaluate.pl`; if **Perl is unavailable or the script fails**, the output is empty.
Install Perl (1.2), or temporarily comment out the `merge(...)` call on line 258 of `run_model.py` and select on dev loss.

### 11.4 `AssertionError: PMC-2222968-21-Materials_and_Methods`

This is a GE13-specific sanity check at `run_model.py:335`. It fires when you swap in other data or change the
`prior.json` filtering; delete that line if you use a different GE13 subset.

### 11.5 `CUDA out of memory`

Try in this order: `--train_batch_size 4` → `--max_len 384` → `--gradient_accumulation_steps 2`.
The dev/test batch sizes can be lowered as well.

### 11.6 `--do_train False` has no effect

The flags are declared as `type=bool`, and `bool("False") == True`. Disable a stage by passing an **empty value**:

```bash
python run_model.py --do_test=
```

(or edit the `default=` in the script).

### 11.7 CUDA unavailable / training on CPU although the GPU is idle

There is a typo at `run_model.py:49` (and `main.py:34`):

```python
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.seed)   # seed=42 → only GPU 42 is exposed, i.e. none
```

Change it to `str(args.gpu_id)` or delete the line (the following `torch.cuda.set_device(args.gpu_id)` already does the job).

### 11.8 Training is very slow / every epoch takes ages

Each epoch runs: full dev inference → writes 249 `*.a2` files → spawns two Perl processes. That is the bottleneck.
Call `merge(...)` only every 5 epochs, or select on dev loss and evaluate once at the end.

### 11.9 `FileNotFoundError: sources/GE13/std-train.json`

Generated by `sources/convert-tag-GE13.py` and not shipped. See [8.1](#81-prerequisite-generate-std-json),
or simply use the `run_model.py` pipeline.

### 11.10 `--task_name MLEE` → missing files

`sources/MLEE/` is empty and the MLEE conversion scripts are hard-coded to `../Joint-Task/datasets/MLEE`.
Only GE13 is supported out of the box.

### 11.11 Perl not found on Windows / garbled Chinese text

- `perl` must be on `PATH` (tick “Add to PATH” during the Strawberry Perl install); verify with `perl -v`.
- All sources are UTF-8; `鍚玞ause`-style garbage in the console is just the PowerShell code page (GBK) — harmless.
  Run `chcp 65001` if it bothers you.

### 11.12 Disk usage

| Directory | Size (measured) |
| --- | --- |
| `sources/` (`*.json` + `*.pt` + raw corpus) | ≈ 248 MB |
| `merge/` (mostly `merge.bin`) | ≈ 420 MB |
| `experiments/` (preprocessed caches) | ≈ 44 MB |
| `evaluation/` and the rest of the code | < 1 MB |
| `bert/scibert_scivocab_uncased/` (you download it) | ≈ 440 MB |
| **Total** | **≈ 0.7 GB (without SciBERT) / 1.15 GB (with)** |

Training additionally writes `experiments/GE13/best_model_42.bin` (≈ 440 MB per run).

---

## 12. Reference results

From the historical log `log/GE13.log` (Oct–Nov 2022, seed=42, devel):

```text
f1 (Overall): 68.0096%
f1 (Event):   56.8600%   ← that epoch
current best f1: 58.1900% (best epoch 11)   ← best dev event-level F1 observed
```

The CSBJ submission (UBEE-LMGCN) reports **58.83%** end-to-end F1 on GE13 **test** (and 64.11% on MLEE).
The 58.19% above is in the same range and can serve as a rough reproduction target
(note that it is not the same code/checkpoint).

---

## Appendix: where to change what

| I want to… | Look at / edit |
| --- | --- |
| change the data split | `sources/GE13/*.pt` (`run_model.py`) / `sources/GE13/std-*.json` (`main.py`) |
| change the label space | `experiments/GE13/schema.json` (delete it and `dataloader.build_map()` regenerates it) |
| change the event structural constraints | `sources/GE13/structure.json` |
| change trigger priors / filtering | `sources/GE13/prior.json` + `metrics.py::generation()` (MLEE blacklist lives inside the function) |
| change the evaluation protocol | the `a2-evaluate.pl` arguments at the end of `merge.py` (`-t1 -sp`) |
| plug in a graph network | `model/graph_reason.py` (requires `pip install dgl`), integrate it in `model/model_base.py` |
