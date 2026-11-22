# DMG-BEE 运行说明

> English version: **[README.md](README.md)**

> 面向 **BioNLP Shared Task 2013 GE (GE13)** 的生物医学事件抽取（Biomedical Event Extraction, BEE）实验工程。
> 代码为 Python + PyTorch 实现，主干流程：**SciBERT 编码 → 句级 span 全局指针打分（含 RoPE）→ 多标签交叉熵 → 事件融合 → GE13 官方 Perl 脚本评测**。
> 工程内另保留一条“触发词 → 论元”两阶段流水线和一套 GCN 图推理模块（见第 8、9 节）。

---

## 目录

1. [环境要求](#1-环境要求)
2. [目录结构](#2-目录结构)
3. [数据说明](#3-数据说明)
4. [快速开始（3 分钟跑通推理/评测链路）](#4-快速开始3-分钟跑通推理评测链路)
5. [完整训练流程：`run_model.py`](#5-完整训练流程run_modelpy)
6. [结果文件与单独复评](#6-结果文件与单独复评)
7. [测试集评测与在线提交](#7-测试集评测与在线提交)
8. [两阶段流水线：`main.py`](#8-两阶段流水线mainpy)
9. [事件融合模型（DualCL）：`merge/GE13/`](#9-事件融合模型dualclmergege13)
10. [一键复现命令清单](#10-一键复现命令清单)
11. [常见问题 FAQ](#11-常见问题-faq)
12. [参考结果](#12-参考结果)

---

## 1. 环境要求

| 项目 | 要求 | 说明 |
| --- | --- | --- |
| 操作系统 | Windows 10/11 或 Linux | 官方评测脚本为 Perl 脚本，跨平台可用 |
| Python | **3.8**（3.6–3.8 为原始开发版本） | 3.10+ 未验证；`__pycache__` 中可见 cpython-36/37/38 |
| PyTorch | ≥ 1.8（原始环境 **1.10.0 + CUDA 10.2**） | 代码中使用 `torch.optim.AdamW`、`pack_padded_sequence(enforce_sorted=False)` |
| transformers | 4.x（原环境约 4.6–4.11） | 只用到 `AutoTokenizer / AutoModel / AutoConfig / get_linear_schedule_with_warmup` |
| GPU | 显存 ≥ 8 GB（推荐 11 GB 以上） | SciBERT-base，`max_len=512`，训练 `batch_size=8` |
| **Perl** | 必须 | GE13 官方评测 `a2-normalize.pl` / `a2-evaluate.pl`，**训练时每个 epoch 都会调用**，缺失会直接报错（见 FAQ 11.3） |
| spacy + scispacy `en_core_sci_md` | 仅在“从原始语料重建数据集”时需要 | `sources/convert-doc-GE13.py` 需要句法依赖 |
| dgl | 可选 | 仅 `model/graph_reason.py`（MixHop/GCN）需要，当前入口脚本未引用 |

### 1.1 创建环境

```bash
# 建议先建虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

```bash
# 1) PyTorch：请按本机 CUDA 版本到 https://pytorch.org 取命令，例如 CUDA 11.8：
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 2) 其余依赖
pip install "transformers>=4.6,<4.40" tqdm numpy scikit-learn

# 3) 可选：重建数据集时才需要
pip install spacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_md-0.5.3.tar.gz
```

### 1.2 安装 Perl（必需）

| 平台 | 安装方式 |
| --- | --- |
| Windows | 安装 [Strawberry Perl](https://strawberryperl.com/)（或 `choco install strawberryperl`），安装后确认 `perl -v` 可用 |
| Ubuntu / Debian | `sudo apt-get install -y perl` |
| CentOS | `sudo yum install -y perl` |

> 验证：`perl -v` 能输出版本号即可。

### 1.3 放置预训练语言模型（必需）

仓库中的 `bert/scibert_scivocab_uncased/` 是**空目录**，必须自行下载 SciBERT 权重放进去：

```text
DMG-BEE/
└── bert/
    └── scibert_scivocab_uncased/
        ├── config.json
        ├── vocab.txt
        ├── pytorch_model.bin      (或 model.safetensors)
        └── tokenizer_config.json / tokenizer.json
```

- HuggingFace：<https://huggingface.co/allenai/scibert_scivocab_uncased>
- 命令行下载：
  ```bash
  pip install huggingface_hub
  huggingface-cli download allenai/scibert_scivocab_uncased --local-dir bert/scibert_scivocab_uncased
  ```
- 国内网络可改用镜像：`export HF_ENDPOINT=https://hf-mirror.com` 后再执行上面的命令。

> `main.py` / `run_model.py` 中的 `--bert_path` 默认即 `./bert/scibert_scivocab_uncased`，也可指向任意本地 BERT 目录（注意 `--bert_name` 会作为数据缓存文件名后缀）。

---

## 2. 目录结构

```text
DMG-BEE/
├── run_model.py            ★ 主入口：单阶段训练 / 验证 / 融合 / 评测（当前数据可直接跑）
├── main.py                   备选入口：两阶段（trigger → argument）流水线
├── model_base.py             run_model.py 使用的模型：MentionPointer（BERT + BiGRU + RoPE 全局指针）
├── dataloader.py             run_model.py 使用的数据处理（SpanDataloader / MyDataset / my_collate）
├── metrics.py                span 解码、触发器/论元/网络打分、事件合并（event_merge）
├── merge.py                  预测结果 → BioNLP *.a1/*.a2/*.txt，并调用官方 Perl 脚本评测
├── logger.py                 日志（写入 ./log/{task_name}.log）
├── binding_clean.py          工具：Binding 事件距离统计（生成 binding.pt，仅分析用）
├── BioNLP在线测试文档.docx    GE13 在线评测（上传 tar.gz）操作截图说明
├── filename.tar.gz           历史某次 test 预测打包（experiments/GE13/test 的 a1/a2/txt）
│
├── model/                    （main.py 使用）
│   ├── model_base.py         DMGModel：BERT + BiGRU + 3 个头（trigger/relation/sentence）
│   ├── graph_reason.py       GCN / Edgeupdate / MixHop（需要 dgl，当前未被引用）
│   └── loss_func.py          CELoss / SupConLoss / DualLoss（当前未被引用）
├── dataloader/dataloader.py  （main.py 使用）读取 sources/GE13/std-*.json
├── metrics/metrics.py        （main.py 使用）performance_report / generation
│
├── bert/                     SciBERT 权重目录（需自行下载，见 1.3）
├── log/GE13.log              历史训练日志（含各 epoch 指标）
│
├── sources/                  数据与数据转换脚本
│   ├── GE13/                 ★ 数据集（GE13 训练/验证/测试 + 各类索引文件）
│   ├── MLEE/                 （空）MLEE 数据集未随仓库提供
│   ├── CG13/                 另一套语料目录（内容为空，未使用）
│   ├── utils/                分词、分句、UMLS 过滤等工具
│   ├── convert-doc-GE13.py   原始 *.txt/*.a1/*.a2 → GE13/*.json（含 span/prior/tag_limits/structure）
│   ├── convert-tag-GE13.py   *.json → std-*.json（两阶段流水线用）
│   ├── convert-merge-GE13*.py *.json → *.pt（主流程用）/ merge-*.json（融合模型用）
│   └── convert-*-MLEE.py     MLEE 版本（路径写死 ../Joint-Task/datasets/MLEE，需自行改）
│
├── experiments/GE13/         训练产出目录（缓存张量、schema、预测 json）
│   ├── train-scibert_scivocab_uncased.pt   预处理缓存（13.8 MB）
│   ├── devel-scibert_scivocab_uncased.pt   预处理缓存（14.2 MB）
│   ├── test-scibert_scivocab_uncased.pt    预处理缓存（13.8 MB）
│   ├── schema.json          标签空间（15 类触发词、28 类关系、49 类 POS、47 类依存）
│   ├── structure.json       事件结构约束（论元角色合法性）
│   └── devel_*_42.json      历史上一次 devel 预测（触发器/论元/网络）
│
├── merge/GE13/              事件融合（DualCL）相关
│   ├── train_merge.py       训练融合模型，产出 merge.bin
│   ├── merge.bin            已训练好的融合模型（约 420 MB）
│   ├── merge_v1.py          规则融合：预测 json → *.a2
│   ├── data_merge.py        → predictions.json
│   ├── inference.py         推理占位脚本
│   └── pred-devel/          融合后的一组 devel 预测 *.a2（示例）
│
└── evaluation/              官方评测工具
    ├── GE13/evaluation.py   调用 Perl 脚本 → 端到端 F1
    ├── GE13/tools/*.pl      a2-normalize.pl / a2-evaluate.pl / a2-decompose.pl / a2d-evaluate.pl
    └── MLEE/evaluation-MLEE.py + test/（MLEE 测试集 a1/a2/txt）
```

---

## 3. 数据说明

### 3.1 GE13 目录内容

```text
sources/GE13/
├── train/  devel/  test/     原始语料：*.txt（正文）、*.a1（实体）、*.a2（事件标注，test 无）、*.json（官方 JSON）
├── train.json  devel.json  test.json    转换后的实例（JSON Lines，每行一个句子级实例）
├── train.pt   devel.pt   test.pt        ★ run_model.py 直接读取的 pickle 数据
├── std-train.json  std-devel.json  std-test.json    （需由 convert-tag-GE13.py 生成，供 main.py）
├── span-train.json  span-devel.json  span-test.json 文档 → 句子的 span 索引（删除跨句 span 用）
├── prior.json               触发器词典（876 个词 → 事件类型先验）
├── tag_limits.json          POS 词性 → id（49 类）
├── structure.json           事件类型 → 合法论元角色/类型（结构约束）
├── tokens.json / dist.json  词表与论元距离统计（当前代码未直接读取）
└── segment/*.jsonl          按段落切分的中间文件
```

| 划分 | 文档数 | 句子级实例数（`*.pt`） | 是否有金标 |
| --- | --- | --- | --- |
| train | 222 | **1198** | 有 |
| devel | 249 | **1318** | 有 |
| test | 305 | **1508** | **无**（需在线评测） |

> 实例数由 `pickle.load(sources/GE13/*.pt)` 实测。

### 3.2 标签空间（`experiments/GE13/schema.json`）

| 键 | 数量 | 含义 |
| --- | --- | --- |
| `lab` | 15 | 触发词/事件类型（含 `Entity`、`Anaphora`） |
| `rel` | 28 | 论元角色 + 嵌套事件关系（`Theme`、`Theme_rev`、`Theme_nested`、`Cause` 及各种事件类型） |
| `ent` | 3 | `O` / `B-Protein` / `I-Protein` |
| `tag` | 49 | POS 词性 |
| `dep` | 47 | 依存标签 |

### 3.3 （可选）从原始 BioNLP 文件重建数据集

> 仓库已提供转换好的 `*.json` / `*.pt`，**正常使用无需重建**；只有在更换语料或修改预处理逻辑时才需要。

```bash
# ① 原始 *.txt/*.a1/*.a2 → GE13/*.json + span-*.json + prior.json + tag_limits.json + structure.json + tokens.json
python convert-doc-GE13.py

# ② *.json → std-*.json（仅 main.py 两阶段流水线需要）
python convert-tag-GE13.py

# ③ *.json → *.pt（run_model.py 需要）
python convert-merge-GE13-v5.py

# ④ *.json → merge-train.json / merge-devel.json（仅融合模型 train_merge.py 需要）
python convert-merge-GE13.py
```

**注意（重要）：**

1. 上述转换脚本内部路径写成 `./GE13/...`，**必须在 `sources/` 目录下执行**；而 `convert-doc-GE13.py` 第 9 行是 `from sources.utils.tokenization_bert import BasicTokenizer`，要求以仓库根目录为 `sys.path`。两者冲突，二选一处理：
   - 方式 A（推荐）：把第 9 行改成 `from utils.tokenization_bert import BasicTokenizer`，然后在 `sources/` 下运行；
   - 方式 B：在仓库根目录运行，但输出会落在 `./GE13/`，需再移动回 `sources/GE13/`。
2. `convert-doc-GE13.py` 需要 `spacy.load("en_core_sci_md")`（scispaCy 模型），缺省会 `OSError`。
3. 脚本没有 `if __name__ == '__main__'`，**import 即执行**，不要被其他脚本 import。
4. 转换脚本各版本（`convert-merge-GE13.py` / `-v1...-v5` / `convert-tag-*`）是历史迭代版本，`v5` 产出当前 `*.pt` 格式；`convert-merge-GE13.py` 产出融合模型用的 `merge-*.json`。
5. `sources/MLEE/` 为空、`convert-doc-MLEE.py` 写死 `../Joint-Task/datasets/MLEE`，**MLEE 分支当前不可直接运行**。

---

## 4. 快速开始（3 分钟跑通推理/评测链路）

若只想验证环境、复现一次评测（**不需要 GPU、不需要训练**），用仓库中已有的 devel 预测结果跑融合 + 官方评测：

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

Windows PowerShell 不支持 heredoc，请把上面 `PY` 之间的内容存成 `check_eval.py` 再 `python check_eval.py`。

成功标志：

- 控制台打印 `a2-evaluate.pl` 的完整报告（含 `Approximate Span/Recursive` 表）；
- 生成 `experiments/GE13/devel/{a1,a2,txt}/` 三个目录（本次预测的 standoff 文件）；
- 最后打印 `end-to-end F1 = 0.5x–0.58`（该值来自仓库中已保存的这组 devel 预测，仅用于确认链路通畅）。

> 该步骤依赖 Perl。若 `perl` 不可用，`merge()` 会在解析评测输出时抛 `IndexError: list index out of range`。

---

## 5. 完整训练流程：`run_model.py`

### 5.1 一条命令

```bash
cd DMG-BEE
python run_model.py
```

默认配置（`--task_name GE13`、`--seed 42`、`--gpu_id 0`）会依次完成：

1. 读取 `sources/GE13/{train,devel,test}.pt`（无缓存时自动预处理并缓存到 `experiments/GE13/*.pt`）；
2. 构建/加载 `experiments/GE13/schema.json` 标签空间；
3. 训练 `MentionPointer`（SciBERT + BiGRU + RoPE 全局指针 + 多标签交叉熵）；
4. **每个 epoch** 在 devel 上解码 → `merge()` 生成 `*.a2` → 调 Perl 脚本算官方 F1；
5. 以「事件级 F1」为准保存最优模型 `experiments/GE13/best_model_42.bin`，并写出 devel 预测 json；
6. 若连续 `--early_stop`(默认 50) 个 epoch 无提升则提前结束。

日志同时输出到控制台和 `log/GE13.log`，关键行形如：

```text
devel metric
precision (Overall): 68.0096%
recall (Overall): ...
f1 (Overall): 68.0096%
f1 (Event): 56.8600%
current best f1: tri: 58.1900%
current best epoch: 11
```

### 5.2 参数说明（`run_model.py`）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--task_name` | `GE13` | 仅 `GE13` / `MLEE`；**MLEE 数据缺失，实际只能用 GE13** |
| `--seed` | `42` | 随机种子，同时用于最优模型文件名 `best_model_{seed}.bin` |
| `--train_path` / `--devel_path` / `--test_path` | `sources/GE13/{train,devel,test}.pt` | pickle 格式实例 |
| `--prior_path` | `sources/GE13/prior.json` | 触发器先验词典 |
| `--output_path` | `experiments/GE13` | 缓存与预测输出目录 |
| `--bert_path` | `bert/scibert_scivocab_uncased` | 预训练模型目录 |
| `--max_len` | `512` | 子词最大长度，超出截断 |
| `--inner_dim` | `64` | 全局指针每个标签的向量维度 |
| `--ent_dim` | `50` | 实体类型 embedding 维度 |
| `--RoPE` | `True` | 是否启用旋转位置编码 |
| `--drop_rate` | `0.5` | dropout |
| `--lr` | `5e-5` | 学习率（AdamW） |
| `--epoch` | `300` | 最大 epoch（实际由 early stop 决定） |
| `--early_stop` | `50` | 无提升轮数上限 |
| `--max_grad_norm` | `1.0` | 梯度裁剪 |
| `--train_batch_size` | `8` | 显存不足时优先调小 |
| `--devel_batch_size` / `--test_batch_size` | `32` | 推理 batch |
| `--gpu_id` | `0` | 使用第几张卡 |
| `--do_train` / `--do_devel` / `--do_test` | `True` | 开关（**注意布尔坑，见 FAQ 11.6**） |

> `--stage`、`--fix_bert_embedding` 在 `run_model.py` 中保留但未被主流程使用（`--stage` 是两阶段流水线的参数）。

### 5.3 调参/提速建议

| 目标 | 做法 |
| --- | --- |
| 快速验证能否跑通 | `python run_model.py --epoch 2 --early_stop 2` |
| 减少显存占用 | `--train_batch_size 4 --max_len 384` |
| 缩短每轮时间 | 注释 `run_model.py` 第 258 行 `merge(...)` 调用（改用验证集 loss 选模型），或改成每 5 个 epoch 评测一次 |
| 换卡 | `--gpu_id 1`；并建议把 `run_model.py` 第 49 行 `os.environ['CUDA_VISIBLE_DEVICES'] = str(args.seed)` 改成 `str(args.gpu_id)`（原写法是笔误，见 FAQ 11.7） |

---

## 6. 结果文件与单独复评

训练/预测完成后，`experiments/GE13/` 下的关键文件：

| 文件 | 内容 |
| --- | --- |
| `best_model_42.bin` | devel 事件级 F1 最优的模型权重（`seed=42`） |
| `devel_trigger_p_42.json` / `_g_42.json` | devel 触发器 预测 / 金标（文档 → span 集合） |
| `devel_relation_p_42.json` / `_g_42.json` | devel 论元关系 预测 / 金标 |
| `devel_network_p_42.json` / `_g_42.json` | devel 嵌套事件链 预测 / 金标 |
| `devel_entity_g_42.json` | devel 实体（金标，用于生成 a1） |
| `devel/{a1,a2,txt}/` | 融合后的 BioNLP standoff 文件（交给官方评测） |
| `schema.json` / `structure.json` | 标签空间与结构约束 |

**只做融合 + 评测（不重新训练）**：

```bash
# 方式一：把带 seed 后缀的预测复制成 merge.py __main__ 期望的文件名，然后运行
#   devel_trigger_p_42.json  → devel_trigger_p.json
#   devel_relation_p_42.json → devel_relation_p.json
#   devel_network_p_42.json  → devel_network_p.json
#   devel_entity_g_42.json   → devel_entity_g.json
python merge.py

# 方式二：直接调用 merge()（推荐，见第 4 节脚本），可自由指定 mode="devel"/"test"
```

`merge()` 逐文档做的事：复制 `*.txt` + 金标 `*.a2` 到 `a1/`，按预测写出实体 `*.a1` 与事件 `*.a2`，然后执行：

```bash
./evaluation/GE13/tools/a2-normalize.pl -g ./sources/GE13/devel/ -u ./experiments/GE13/devel/a2/*.a2
./evaluation/GE13/tools/a2-evaluate.pl  -g ./sources/GE13/devel/ -t1 -sp ./experiments/GE13/devel/a2/*.a2
```

> 因此 **`merge.py` 必须在仓库根目录（`DMG-BEE/`）下运行**，因为它用 `./sources/...`、`./evaluation/...` 相对路径。

---

## 7. 测试集评测与在线提交

GE13 的 test 集**不提供 `*.a2` 金标**，因此本地只能产出预测、不能算分，需要上传到 BioNLP 官方评测服务（流程见 `BioNLP在线测试文档.docx`）：

1. **生成 test 预测**：把 `run_model.py` 末尾改成

   ```python
   Pipeline = PipeLineTrainer(args=args, logger=exp_logger)
   # Pipeline.run_train()
   Pipeline.prediction(Pipeline.test_data, mode="test", source_path="./sources/GE13/test")
   ```

   运行后会：加载 `experiments/GE13/best_model_42.bin` → 解码 → 写 `test_*_42.json` → `merge(mode="test")` 生成
   `experiments/GE13/test/{a1,a2,txt}/`，并自动打包 `filename.tar.gz`。

2. **检查压缩包结构**：根目录直接是 `*.a1/*.a2/*.txt`（不含 gold、不含多余层级）。

   ```bash
   tar -tzf filename.tar.gz | head
   # ./experiments/GE13/test/a1/PMC-....a1 ...
   ```

   ⚠ 若需要“扁平”的包（部分评测要求 `tar -zcvf filename.tar.gz *` 在 a2 所在目录执行），请进入 `experiments/GE13/test/a2/` 再打包：
   ```bash
   cd experiments/GE13/test/a2 && tar -zcvf ../../../../filename.tar.gz *
   ```

3. **申请评测权限**：向 `majordomo@bionlp-st.org` 发送邮件，正文 `subscribe bionlp-st`，按回复邮件确认邮箱已加入评测服务器。

4. **上传**：打开 <http://bionlp-st.dbcls.jp/GE/2011/eval-development/>（GE13 评测入口），填入邮箱、上传 `tar.gz`，查看结果。

5. **看哪一行成绩**：结果表第二张表 **Approximate Span / Recursive** 的 F1 即论文对比口径（本地 `a2-evaluate.pl -t1 -sp` 与之对应）。

---

## 8. 两阶段流水线：`main.py`

`main.py` 是早期“**先抽触发词、再分类论元**”的两阶段实现，模型为 `model/model_base.py::DMGModel`
（SciBERT → BiGRU → LayerNorm → 3 个分类头：`tri`(15 类)、`rel`(28 类)、`lab`(句级 2 类)），
数据来自 `sources/GE13/std-*.json`。

### 8.1 前置：生成 std-*.json

```bash
cd sources
python convert-tag-GE13.py     # 产出 GE13/std-train.json / std-devel.json / std-test.json
```

### 8.2 训练命令

```bash
cd DMG-BEE

# 第一阶段：触发词标注
python main.py --stage trigger --task_name GE13 --lr 1e-5 --train_batch_size 16 --epoch 300
# 产出 experiments/GE13/best_trigger_model.bin

# 第二阶段：论元分类（自动加载第一阶段的模型与 devel/test 触发器 logits）
python main.py --stage argument --task_name GE13 --lr 1e-5 --train_batch_size 16 --epoch 300
# 产出 experiments/GE13/best_argument_model.bin
```

> 注意：`main.py` 末尾默认执行 `Pipeline.prediction()`（需要 `best_argument_model.bin`），
> 因此**没训练过时直接 `python main.py` 会报文件不存在**；首次使用请先把最后两行改成
> `Pipeline.run_train()`，或按 8.3 打补丁后用参数控制。

### 8.3 已知缺件（该分支为未完成状态）

`main.py` 这条流水线在仓库中是**半成品**，直接运行会遇到下列问题，需按需补 3 处：

| 位置 | 现象 | 修复 |
| --- | --- | --- |
| `main.py:13` `from dataloader.dataloader import Dataloader` | `ImportError`（模块里只有 `SpanDataloader`） | 改成 `from dataloader.dataloader import SpanDataloader as Dataloader` |
| `model/model_base.py:19` 使用 `args.edge_dim` | `AttributeError: 'Namespace' object has no attribute 'edge_dim'` | 在 `main.py` 的 argparse 中补 `parser.add_argument('--edge_dim', type=int, default=50)` |
| `metrics/metrics.py:22` 读 `output["lab"]["predicts"]` | `TypeError`（`DMGModel` 返回的是张量 `logits`） | 改成 `torch.argmax(output["lab"], dim=-1).tolist()` |
| 参数缺失 | 历史日志中出现过 `--num_layer`、`--gnn_type`，当前 `main.py` 未定义 | 仅在使用图模块时才需要补 |

补齐后即可用 `--stage trigger` → `--stage argument` 顺序训练；其输出文件名（`devel_trigger_pred.json`、`devel_relation_pred.json`）正是第 9 节融合模型所读取的。

---

## 9. 事件融合模型（DualCL）：`merge/GE13/`

当触发器/论元预测需要进一步“合并成完整事件”（尤其是嵌套事件）时，使用基于对比学习的融合分类器：

```bash
cd merge/GE13

# ① 训练融合模型（数据来自 sources/GE13/merge-train.json / merge-devel.json）
python train_merge.py --data_dir ../../sources/GE13 --model_name ../../bert/scibert_scivocab_uncased
# 产出 ./merge.bin（仓库已提供一份训练好的 420 MB 权重，可直接跳过本步）

# ② 训练/推理后处理：把预测 json 合成为 BioNLP *.a1/*.a2
python merge_v1.py        # 读取 ../../experiments/GE13/devel_trigger_pred.json 等，写入 devel-pred/
python data_merge.py      # 汇总为 predictions.json
python analysis.py        # 与 ../../sources/GE13/merge-devel.json 对比统计

# ③ 单独评测
cd ../../evaluation/GE13
python evaluation.py      # 调用 ../../sources/GE13/devel 与 ../../experiments/GE13/devel-pred 对比
```

| 脚本 | 输入 | 输出 |
| --- | --- | --- |
| `train_merge.py` | `sources/GE13/merge-train.json`、`merge-devel.json`；`--method dualcl/ce/scl` | `merge/GE13/merge.bin`、`log/dualcl.log` |
| `merge_v1.py` | `sources/GE13/devel/`、`experiments/GE13/devel_{trigger,relation}_pred.json` | `experiments/GE13/devel-pred/*.a2` |
| `data_merge.py` | `experiments/GE13/devel_relation_pred.json`、`sources/GE13/devel.json` | `merge/GE13/predictions.json` |
| `inference.py` | `predictions.json` | 演示用占位脚本（12 行） |

> ⚠ 路径注意：这些脚本的默认相对路径假定**当前目录为 `merge/GE13/`**，且预测文件名**不带 seed 后缀**
> （即 `devel_trigger_pred.json`，由 `main.py` 产出）。若用 `run_model.py` 的 `*_p_42.json`，请先改名或修改脚本内路径。

---

## 10. 一键复现命令清单

以下为从零到出分的完整命令（Linux/macOS bash；Windows 用 PowerShell 逐行执行，注意路径分隔符）：

```bash
# 0) 环境
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install "transformers>=4.6,<4.40" tqdm numpy scikit-learn
sudo apt-get install -y perl
huggingface-cli download allenai/scibert_scivocab_uncased --local-dir bert/scibert_scivocab_uncased

# 1) 进入工程根目录（所有相对路径的基准）
cd DMG-BEE

# 2) 自检：CUDA 与 Perl
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
perl -v | head -2

# 3) 训练 + 验证（每 epoch 自动融合并跑官方评测）
python run_model.py --gpu_id 0

# 4) 单独复评（先把 *_p_42.json 复制成 *_p.json，或直接运行第 4 节的调用脚本）
python merge.py

# 5) 测试集推理 + 打包（改 run_model.py 末尾为 Pipeline.prediction(...)，见第 7 节）
#    → experiments/GE13/test/{a1,a2,txt}/ 与 filename.tar.gz
```

---

## 11. 常见问题 FAQ

### 11.1 `ModuleNotFoundError: No module named 'transformers'`

依赖未装或装在了别的解释器里：确认 `which python` / `where python` 与 `pip -V` 指向同一环境，然后重装。

### 11.2 `OSError: Error no file named pytorch_model.bin ... scibert_scivocab_uncased`

`bert/scibert_scivocab_uncased/` 是空目录，按 [1.3](#13-放置预训练语言模型必需) 下载权重。

### 11.3 训练第 1 个 epoch 结束就 `IndexError: list index out of range`

`merge.py` 在解析 `a2-evaluate.pl` 的输出，而 **Perl 不可用/脚本执行失败** → 输出为空。
解决：装 Perl（见 1.2），或临时注释 `run_model.py` 第 258 行的 `merge(...)` 调用并改用 loss 选模型。

### 11.4 `AssertionError: PMC-2222968-21-Materials_and_Methods`

`run_model.py:335` 是 GE13 专用断言，检查该 devel 文档是否出现在实体集合里。
如果替换了数据集/改动了 `prior.json` 过滤，会触发该断言——换用其它 GE13 数据时删掉这一行即可。

### 11.5 `CUDA out of memory`

依次尝试：`--train_batch_size 4` → `--max_len 384` → `--gradient_accumulation_steps 2`。
`devel/test` 的 `batch_size` 也可降到 8。

### 11.6 `--do_train False` 不生效（开关无效）

argparse 里写的是 `type=bool`，`bool("False") == True`。
关闭某个阶段请用**空值**形式：`python run_model.py --do_test=`（等号后留空），或直接改脚本里的 `default=`。

### 11.7 GPU 明明空闲却报 CUDA 不可用 / 只用了 CPU

`run_model.py:49`（`main.py:34`）有笔误：

```python
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.seed)   # seed=42 → 只暴露 42 号卡（等于没有）
```

改成 `str(args.gpu_id)`，或删掉这一行（后面已有 `torch.cuda.set_device(args.gpu_id)`）。

### 11.8 训练非常慢 / 每个 epoch 都要等很久

每个 epoch 都会做：devel 全量推理 → 生成 249 个 `*.a2` → 启动 2 个 Perl 进程评测。这是“慢”的主因。
优化：把 `merge(...)` 改成每 5 个 epoch 调一次，或先用 loss 粗选、训练结束再统一评测。

### 11.9 `FileNotFoundError: sources/GE13/std-train.json`

该文件由 `sources/convert-tag-GE13.py` 生成，仓库默认没带。按 [8.1](#81-前置生成-std-json) 生成，或改用 `run_model.py` 主流程。

### 11.10 `--task_name MLEE` 报文件不存在

`sources/MLEE/` 为空，MLEE 版本转换脚本还写死了 `../Joint-Task/datasets/MLEE` 路径。当前工程实际只支持 GE13。

### 11.11 Windows 下 Perl 脚本找不到 / 中文乱码

- `perl` 必须在 `PATH` 中（Strawberry Perl 安装时勾选“Add to PATH”），命令行执行 `perl -v` 自检。
- 源码文件均为 UTF-8；控制台出现 `鍚玞ause` 之类是 PowerShell 代码页（GBK）显示问题，不影响运行。需要时执行 `chcp 65001`。

### 11.12 磁盘占用

| 目录 | 大小（实测） |
| --- | --- |
| `sources/`（`*.json` + `*.pt` + 原始语料） | ≈ 248 MB |
| `merge/`（主要是 `merge.bin`） | ≈ 420 MB |
| `experiments/`（预处理缓存） | ≈ 44 MB |
| `evaluation/`、其余代码 | < 1 MB |
| `bert/scibert_scivocab_uncased/`（需自行下载） | ≈ 440 MB |
| **合计** | **≈ 0.7 GB（不含 SciBERT 权重）/ 1.15 GB（含）** |

训练还会在 `experiments/GE13/` 生成 `best_model_42.bin`（每份约 440 MB）。

---

## 12. 参考结果

历史日志 `log/GE13.log`（2022-10 ~ 2022-11，seed=42，devel）：

```text
f1 (Overall): 68.0096%
f1 (Event):   56.8600%   ← 该轮
current best f1: 58.1900%（best epoch 11）   ← devel 上最好的一次事件级 F1
```

论文（CSBJ 投稿的 UBEE-LMGCN）报告的 GE13 **test** 端到端 F1 为 **58.83%**（MLEE 为 64.11%）；
与本仓库 devel 上的 58.19% 属同一量级，可作为“复现是否成功”的粗略参照（注意二者并非同一套代码/权重）。

---

## 附：常用文件速查

| 我想…… | 看/改这里 |
| --- | --- |
| 换数据集划分 | `sources/GE13/*.pt`（`run_model.py`） / `sources/GE13/std-*.json`（`main.py`） |
| 改标签空间 | `experiments/GE13/schema.json`（删掉会由 `dataloader.build_map()` 重新生成） |
| 改事件结构约束 | `sources/GE13/structure.json` |
| 改触发器先验过滤 | `sources/GE13/prior.json` + `metrics.py::generation()`（MLEE 专用黑名单在函数内） |
| 改评测口径 | `merge.py` 末尾的 `a2-evaluate.pl` 参数（`-t1 -sp`） |
| 加图神经网络 | `model/graph_reason.py`（需 `pip install dgl`），在 `model/model_base.py` 中接入 |
