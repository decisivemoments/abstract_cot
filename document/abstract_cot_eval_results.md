# Abstract-CoT Eval Results

## 1. 当前记录范围

这份文档用于持续记录项目中的 benchmark eval 结果。

当前已记录结果：

- baseline: `Qwen3-0.6B`
- warmup-only: `Qwen3-0.6B` + `mvp-warmup-8kc-1md`
- 数据集：
  - `GSM8K`
  - `MATH-500`
  - `AIME 2025`

---

## 2. 评测配置

### 2.1 Baseline

- run name: `qwen3_0.6b_baseline`
- model path: `server_assets/models/Qwen3-0.6B`
- tokenizer path: `server_assets/models/Qwen3-0.6B`
- dtype: `bfloat16`
- attention: `flash_attention_2`
- modes:
  - `direct-answer`
  - `natural-cot`

### 2.2 Warmup-only

- run name: `qwen3_0.6b_warmup`
- model path: `outputs/experiments/mvp-warmup-8kc-1md`
- tokenizer path: `outputs/experiments/mvp-warmup-8kc-1md`
- tokenizer artifacts:
  - `outputs/experiments/mvp-warmup-8kc-1md/tokenizer/abstract_tokenizer_artifacts.json`
- dtype: `bfloat16`
- attention: `flash_attention_2`
- mode:
  - `abstract-cot`

### 2.3 Benchmark 设置

所有 run 共用同一组 benchmark 设置：

| Dataset | Path | Split | Batch Size | Max Answer Tokens | Max Trace Length |
| --- | --- | --- | ---: | ---: | ---: |
| GSM8K | `server_assets/datasets/gsm8k` | `test` | 128 | 256 | 128 |
| MATH-500 | `server_assets/datasets/MATH-500` | `test` | 64 | 512 | 128 |
| AIME | `server_assets/datasets/aime_2025` | `train` | 64 | 512 | 128 |

---

## 3. Accuracy 对比

### 3.1 主结果

| Dataset | Baseline Direct | Baseline Natural-CoT | Warmup Abstract-CoT |
| --- | ---: | ---: | ---: |
| GSM8K | 0.3647 | 0.3078 | 0.3525 |
| MATH-500 | 0.2400 | 0.1520 | 0.2340 |
| AIME | 0.0333 | 0.0000 | 0.0000 |

### 3.2 样本数

| Dataset | Num Samples |
| --- | ---: |
| GSM8K | 1319 |
| MATH-500 | 500 |
| AIME | 30 |

### 3.3 有效答案抽取率

本轮所有设置的 `extraction_rate` 都是 `1.0`，说明当前答案抽取逻辑没有成为主瓶颈。

---

## 4. 详细结果

### 4.1 Qwen3-0.6B Baseline

| Dataset | Mode | Accuracy | Correct / Total | Avg Answer Chars |
| --- | --- | ---: | ---: | ---: |
| GSM8K | `direct-answer` | 0.3647 | 481 / 1319 | 843.59 |
| MATH-500 | `direct-answer` | 0.2400 | 120 / 500 | 1481.89 |
| AIME | `direct-answer` | 0.0333 | 1 / 30 | 1567.07 |
| GSM8K | `natural-cot` | 0.3078 | 406 / 1319 | 849.20 |
| MATH-500 | `natural-cot` | 0.1520 | 76 / 500 | 1492.37 |
| AIME | `natural-cot` | 0.0000 | 0 / 30 | 1644.73 |

### 4.2 Qwen3-0.6B Warmup-only Abstract-CoT

| Dataset | Mode | Accuracy | Correct / Total | Avg Answer Chars | Avg Trace Chars |
| --- | --- | ---: | ---: | ---: | ---: |
| GSM8K | `abstract-cot` | 0.3525 | 465 / 1319 | 587.21 | 180.57 |
| MATH-500 | `abstract-cot` | 0.2340 | 117 / 500 | 932.54 | 105.03 |
| AIME | `abstract-cot` | 0.0000 | 0 / 30 | 1105.00 | 122.87 |

---

## 5. 当前结论

这批结果的直接结论是：

1. `0.6B` 模型上，当前 warmup-only `abstract-cot` 没有明显超过 baseline `direct-answer`。
2. 在 `GSM8K` 和 `MATH-500` 上，warmup-only 结果都略低于 baseline `direct-answer`。
3. 在 `AIME` 上，`0.6B` 的整体表现都很弱，当前结果没有提供支持 abstract warm-up 有效的证据。

更具体地说：

- GSM8K:
  - baseline direct-answer: `0.3647`
  - warmup abstract-cot: `0.3525`
  - 差值: `-0.0121`

- MATH-500:
  - baseline direct-answer: `0.2400`
  - warmup abstract-cot: `0.2340`
  - 差值: `-0.0060`

- AIME:
  - baseline direct-answer: `0.0333`
  - warmup abstract-cot: `0.0000`
  - 差值: `-0.0333`

同时还可以看到一个现象：

- warmup abstract-cot 的答案输出长度明显短于 baseline
- 但输出更短本身并没有带来 accuracy 增益

---

## 6. 当前解释

这批结果更支持下面两种解释之一：

1. `0.6B` 模型容量过小，无法承载论文中的 abstract reasoning 机制。
2. 当前 warm-up 只用了 `8k` 以下的数据，训练覆盖面不足，导致 abstract-cot 没学到足够有效的表示。

这两点并不互斥。

因此，这组结果目前更适合作为：

- 工程链条已跑通的记录
- 后续更大模型和更全数据训练的 baseline 对照

而不适合作为论文复现成败的最终判断。

---

## 7. 下一步

基于这批结果，当前最合理的下一步是：

1. 统计 `Dolci-Think-SFT-7B` 全量长度分布。
2. 设计长样本场景下的 batch size / gradient accumulation 策略。
3. 推进 `4B` 及以上模型训练。
4. 再重复同一套 eval，对比：
   - baseline
   - warmup-only abstract-cot

只有在更大模型规模和更完整数据覆盖下，当前复现路径是否有效才有判断价值。
