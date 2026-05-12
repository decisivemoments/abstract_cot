# Abstract-CoT 评测设计

## 1. 目标与范围

本设计文档只覆盖当前最需要的评测问题：

1. 对已经完成 warm-up 的模型做 benchmark eval
2. 只比较 `baseline` 和 `warmup-only abstract CoT`
3. 不纳入 RL、DPO、continuous latent、额外蒸馏变体

这里的核心问题不是“做多少实验”，而是先把一条可复现、可扩展、可汇总的评测链搭起来。

---

## 2. 当前评测对象

当前评测对象限定为两类模型状态：

1. `baseline`
   - 未经过 Abstract-CoT warm-up 的原始模型
   - 例如 `Qwen3-4B` 或其他后续选定基座

2. `warmup-only`
   - 只做过当前 Stage 1 warm-up 的模型
   - 不做 RL、不做额外后训练

评测结论只回答一个最基本的问题：

> 同一基座模型在做了 warm-up only 的 Abstract-CoT 后，是否相对 baseline 有稳定增益。

---

## 3. 评测模式

为了让结果尽可能可解释，先固定成两种推理模式。

### 3.1 Baseline 模式

baseline 使用普通 CausalLM 推理，不要求输出 abstract trace。

建议先实现两个 baseline prompt 变体：

1. `direct-answer`
   - 直接要求模型输出最终答案

2. `natural-cot`
   - 允许模型先思考再回答
   - 但不使用 abstract token，也不使用 constrained decoding

这样做的原因：

- `direct-answer` 是最简单基线
- `natural-cot` 是更合理的 reasoning 基线
- warmup-only 的增益不能只和最弱 prompt 比

### 3.2 Warmup-only Abstract-CoT 模式

warmup-only 模式按当前训练设计对应的推理流程执行：

1. 给定 `X`
2. 用 constrained decoding 生成 abstract trace `Z`
3. 再让模型基于 `X + Z` 生成最终答案 `Y`

是否让推理时额外暴露 verbal CoT `C`，这里先明确为：

- **eval 时默认不提供 `C`**

原因：

- distill 阶段本身就是 prompt-only trace
- 真正测试时通常也拿不到参考 verbal CoT
- 如果把 `C` 作为输入，任务定义会偏离实际 inference 场景

---

## 4. 第一版 benchmark 选择

第一版只选标准、公开、回答可判定的数据集，优先数学与符号推理任务。

建议第一批：

1. `GSM8K`
2. `MATH-500`
3. `AIME 2024/2025` 或项目环境中更容易拿到的 AIME 子集

如果需要再补一个非纯数学集合，可考虑第二批再加：

4. `GPQA`
5. `MMLU-Pro` 或 `MUSR` 一类更广泛的推理集

第一版不建议一开始就铺太多数据集。原因是：

- 先做通数学推理链最接近当前 warm-up 数据分布
- 数据适配、答案抽取、评分规则都需要分别实现
- benchmark 数量太多会放大工程噪音

数据管理约束：

- eval 数据集与训练数据一样，默认先下载到本地 `server_assets/datasets/...`
- eval 运行时优先读取本地目录
- 不把在线 Hugging Face 拉取作为默认运行方式

---

## 5. 数据集格式与评分要点

这一节只写第一版实现真正需要知道的信息，即：

- 输入长什么样
- 目标答案长什么样
- 最终怎么判分

### 5.1 GSM8K

任务特点：

- 小学到中学水平的数学文字题
- 每条样本通常包含一个 `question`
- 参考答案通常带有推导文本和最终答案

实现时建议统一抽成：

```python
{
  "sample_id": str,
  "question": str,
  "target_answer": str,
  "metadata": {...}
}
```

评分：

- 只看最终数值答案是否匹配
- 优先做数字规范化后再比较
- 不要求生成过程逐字一致

注意点：

- 参考答案里常常有解释文本
- 需要从 gold 中提取最终答案字段
- 模型输出也需要做答案抽取

### 5.2 MATH-500

任务特点：

- 更难的数学题
- 题目通常是自然语言或 LaTeX 风格数学文本
- 标准答案通常是短答案，但可能有格式变化

统一字段建议：

```python
{
  "sample_id": str,
  "question": str,
  "target_answer": str,
  "subject": str | None,
  "level": str | None,
}
```

评分：

- 第一版仍采用 answer matching
- 需要做比 GSM8K 更严格的标准化
- 包括去空格、统一 LaTeX 包裹、去掉多余文本前后缀

注意点：

- MATH 类数据答案格式波动大
- 第一版不要追求完全符号等价证明器
- 先做可控的 normalization + exact match

### 5.3 AIME

任务特点：

- 竞赛数学
- 最终答案通常是 `0-999` 间整数
- 题目数量小，但区分度高

统一字段建议：

```python
{
  "sample_id": str,
  "question": str,
  "target_answer": str,
  "year": int | None,
}
```

评分：

- 最终整数答案 exact match

注意点：

- 输出抽取相对简单
- 但推理难度高，比较适合作为 warmup-only 是否真有用的强信号集

---

## 6. 第一版统一数据接口

评测框架里建议先统一成一个轻量 schema，不直接暴露底层数据集各自字段。

建议的内部样本格式：

```python
@dataclass(frozen=True)
class EvalSample:
    sample_id: str
    dataset_name: str
    prompt: str
    target_answer: str
    metadata: dict[str, Any]
```

每个 benchmark adapter 负责：

1. 加载原始数据
2. 转成 `EvalSample`
3. 提供 dataset-specific 的答案抽取和评分函数

---

## 7. 推理输出接口

为了让 baseline 和 warmup-only 用同一套汇总逻辑，推理输出也应统一。

建议：

```python
@dataclass(frozen=True)
class EvalPrediction:
    sample_id: str
    dataset_name: str
    mode: str
    prompt_text: str
    generated_trace_text: str | None
    generated_answer_text: str
    extracted_answer: str | None
    target_answer: str
    is_correct: bool
    metadata: dict[str, Any]
```

说明：

- `mode` 至少包含 `direct-answer`、`natural-cot`、`abstract-cot`
- `generated_trace_text` 对 baseline 为 `None`
- `generated_trace_text` 对 abstract-cot 则保存生成的 `Z`

---

## 8. Prompt 设计

第一版 prompt 目标是可复现，而不是追求 prompt engineering 极限。

### 8.1 direct-answer

```text
Solve the following problem and give the final answer only.

{question}
```

### 8.2 natural-cot

```text
Solve the following problem step by step, then give the final answer clearly.

{question}
```

### 8.3 abstract-cot

abstract-cot 不直接把“思考过程”暴露成自然语言，而是：

1. 对 `question` 调用 constrained abstract decoding
2. 拿到 `Z`
3. 再把 `question + Z` 送入 answer generation prompt

第二阶段 answer prompt 建议固定成：

```text
Question:
{question}

Abstract reasoning:
{abstract_trace}

Answer:
```

---

## 9. 指标

第一版先只做最硬的主指标，不上复杂分析指标。

### 9.1 主指标

每个数据集都至少输出：

1. `accuracy`
2. `num_samples`
3. `num_valid_predictions`

其中：

- `accuracy = correct / total`
- `num_valid_predictions` 表示成功抽取到最终答案的样本数

### 9.2 辅助统计

建议顺手记录，但不作为第一优先：

1. 平均输出 token 长度
2. 平均 abstract trace 长度
3. 抽取失败率
4. 空答案率

这些统计对后续分析很有用，尤其是判断 warm-up 后模型到底是在：

- 真提升了推理
- 还是只是更爱输出长文本

---

## 10. 输出产物

每次 eval run 建议产出三类文件。

### 10.1 配置快照

例如：

- `eval_config.json`

记录：

- model path
- tokenizer path
- benchmark 列表
- prompt mode
- decoding 参数
- max samples
- timestamp

### 10.2 样本级预测

例如：

- `predictions.jsonl`

每条包含：

- 输入样本
- 原始生成文本
- trace
- 抽取答案
- gold
- 是否正确

### 10.3 汇总结果

例如：

- `metrics.json`
- `summary.md`

其中 `summary.md` 方便直接人工查看，例如：

```text
model: Qwen3-4B-warmup
mode: abstract-cot

GSM8K: 63.4
MATH-500: 21.8
AIME24: 7.5
```

---

## 11. 建议的代码结构

第一版不需要太重，但目录建议清晰。

建议新增：

```text
abstract_cot/src/abstract_cot/eval/
  schema.py
  runner.py
  prompts.py
  answer_extraction.py
  metrics.py
  report.py
  generation.py
  adapters/
    gsm8k.py
    math500.py
    aime.py
```

以及脚本入口：

```text
abstract_cot/scripts/run_eval.py
```

职责建议如下：

- `schema.py`
  - `EvalSample`, `EvalPrediction`, `EvalResult`

- `runner.py`
  - 数据集循环、batch 推理、汇总调度

- `generation.py`
  - baseline generation
  - abstract trace generation
  - answer generation

- `answer_extraction.py`
  - 从模型输出中抽取最终答案

- `metrics.py`
  - correctness 判断和聚合

- `report.py`
  - 写 `json/jsonl/md`

- `adapters/*.py`
  - 各数据集加载与字段映射

---

## 12. 第一版实现策略

实现顺序建议固定，不要一开始就铺开所有功能。

### Step 1

先做单数据集、单模式跑通：

- 数据集：`GSM8K`
- 模式：`direct-answer`

目标：

- 跑通数据加载
- 跑通生成
- 跑通答案抽取
- 跑通 accuracy 汇总

### Step 2

在同一套 GSM8K 上加：

- `natural-cot`
- `abstract-cot`

目标：

- 跑通三种 mode 的统一输出
- 确认 trace 记录逻辑没问题

### Step 3

扩展到：

- `MATH-500`
- `AIME`

目标：

- 补齐 adapter
- 校准 normalization / answer extraction

### Step 4

增加：

- 多 benchmark 汇总表
- 多模型对比
- 多 run 结果归档

---

## 13. 当前明确不做的事情

第一版 eval 文档明确排除：

1. RL 后模型评测
2. 多种训练变体的大矩阵 sweep
3. 复杂 judge model 评分
4. theorem prover / symbolic solver 打分
5. trace 质量的人类偏好标注

这些都可以后续再加，但不该阻塞第一版评测链。

---

## 14. 当前结论

当前最合理的评测落地方向是：

1. 先只评 `baseline` vs `warmup-only`
2. 先只评最有代表性的 reasoning benchmark
3. 先统一数据、推理输出、答案抽取、指标汇总四层接口
4. 先把 `GSM8K -> MATH-500 -> AIME` 这条最小链跑通

这样做的好处是：

- 工程范围可控
- 结果解释清晰
- 后面无论接 RL、analysis 还是更多模型规模，都会复用这套 eval 骨架
