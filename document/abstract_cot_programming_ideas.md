# Abstract-CoT 编程 Idea 指导文档

> 来源文章：*Thinking Without Words: Efficient Latent Reasoning with Abstract Chain-of-Thought*  
> 目的：将论文中的方法、训练流程、实验观察和可转化为编程任务的 idea 进行整理，作为下一步实现或复现实验的方向参考。本文不包含具体系统设计，只提取可实现的核心 idea。

---

## 1. 核心问题与目标

传统 Chain-of-Thought（CoT）通过生成自然语言推理过程提升复杂任务表现，但会带来显著的推理 token 成本、延迟和训练阶段轨迹膨胀。文章提出的问题是：

**是否可以用一段短的、非自然语言的离散抽象 token 序列，替代长篇自然语言 CoT，并仍然保留 CoT 对答案生成的帮助？**

论文答案是可以。方法称为 **Abstract Chain-of-Thought（Abstract-CoT）**。

其目标包括：

- 用一组新增的 reserved abstract tokens 作为“抽象推理语言”。
- 在回答前生成短序列抽象 token，而不是自然语言推理文本。
- 通过后训练让模型学会使用这些原本随机初始化、无语义的 token。
- 在推理时减少 reasoning tokens，同时尽量维持或接近 verbal CoT + RL 的性能。
- 保留一个可控、可分析的中间推理段，而不是完全隐式推理。

---

## 2. 方法总体思路

Abstract-CoT 位于两个极端之间：

- **显式自然语言 CoT**：可读、可解释，但 token 成本高。
- **完全隐式推理**：没有额外文本成本，但难以控制和分析。

Abstract-CoT 的中间路线是：

1. 扩展 tokenizer，加入一组新的抽象 token。
2. 让模型在最终回答前生成：

```text
<beginabstract> <TOKEN_A> <TOKEN_X> ... <TOKEN_K> <endabstract>
```

3. 抽象 token 本身不对应自然语言含义，但通过训练学会携带推理信息。
4. 最终答案只依赖 prompt 和 abstract trace，而不是依赖完整自然语言 CoT。

---

## 3. 基本符号与数据形式

文章中的基本变量：

| 符号 | 含义 |
|---|---|
| `x` | 用户 prompt / 问题 |
| `c` | gold verbal CoT，自然语言推理链 |
| `y` | gold answer / 目标答案 |
| `z` | 抽象 token 序列 |
| `V` | 原始模型词表 |
| `V_abs` | 新增抽象 token codebook |
| `m` | 抽象 token 序列长度 |
| `m_max` | 最大抽象 token 长度 |
| `π_θ` | 语言模型策略 |
| `π^abs_θ` | 被限制在抽象 token 集合上的生成策略 |

训练数据形式：

```text
D = {(x_i, c_i, y_i)}
```

其中 `c_i` 只在 warm-up 的第一类训练阶段中使用；最终推理阶段没有 `c`。

抽象 trace 形式：

```text
z_tilde = <beginabstract> z_1 z_2 ... z_m <endabstract>
```

其中：

```text
z_i ∈ V_abs
m <= m_max
```

---

## 4. 抽象词表与 tokenizer 扩展

### 4.1 抽象 token codebook

文章使用一组新 token，例如：

```text
<TOKEN_A>, <TOKEN_B>, ..., <TOKEN_Z>, <TOKEN_AA>, ...
```

当 token 数量超过 26 时，继续使用双字母形式，如 `<TOKEN_AA>` 到 `<TOKEN_ZZ>`。

### 4.2 分隔符

额外加入两个边界 token：

```text
<beginabstract>
<endabstract>
```

它们用于标记抽象推理段开始和结束。

### 4.3 新 token 的冷启动问题

新增 token 的 embedding 初始是随机的，没有语义。模型一开始无法使用这些 token 表达有用推理信息。

因此，论文的关键不是“加 token”本身，而是设计一个训练流程，让这些 token 的 embedding 和生成策略逐步变得有用。

---

## 5. 推理时的生成流程

推理阶段只给模型输入 prompt `x`。

模型需要先生成抽象推理段，再生成最终答案：

```text
输入：x
生成：<beginabstract> z_1 ... z_m <endabstract>
生成：y
```

其中抽象段生成受到约束：

- `<beginabstract>` 后只能生成：
  - `V_abs` 中的抽象 token
  - `<endabstract>`
- 抽象 token 数不能超过 `m_max`
- 如果达到 `m_max`，强制生成 `<endabstract>`
- `<endabstract>` 后恢复普通 unconstrained decoding，用原始词表生成答案

---

## 6. 约束解码 idea

文章定义一个限制后的策略 `π^abs_θ`，只允许在抽象 token 集合内采样。

允许集合：

```text
A = V_abs ∪ {<endabstract>}
```

在每一步抽象生成中，对原模型概率进行 mask 和重新归一化：

```text
π^abs_θ(a | h) = π_θ(a | h) * 1[a ∈ A] / sum_{u ∈ A} π_θ(u | h)
```

其中上下文为：

```text
h = x ∪ {<beginabstract>} ∪ previous abstract tokens
```

可实现的 idea：

- 增加 constrained decoding wrapper。
- 在 abstract mode 下只保留 abstract vocab 和 end delimiter 的 logits。
- 记录抽象 token 长度。
- 到达 `m_max` 时强制结束抽象段。
- 结束后切换回普通 generation mode。

---

## 7. 两阶段训练总览

Abstract-CoT 的训练包含两大阶段：

```text
Stage 1: Policy Iteration Warm-Up
  1. Bottlenecked SFT with Abstract Tokens
  2. Self-Distillation Without Verbal CoT
  以上两步重复 T 轮

Stage 2: Warm-Started RL
  使用 GRPO + constrained decoding 优化 abstract trace 和 answer
```

核心逻辑：

- Warm-up 负责解决新增 token embedding 的冷启动问题。
- Self-distillation 负责让模型从 prompt alone 直接生成 abstract trace。
- RL 负责探索更优抽象 token 序列，使其带来高质量回答。

---

## 8. Stage 1：Policy Iteration Warm-Up

Warm-up 是论文最关键的训练机制之一。每轮迭代 `t = 1 ... T` 都包含两个子阶段。

### 8.1 每轮数据划分

每轮使用两个数据子集：

```text
D_t,1：用于 bottlenecked SFT
D_t,2：用于 self-distillation
```

文章将整体训练数据分阶段组织为：

```text
D = union over t of {(D_t,1, D_t,2)}
```

### 8.2 迭代流程

伪流程：

```text
初始化基础 instruction-tuned 模型 θ^(0)
加入 V_abs 和分隔符的新 embedding

for t = 1 ... T:
    在 D_t,1 上生成 abstract traces z_tilde^(t)
        if t == 1:
            使用随机初始化生成方式
        else:
            使用当前模型在 x + c 条件下 constrained decoding 生成

    用 [x; c; z_tilde; y] 做 bottlenecked SFT，得到 θ_bar^(t)

    在 D_t,2 上从 prompt alone 生成 z_tilde'
        z_tilde' ~ π^abs_{θ_bar^(t)}(. | x)

    用 [x; z_tilde'; y] 做 self-distillation，得到 θ^(t)

返回 θ^(T)
```

---

## 9. Bottlenecked SFT with Abstract Tokens

这是 warm-up 中的第一步。

### 9.1 输入结构

训练序列拼接为：

```text
s = [x ; c ; z_tilde ; y]
```

即：

```text
prompt + verbal CoT + abstract trace + answer
```

### 9.2 第一轮 abstract trace 初始化

当 `t = 1` 时，抽象 token 还没有意义。论文使用随机初始化方案：

- 对 verbal CoT 的每个 step `ℓ`：
  - step 长度为 `|ℓ|`
  - 随机采样抽象 token 数量：`rand(1, |ℓ| / 2)`
  - 抽象 token 从 `V_abs` 中均匀随机选取

文章比较过其他初始化方式：

- 按字母循环选择 token
- 强行使用 power-law 分布
- 均匀随机选择 token

最终发现：**均匀随机分布最有效**。

### 9.3 后续轮 abstract trace 生成

当 `t >= 2` 时，不再随机，而是由当前模型 on-policy 生成：

```text
z_tilde^(t) ~ π^abs_θ(. | x, c)
```

注意这里仍然给模型 verbal CoT `c` 作为条件，用于帮助生成更有信息量的 abstract trace。

### 9.4 注意力瓶颈 mask

Bottlenecked SFT 的关键是 block-structured attention mask。

将序列位置划分为：

| 区域 | 含义 |
|---|---|
| `X` | prompt |
| `C` | verbal CoT |
| `Z` | abstract sequence，包括分隔符 |
| `Z_abs` | abstract codebook tokens，不含分隔符 |
| `Y` | answer |

mask 规则：

1. abstract tokens 可以 attend 到：

```text
X + C + 已生成的 Z
```

也就是 abstract trace 可以读取 prompt 和 verbal CoT，从自然语言 CoT 中吸收信息。

2. answer 只能 attend 到：

```text
X + Z + 已生成的 Y
```

answer 不能 attend 到 verbal CoT `C`。

也就是说，最终答案无法直接读取自然语言 CoT；它只能通过 abstract tokens 间接获得 CoT 信息。

### 9.5 信息瓶颈含义

这个训练方式相当于强制形成：

```text
C -> H_Zabs -> Y    conditioned on X and Z
```

即：

- verbal CoT 的信息必须压缩进 abstract token 的 hidden states。
- answer 对 verbal CoT 的依赖只能通过 abstract segment 传递。
- `m_max` 控制瓶颈容量：抽象 token 越多，能传递的信息越多；越少，压缩越强。

### 9.6 SFT loss

训练目标只对 abstract tokens 和 answer 计算 loss：

```text
L_SFT = - sum_{j ∈ Z_abs ∪ Y} log π_θ(s_j | s_<j ; A)
```

其中 `A` 是上面的 bottleneck attention mask。

### 9.7 可选变体

文章提到可选做法：

- 将 `z_tilde^(t)` 视为固定，不对 abstract token 本身回传 loss。
- 这样 embedding 只通过 answer loss 的梯度被更新。

这可以作为一个 ablation idea。

---

## 10. Self-Distillation Without Verbal CoT

这是 warm-up 中的第二步。

### 10.1 目标

Bottlenecked SFT 让 abstract tokens 学会承载 verbal CoT 信息，但推理时没有 verbal CoT。因此需要训练模型：

```text
只看 x，也能生成有用的 abstract trace
```

### 10.2 数据构造

用经过 bottlenecked SFT 后的模型，从 prompt alone 生成 abstract trace：

```text
z_tilde ~ π^abs_θ(. | x)
```

然后与 gold answer 配对：

```text
D_distill^(t) = {(x_i, z_tilde_i, y_i)}
```

### 10.3 训练形式

训练序列：

```text
[x ; z_tilde ; y]
```

不再包含 verbal CoT。

使用标准 causal SFT，loss 仍然覆盖 abstract tokens 和 answer：

```text
L_Distill = - sum_{j ∈ Z_abs ∪ Y} log π_θ(s_j | s_<j)
```

### 10.4 关键 idea

Self-distillation 将 teacher-guided abstract trace 迁移成 inference-time 可用策略：

- 不再依赖 `c`
- 训练模型从 `x` 直接进入 abstract reasoning mode
- 逐轮迭代后，abstract token 的 embedding 和生成分布都会被强化

---

## 11. Stage 2：Warm-Started RL

Warm-up 后，模型已经能生成初步有用的 abstract trace。第二阶段使用 RL 继续优化抽象序列。

### 11.1 为什么不能直接 cold-start RL

论文发现：

- RL-only / cold-start RL 通常表现不佳。
- 随机初始化的 abstract token embedding 太难直接通过 RL 学出来。
- Warm-up 提供了必要的 burn-in，使 RL 有可优化的初始策略。

因此完整方法是：

```text
Warm-up + RL
```

而不是：

```text
RL-only
```

### 11.2 RL rollout 过程

对每个 prompt `x`，采样 `K` 条轨迹：

```text
z_tilde_k ~ π^abs_θ(. | x)
y_k ~ π_θ(. | x, z_tilde_k)
```

生成分两段：

1. 使用 guided-regex / constrained decoding 生成 abstract trace。
2. 追加 `<endabstract>` 后，用 unconstrained decoding 生成 answer。

### 11.3 奖励模型

文章使用 generative reward model：

```text
gpt-oss-20b
```

它对输出进行 0 到 10 分评分，维度包括：

- Helpfulness
- Accuracy
- Clarity
- Relevance
- Safety & Harmlessness

奖励模型输出 JSON：

```json
{
  "score": <0-10>,
  "reasoning": "<2-4 sentences>"
}
```

使用 generative reward model 的原因：

- 不只适用于数学这类可验证任务。
- 也适用于 instruction-following、multi-hop QA 等非严格可验证任务。

### 11.4 GRPO 优化

文章使用 GRPO。

对每组 `K` 个 rollout，先计算 reward：

```text
R_hat_k = R_hat(x, z_tilde_k, y_k)
```

然后标准化为 advantage：

```text
A_k = (R_hat_k - mean(R_hat_1:K)) / (std(R_hat_1:K) + ε)
```

更新时 action space 覆盖两部分：

- abstract trace tokens
- answer tokens

默认做法是同时更新二者的 log-prob：

```text
sum over abstract tokens log π^abs_θ(z_k,t | x, z_k,<t)
+
sum over answer tokens log π_θ(y_k,t | x, z_tilde_k, y_k,<t)
```

并加 KL regularization 到 reference policy：

```text
π_ref = warm-started model
```

KL 同时作用于：

- abstract distribution
- answer distribution

### 11.5 可选 RL 变体

文章提到另一个可选方案：

- RL 只更新 abstract tokens。
- answer 使用固定 decoding rule。

这可以作为更细粒度的 ablation idea，用来观察性能提升到底来自 abstract policy，还是 answer policy 也被强化。

---

## 12. 实验配置中可借鉴的参数

论文实验中的主要配置：

| 项目 | 设置 |
|---|---|
| Warm-up 数据 | Dolci-Think-SFT 子采样 600k |
| RL 数据 | Dolci-Think-RL |
| 模型 | Qwen3-8B, Qwen3-4B, Granite-4.0-Micro, Qwen3-32B ablation |
| 思考模式 | 关闭原模型 thinking mode，用标准 CoT prompting 控制比较 |
| Warm-up 轮数 | `T = 3` |
| 抽象词表大小 | 主实验 `M = 64` |
| 最大抽象长度 | `m_max = 128` |
| 每个 warm-up 子阶段 | 3 epochs |
| RL 训练量 | 1M episodes |
| SFT 硬件 | 8×NVIDIA H100 |
| RL 硬件 | up to 32×NVIDIA H100 |

---

## 13. Baselines 与对照实验 idea

论文比较了以下 baseline，可作为实现时的评估对照：

1. **Baseline**：直接从 prompt 生成答案。
2. **Pause Tokens**：在生成前插入 `m_max` 个 `<pause>` token。
3. **Stepwise Internalization / ICoT-SI**：逐步移除 CoT steps，把显式推理内化。
4. **SFT no CoT**：只用 `(x, y)` 做监督微调。
5. **SFT CoT**：用 `(x, c, y)` 做监督微调。
6. **SFT + RL**：先做 verbal CoT SFT，再做 RL。
7. **Abstract-CoT RL-only**：直接从随机 abstract token embedding 开始 RL。
8. **Abstract-CoT Warm-up only**：只做 policy iteration warm-up。
9. **Abstract-CoT Warm-up + RL**：完整方法。

重要结论：

- Pause token 虽然能缩短 token 数，但通常性能差。
- Cold-start RL 表现不稳定且经常低于 base model。
- Warm-up alone 有提升，但通常仍不如 verbal CoT + RL。
- Warm-up + RL 才是完整有效组合。

---

## 14. 主要实验结果中可转化的 insight

### 14.1 Token 效率

Abstract-CoT 在多个任务上显著减少 reasoning tokens：

- MATH-500：约 `10.4× - 11.6×` 更少 reasoning tokens。
- AlpacaEval：约 `1.9× - 2.2×` 更少。
- HotpotQA：约 `4.0× - 4.3×` 更少。
- GPQA-Diamond：约 `7.9×` 更少。
- AIME'25：约 `2.7×` 更少。

### 14.2 性能趋势

- 在 Qwen3-8B、Qwen3-4B、Granite-4.0-Micro 上，Abstract-CoT Warm-up + RL 接近或超过 SFT + RL。
- 在 AlpacaEval 上，Abstract-CoT 对所有模型都有明显收益。
- 在 HotpotQA 上，Abstract-CoT 也能达到或接近 verbal CoT + RL。
- 在更难的 AIME'25 / GPQA-Diamond 上，Abstract-CoT 仍能在大幅降低 token 的同时接近 SFT + RL。
- Qwen3-32B ablation 说明方法可扩展到更大模型。

---

## 15. 抽象 token 分布分析 idea

文章观察到，经过训练后，抽象 token 使用频率呈现类似自然语言 Zipf’s law 的 power-law 分布。

### 15.1 现象

- Warm-up 初始使用均匀随机 token。
- On-policy generation 和 RL 后，token 分布变得不均匀。
- 某些 token 被大量复用。
- 部分低频 token 也开始在训练中被激活。
- 论文中特别观察到 `<TOKEN_F>` 在 RL 后明显成为高频 token。

### 15.2 含义

这说明模型可能学到了一种抽象推理语言：

- 高频 token 可能对应通用推理操作或概念。
- 低频 token 可能对应更稀有的任务模式或概念。
- 抽象词表并不是随机噪声，而是在训练中形成结构化使用模式。

### 15.3 编程可分析方向

- 记录每个阶段 token frequency。
- 对比 warm-up 前、warm-up 后、RL 过程中的分布变化。
- 画 rank-frequency plot。
- 拟合 power-law / Zipf-like 曲线。
- 分析不同任务类型中 token 使用差异。
- 分析高频 token 是否对应某些 prompt cluster。

---

## 16. Vocabulary Size Scaling idea

论文对 `M = 1` 到 `M = 512` 进行了 abstract vocabulary size ablation。

### 16.1 观察

- 增大抽象词表一般会提升性能。
- 但提升会饱和，并在过大时略有下降。
- `M = 64` 是主实验中选择的较优配置。
- Cold-start RL 在各词表大小下仍然普遍不如 base model。
- 较大词表下，power-law 分布更明显，尾部 token 更长。

### 16.2 可实现 idea

- 实现可配置 `M`。
- 跑不同词表规模：`1, 2, 4, 8, 16, 32, 64, 128, 256, 512`。
- 对每个 `M` 分别比较：
  - PI-1
  - PI-2
  - PI-3
  - PI-3 + RL
  - cold-start RL
- 同时记录：
  - benchmark score
  - average abstract length
  - token frequency distribution
  - unused token ratio

---

## 17. Truncation Analysis idea

论文测试了如果强行截断 CoT 会怎样。

### 17.1 做法

对 verbal CoT：

```text
生成自然语言 CoT -> 截断到 k tokens -> 添加 </think> -> 生成答案
```

对 Abstract-CoT：

```text
生成 abstract trace -> 截断到 k tokens -> 添加 <endabstract> -> 生成答案
```

测试 `k = 32, 48, 64`。

### 17.2 观察

- verbal CoT 和 Abstract-CoT 截断后都会降分。
- verbal CoT 降幅更大，尤其在 MATH-500 等需要较长推理的任务上。
- Abstract-CoT 因为原本就是短而有界的 trace，截断时退化更平滑。

### 17.3 可实现 idea

- 实现 inference-time budget truncation。
- 比较 fixed budget 下 verbal CoT vs Abstract-CoT。
- 评估“同样 token 预算下谁更有效”。
- 用截断曲线衡量 abstract trace 的信息密度。

---

## 18. Permutation Testing idea

论文通过打乱推理序列测试抽象语言是否具有顺序组合性。

### 18.1 做法

对每个 prompt：

1. 先生成 CoT。
2. 对 verbal CoT：按 newline / step 级别打乱。
3. 对 Abstract-CoT：对 abstract tokens 做随机排列。
4. 保持其余条件不变，再生成答案并评估。

### 18.2 观察

- verbal CoT 被打乱后性能下降。
- Abstract-CoT 被打乱后也明显下降。
- RL 后的 Abstract-CoT 对 permutation 更敏感，说明它学到了有序序列，而不只是 token bag。

### 18.3 可实现 idea

- 添加 permutation evaluation mode。
- 分别测试：
  - token-level permutation
  - block-level permutation
  - partial permutation
  - high-frequency token 固定、低频 token 打乱
- 用性能下降幅度衡量 abstract language 的 compositionality。

---

## 19. Reward Model Prompt idea

文章附录给出了 generative reward model prompt 的核心结构。

可提取为 reward evaluator 模板：

```text
You are an expert evaluator assessing the quality of AI assistant responses.
Your task is to score a response on a scale from 0 to 10.
```

评分维度：

- Helpfulness：是否满足用户需求。
- Accuracy：是否事实正确，无 hallucination。
- Clarity：是否结构清晰、易理解。
- Relevance：是否聚焦用户问题。
- Safety & Harmlessness：是否安全、尊重、无害。

输出格式：

```json
{
  "score": <number between 0-10>,
  "reasoning": "<2-4 sentences explaining your score>"
}
```

可实现 idea：

- 独立 reward model wrapper。
- 将 conversation history 和 response 拼接成 evaluator input。
- 解析 JSON score。
- 对非法 JSON 或缺失 score 做容错。
- 可切换不同 reward model。
- 数学任务可替换为 verifier reward，通用任务用 generative reward。

---

## 20. 可复现的训练任务拆解

以下是从论文方法中提取的编程任务 idea，不涉及具体工程设计。

### 20.1 Tokenizer 扩展任务

- 支持新增 `M` 个 abstract tokens。
- 支持 `<beginabstract>` 和 `<endabstract>`。
- 初始化新增 embedding。
- 支持不同命名规则：单字母、双字母、扩展字母。

### 20.2 Constrained Decoding 任务

- 支持 abstract mode。
- abstract mode 下 logits 只允许 `V_abs + <endabstract>`。
- 支持最大长度 `m_max`。
- 达到长度上限后强制 end delimiter。
- end delimiter 后切换回普通 decoding。

### 20.3 Bottleneck Attention Mask 任务

- 构造 `[x; c; z; y]` 的 block attention mask。
- 允许 `Z` attend `X + C + previous Z`。
- 禁止 `Y` attend `C`。
- 允许 `Y` attend `X + Z + previous Y`。
- 支持标准 causal mask 的兼容。

### 20.4 Abstract Trace 初始化任务

- 将 verbal CoT 切分为 steps。
- 根据每步长度随机采样 abstract token 数。
- 从 `V_abs` 均匀采样 token。
- 支持替代初始化策略用于 ablation。

### 20.5 Warm-Up Loop 任务

- 实现 `T` 轮 policy iteration。
- 每轮包含：
  - bottlenecked SFT
  - self-distillation
- 管理 `D_t,1` 与 `D_t,2`。
- 保存每轮模型 checkpoint。
- 记录每轮 abstract token 分布。

### 20.6 Self-Distillation 数据生成任务

- 用当前模型从 `x` 生成 `z`。
- 与 gold answer `y` 配对。
- 构造 `[x; z; y]` SFT 样本。
- 不使用 verbal CoT。

### 20.7 Warm-Started RL 任务

- 从 warm-up 后模型初始化 RL。
- 每个 prompt 采样 `K` 条 `(z, y)` trajectory。
- 用 reward model 评分。
- 计算 group-relative advantage。
- 对 abstract tokens 和 answer tokens 做 GRPO 更新。
- 加 KL 到 warm-start reference model。

### 20.8 Evaluation 任务

- 评估任务至少覆盖：
  - 数学推理
  - instruction-following
  - multi-hop QA
  - 更难自然语言 / 数学推理任务
- 同时统计：
  - score
  - total generated tokens
  - abstract tokens
  - answer tokens
  - compression ratio

### 20.9 Analysis 任务

- token frequency evolution。
- rank-frequency / power-law 分布。
- permutation sensitivity。
- truncation sensitivity。
- vocabulary size scaling。
- model size scaling。
- cold-start vs warm-start 对比。

---

## 21. 实现时值得保留的完整方法要点

以下要点不可遗漏：

1. Abstract-CoT 是离散 latent reasoning，不是 continuous latent vector。
2. Abstract tokens 是新增 reserved vocabulary，不是从 teacher CoT 量化来的 token。
3. 训练完全可以在 post-training 阶段完成，不要求 continued pretraining。
4. 新 token embedding 初始无意义，必须 warm-up。
5. Warm-up 不是单纯 SFT，而是 policy iteration：
   - bottlenecked SFT
   - self-distillation
   - 多轮重复
6. Bottlenecked SFT 中，answer 不能 attend verbal CoT。
7. Abstract trace 可以 attend prompt 和 verbal CoT，从中吸收信息。
8. Self-distillation 阶段必须去掉 verbal CoT。
9. 抽象生成必须 constrained decoding。
10. `m_max` 是重要容量控制参数。
11. RL 要从 warm-up 模型开始，而不是随机 cold-start。
12. RL rollout 中先 constrained 生成 abstract trace，再普通生成 answer。
13. GRPO 默认同时更新 abstract trace 和 answer token log-probs。
14. KL reference 是 warm-started model。
15. generative reward model 可支持非可验证任务。
16. 主实验 `M=64`, `m_max=128`, `T=3`。
17. 需要对比 RL-only、warm-up only、warm-up + RL。
18. 需要做 vocabulary scaling、truncation、permutation 分析。
19. 需要记录 token 分布，观察 power-law / Zipf-like 行为。
20. 抽象 token 的有序性和组合性可通过 permutation degradation 验证。

---

## 22. 潜在扩展 idea

论文讨论中提出或暗示的可扩展方向：

### 22.1 Difficulty-aware abstract budget

不同问题可能需要不同长度的 abstract trace。

可探索：

- 根据 prompt 难度动态调整 `m_max`。
- 训练模型自主决定何时结束 abstract trace。
- 使用 reward 中加入 token cost，实现 accuracy-cost trade-off。

### 22.2 Hierarchical abstract codebook

抽象词表可能形成层级结构。

可探索：

- 高频 token 作为通用推理 primitive。
- 低频 token 作为任务特定 primitive。
- 将 token 分成 function-like / domain-like / control-like 子类。
- 让高阶 abstract token 代表可复用 subroutine。

### 22.3 Interpretability / monitorability

Abstract trace 虽不可读，但仍是显式中间段。

可探索：

- 将 abstract token 聚类到任务类型。
- 分析 token 与中间 hidden state 的关系。
- 检测有害或异常 abstract patterns。
- 对 abstract trace 做 audit，而不是只能看最终答案。

### 22.4 Hybrid readable + abstract CoT

文章主要做全 abstract trace，但相关工作中有 hybrid latent/text 方向。

可探索：

- 少量自然语言 checkpoint + abstract tokens。
- 关键步骤保留自然语言，其余步骤 abstract。
- 用户需要解释时再将 abstract trace 翻译成自然语言摘要。

### 22.5 任务自适应 codebook

不同任务可能需要不同抽象 token 分布。

可探索：

- 数学 codebook。
- 编程 codebook。
- 多跳检索 codebook。
- instruction-following codebook。
- 多任务共享 codebook vs 任务专属 codebook。

---

## 23. 风险与注意事项

1. **不可解释性风险**：abstract trace 不可直接读懂，虽然可分析，但不等于自然语言解释。
2. **冷启动困难**：没有 warm-up 时 RL 难以学出有效 token embedding。
3. **奖励黑箱化**：generative reward model 会影响 abstract language 的形成。
4. **抽象 token 可能过拟合任务分布**：需要跨任务评估。
5. **词表过大不一定更好**：论文观察到性能会饱和并可能下降。
6. **过短 budget 可能不足以长程推理**：需要动态预算机制。
7. **mask 实现必须严格**：如果 answer 能偷看 verbal CoT，bottleneck 失效。
8. **评估不能只看准确率**：必须同时统计 token 成本与 trace 长度。
9. **abstract token 顺序很重要**：不能把它当作无序 embedding bag。
10. **不同模型家族需要验证泛化**：论文在 Qwen 与 Granite 上测试，但实现中仍需重新验证。

---

## 24. 最小可行复现路线

可以按以下顺序推进：

1. 在一个小模型上扩展 tokenizer，加入 abstract tokens 和 delimiters。
2. 实现 abstract constrained decoding。
3. 构造小规模 `(x, c, y)` 数据。
4. 实现第一轮随机 abstract trace 初始化。
5. 实现 bottleneck attention mask。
6. 跑 Bottlenecked SFT。
7. 从 prompt alone 生成 abstract trace，构造 self-distillation 数据。
8. 跑 Self-Distillation。
9. 重复 2-3 轮 warm-up。
10. 先评估 warm-up only。
11. 加入 reward model 和 GRPO。
12. 对比 RL-only 与 warm-started RL。
13. 统计 token efficiency。
14. 做 truncation 和 permutation 分析。
15. 做 vocabulary size ablation。

---

## 25. 一句话总结

Abstract-CoT 的核心 idea 是：**通过新增离散抽象 token、瓶颈式 SFT、自蒸馏和 warm-started RL，让模型学会在回答前生成一段短小但有用的“不可读推理语言”，从而在不依赖长篇自然语言 CoT 的情况下获得高效推理能力。**
