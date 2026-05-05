# Abstract-CoT 复现设计文档

## 1. 目标与范围

### 1.1 项目目标

本项目目标是复现论文 *Thinking Without Words: Efficient Latent Reasoning with Abstract Chain-of-Thought* 的核心方法，并形成一个可扩展的实验框架，用于回答三个问题：

1. 抽象离散 token 是否能替代长自然语言 CoT 作为中间推理表示。
2. 在显著减少 reasoning tokens 的前提下，模型性能是否能接近 verbal CoT + RL。
3. 抽象推理语言在训练后是否呈现出可分析的结构性特征，例如顺序性、频率分布和任务适应性。

### 1.2 复现范围

本次设计覆盖以下能力：

- tokenizer 扩展与 abstract token codebook 注入
- abstract trace constrained decoding
- bottlenecked SFT warm-up
- self-distillation warm-up
- warm-started RL
- 评估与分析工具链

不纳入第一阶段范围的内容：

- 多模型家族同时支持
- 大规模分布式训练优化细节
- 在线服务化部署
- 高级可视化平台

### 1.3 复现原则

- 先做最小可行闭环，再扩展 RL 和大规模实验
- 方法保真优先于框架美观
- 训练与评估解耦
- 所有关键中间产物必须可落盘复查
- 所有关键实验参数必须配置化

---

## 2. 论文约束到工程约束的映射

从论文方法中提炼出的不可破坏约束如下。

### 2.1 抽象词表约束

- 必须引入独立的 reserved abstract tokens
- 必须引入 `<beginabstract>` 与 `<endabstract>`
- 抽象 token 不能复用已有自然语言词表 token

### 2.2 解码约束

- 进入 abstract mode 后，只允许生成 `V_abs + <endabstract>`
- abstract trace 长度不能超过 `m_max`
- 一旦输出 `<endabstract>`，后续恢复普通 answer generation

### 2.3 Warm-up 约束

- 第一阶段不是普通 SFT，必须包含 policy iteration
- bottlenecked SFT 中，answer 不允许直接 attend verbal CoT
- self-distillation 阶段必须移除 verbal CoT
- warm-up 至少需要多轮迭代，不能只做单轮随机初始化

### 2.4 RL 约束

- RL 必须从 warm-up checkpoint 启动，而不是 cold-start
- rollout 必须先生成 abstract trace，再生成 answer
- reward 作用于完整 `(x, z, y)` 轨迹
- 需要保留 reference policy 用于 KL 约束

### 2.5 评估约束

- 不能只看任务分数，必须同时看 reasoning token 成本
- 必须记录 abstract token 使用分布
- 必须支持 truncation 和 permutation 分析

---

## 3. 现状分析

当前仓库状态：

- `abstract_cot/` 为空
- `document/abstract_cot_programming_ideas.md` 已整理论文方法和实验观察

这意味着当前项目处于“仅有方法说明、无工程实现”的初始阶段。设计上应避免直接假设已有训练框架、数据流水线或评估基础设施。

因此本项目应按“从零搭建可复现实验仓库”处理，而不是在现有代码上追加功能。

---

## 4. 总体设计

### 4.1 分层结构

系统按五层设计：

1. `基础层`
   - 配置管理
   - tokenizer / vocab 扩展
   - 日志与实验目录管理

2. `模型与解码层`
   - 模型加载
   - embedding 扩展
   - constrained decoding
   - bottleneck attention mask

3. `数据与训练层`
   - warm-up 数据构造
   - self-distillation 数据构造
   - RL rollout 数据构造
   - SFT / RL trainer

4. `评估与分析层`
   - benchmark runner
   - token 统计
   - truncation / permutation evaluator
   - vocabulary scaling analyzer

5. `实验编排层`
   - 单实验入口
   - 多阶段 pipeline
   - checkpoint / artifact 管理

### 4.2 设计核心

工程上要把项目拆成两个闭环：

- `训练闭环`：`x,c,y -> warm-up -> RL -> checkpoint`
- `分析闭环`：`checkpoint -> rollout -> score/token stats/trace analysis`

这两个闭环必须独立运行。否则训练逻辑会和实验分析强耦合，后续做 ablation 会非常痛苦。

---

## 5. 推荐目录结构

建议将仓库组织为以下结构：

```text
abstract_cot/
  configs/
    model/
    data/
    train/
    eval/
    experiment/
  src/
    abstract_cot/
      __init__.py
      tokenization/
        abstract_vocab.py
        tokenizer_extension.py
      decoding/
        constrained_decoder.py
        abstract_generation.py
      modeling/
        model_loader.py
        embedding_resize.py
        attention_mask.py
      data/
        schema.py
        prompt_formatter.py
        warmup_dataset.py
        distill_dataset.py
        rl_dataset.py
      training/
        sft_trainer.py
        bottleneck_sft.py
        self_distill.py
        warmup_loop.py
        grpo_trainer.py
        rollout.py
      reward/
        interfaces.py
        generative_rm.py
        parser.py
      eval/
        benchmark_runner.py
        truncation.py
        permutation.py
        token_stats.py
      analysis/
        frequency.py
        zipf_fit.py
        trace_inspection.py
      utils/
        config.py
        logging.py
        seed.py
        io.py
  scripts/
    run_warmup.py
    run_distill.py
    run_rl.py
    run_eval.py
    run_analysis.py
  outputs/
    experiments/
  document/
```

### 5.1 目录设计理由

- `tokenization/` 与 `decoding/` 分开，避免把“词表定义”和“约束生成逻辑”写死在一起
- `training/` 分解为 warm-up、distill、RL，便于分阶段调试
- `eval/` 与 `analysis/` 分离，前者偏指标，后者偏研究结论
- `scripts/` 只做薄入口，不承载业务逻辑

---

## 6. 核心模块设计

### 6.1 tokenizer 扩展模块

职责：

- 生成 abstract token 列表
- 注入 `<beginabstract>` 与 `<endabstract>`
- 保存扩展后的 tokenizer
- 返回 special token ids 与 abstract vocab ids

接口建议：

```python
build_abstract_vocabulary(size: int, scheme: str) -> list[str]
extend_tokenizer(base_tokenizer_path: str, abs_tokens: list[str]) -> TokenizerArtifacts
```

设计要求：

- 支持 `M` 配置化
- 支持命名方案扩展
- 新 token id 集合必须可序列化保存，供训练和推理共用

### 6.2 embedding 扩展模块

职责：

- 按 tokenizer 新词表尺寸 resize embeddings
- 记录新增 token 范围
- 提供随机初始化策略

注意点：

- 必须保证 tokenizer 与 model embedding 对齐
- 初始化策略先采用默认随机初始化，后续可加 ablation

### 6.3 constrained decoding 模块

职责：

- 在 abstract 阶段对 logits 做 mask
- 控制最大长度 `m_max`
- 在 abstract 结束后切换到普通 decoding

关键接口：

```python
generate_abstract_trace(model, tokenizer, prompt_ids, abs_token_ids, end_id, m_max, sampling_config)
generate_answer_from_trace(model, tokenizer, prompt_ids, trace_ids, gen_config)
generate_full_response(...)
```

实现要求：

- 训练时 rollout 与评估时推理应复用同一套约束逻辑
- 不允许训练和评估各自实现一份 mask 逻辑

### 6.4 bottleneck attention mask 模块

职责：

- 基于 `[x; c; z; y]` 构造 block-structured causal attention mask

逻辑要求：

- `Z` 可看 `X + C + previous Z`
- `Y` 可看 `X + Z + previous Y`
- `Y` 不可看 `C`

设计建议：

- 通过 segment ids 驱动 mask 构造，而不是写死固定位置逻辑
- 这样可支持不同样本长度和 batch padding

### 6.5 warm-up 数据模块

输入样本格式：

```text
{
  "prompt": x,
  "cot": c,
  "answer": y
}
```

职责：

- 构造第一轮随机 abstract trace
- 构造后续轮 on-policy abstract trace
- 产出 bottlenecked SFT 样本

第一轮随机初始化策略：

- 按 verbal CoT step 切分
- 每个 step 随机采样 1 到 `|step|/2` 个 abstract tokens
- token 从 `V_abs` 均匀采样

### 6.6 self-distillation 模块

职责：

- 从当前 warm-up 模型仅基于 `x` 生成 `z`
- 构造 `[x; z; y]` 样本
- 对 abstract tokens 和 answer 一起做 SFT

设计要求：

- teacher rollout 和学生训练的数据快照必须持久化
- 便于复查某轮 distillation 的 trace 质量

### 6.7 RL 模块

职责：

- 基于 warm-start model 执行 grouped rollout
- 对每个 prompt 采样 `K` 组 `(z, y)`
- 使用 reward model 打分
- 计算 GRPO advantage
- 更新 abstract 与 answer token policy

建议拆分：

- `rollout.py`：只负责采样与轨迹记录
- `grpo_trainer.py`：只负责 loss 计算与参数更新
- `reward/`：只负责 reward 推断和解析

这样后续切换 reward 模型或替换 RL 算法时，修改面最小。

---

## 7. 数据设计

### 7.1 数据 schema

统一样本字段建议：

```json
{
  "id": "sample-xxx",
  "prompt": "...",
  "cot": "...",
  "answer": "...",
  "task_type": "math|qa|instruction",
  "meta": {}
}
```

### 7.2 中间产物 schema

warm-up / distill / RL 都应统一记录：

```json
{
  "id": "sample-xxx",
  "prompt": "...",
  "cot": "...",
  "abstract_trace_text": "<beginabstract> <TOKEN_A> ... <endabstract>",
  "abstract_trace_ids": [ ... ],
  "answer": "...",
  "stage": "warmup|distill|rl",
  "round": 1,
  "reward": 7.5
}
```

### 7.3 数据集分层

建议分三类数据：

- `warmup_train`
- `rl_train`
- `eval`

不要让 warm-up 和 RL 共用同一份产物目录，否则 checkpoint 与 trace 的来源会混乱。

---

## 8. 训练流程设计

### 8.1 Phase A: MVP warm-up only

目标是最快打通最小闭环。

流程：

1. 选择基础模型
2. 扩展 tokenizer 与 embeddings
3. 准备小规模 `(x,c,y)` 数据
4. 构造首轮随机 abstract trace
5. 跑 bottlenecked SFT
6. 从 prompt-only 生成 distill traces
7. 跑 self-distillation
8. 重复 `T=2~3` 轮
9. 评估 warm-up only 效果和 token 成本

交付标准：

- 能生成合法 abstract trace
- answer 在推理时不依赖 verbal CoT
- 至少产出一组可用 checkpoint 与 trace artifacts

### 8.2 Phase B: warm-started RL

流程：

1. 加载 warm-up 最终 checkpoint
2. 对每个 prompt 采样 `K` 条轨迹
3. reward model 打分
4. 计算 GRPO loss
5. 更新策略并落盘 rollout 结果

交付标准：

- RL 训练能稳定收敛
- reward、KL、trace length 等曲线可追踪
- 相对 warm-up only 有可测提升

### 8.3 Phase C: 分析实验

包括：

- vocabulary size scaling
- truncation analysis
- permutation testing
- token frequency evolution

这些应在训练稳定后再上，避免前期陷入分析工具开发。

---

## 9. 配置系统设计

建议所有实验通过配置驱动，避免将关键参数散落在脚本中。

### 9.1 核心配置项

模型配置：

- base model path
- tokenizer path
- precision
- gradient checkpointing

abstract 配置：

- `M`
- `m_max`
- token naming scheme
- init seed

warm-up 配置：

- `T`
- each stage epochs
- batch size
- learning rate

RL 配置：

- rollout group size `K`
- KL coefficient
- reward model backend
- max answer tokens

评估配置：

- benchmarks
- decode settings
- truncation budgets
- permutation modes

### 9.2 配置原则

- 一个实验只对应一个主配置文件
- 所有输出目录按实验 id 命名
- 实验配置原样复制到输出目录，保证可复现

---

## 10. 实验与评估设计

### 10.1 基础对照

第一阶段最少应支持以下 baseline：

1. `base`：prompt -> answer
2. `sft_no_cot`
3. `sft_cot`
4. `abstract_warmup_only`
5. `abstract_warmup_rl`

如果资源不足，可先不做 `RL-only`，但接口要预留。

### 10.2 指标

每次评估至少记录：

- task score
- total generated tokens
- abstract token count
- answer token count
- compression ratio
- average trace length
- invalid trace ratio

### 10.3 分析实验

`truncation`

- 对 abstract trace 截断到 `32/48/64/...`
- 比较准确率与 token 成本变化

`permutation`

- token-level shuffle
- block-level shuffle
- partial shuffle

`frequency analysis`

- token rank-frequency
- unused token ratio
- 高频 token 随训练轮次变化

---

## 11. 工程风险与应对

### 11.1 最大风险

`attention mask 实现错误`

如果 `Y` 能看到 `C`，整个 bottleneck 假设失效。这个风险必须通过单元测试和可视化检查优先消除。

### 11.2 第二风险

`训练与推理解码逻辑不一致`

如果 rollout 和 eval 用两套 constrained decoding，实现很容易漂移，最终实验不可信。

### 11.3 第三风险

`中间产物不可追溯`

如果不保存 warm-up 每轮生成的 trace、distill 数据和 RL rollout，后续很难定位性能变化来源。

### 11.4 应对策略

- 对 mask 写严格单测
- 对 constrained decoding 写 golden tests
- 对每轮中间 trace 做 sample dump
- 对 reward parsing 做容错
- 所有实验输出统一归档

---

## 12. 最小里程碑

### M1: 基础设施

- tokenizer 扩展
- embedding resize
- constrained decoding
- artifact 管理

验收标准：

- 能从基础模型生成合法 abstract trace

### M2: warm-up MVP

- bottleneck attention mask
- warm-up 数据管线
- bottlenecked SFT
- self-distillation

验收标准：

- 2 到 3 轮 warm-up 可跑通

### M3: RL MVP

- grouped rollout
- reward model wrapper
- GRPO 更新

验收标准：

- RL 能从 warm-up checkpoint 启动并产出稳定日志

### M4: 评估与分析

- benchmark runner
- truncation / permutation
- token frequency analysis

验收标准：

- 形成完整实验报告与核心图表

---

## 13. 实施建议

### 13.1 基础模型选择

第一轮建议只支持一个中等规模开源模型，优先保证闭环而不是并行支持多模型。

建议策略：

- MVP 阶段选 1 个模型
- warm-up 跑通后再抽象模型适配层

### 13.2 开发顺序

推荐顺序如下：

1. tokenizer 扩展
2. constrained decoding
3. attention mask 单测
4. bottleneck SFT
5. self-distillation
6. warm-up 评估
7. RL
8. 分析实验

原因很直接：如果前三步不稳定，后面所有训练结果都不可信。

### 13.3 文档与实验同步

每个里程碑都应同步更新：

- 当前支持能力
- 已知限制
- 默认实验命令
- 关键配置说明

否则项目很快会变成“只有作者自己能跑”的私有脚本集合。

---

## 14. 本次设计结论

这个项目不应一开始就被设计成“大而全训练平台”，而应是一个围绕 Abstract-CoT 方法验证的分阶段实验系统。

第一优先级是确保以下三件事成立：

1. abstract token 与 constrained decoding 机制正确
2. bottlenecked SFT 的信息瓶颈真正生效
3. warm-up 后模型能在没有 verbal CoT 的情况下生成可用 abstract trace

在此基础上，再引入 RL 和系统性分析实验。这样设计可以最大程度降低前期复杂度，同时保留后续扩展到论文完整实验的路径。

---

## 15. 远程开发与运行约束

当前执行约束已明确为：

- 本地机器负责 git 管理与代码编辑
- `cisl113` 负责训练、模型下载和数据下载
- 本地到远端通过 `mutagen` 做 one-way 代码同步
- 服务器 Python 环境使用 `conda`，依赖安装使用 `uv pip`

### 15.1 同步策略

推荐策略：

- 本地目录作为唯一 source of truth
- 远端目录只作为运行镜像
- `mutagen project start` 驱动同步
- `mutagen.yml` 作为项目级配置文件，由 Mutagen 自动检索
- project YAML 中的 sync defaults 使用 `one-way-safe`
- `mutagen` 参数写入 YAML 配置文件管理
- 忽略 `.git`、`.venv`、`outputs`、`server_assets`、`models`、`datasets`、`data`、`__pycache__`

这样做的原因是：

- 远端训练过程会产生大量中间产物，不适合回流到本地 git 工作区
- 本地和远端职责清晰，不会出现“远端临时改动污染版本管理”的问题

### 15.2 服务器目录约定

约定：

- 远端项目根目录：`~/workspace/abstract_cot`
- 远端 conda 环境名：`abstract-cot`
- 远端缓存根目录：`~/workspace/.cache/abstract_cot`

这些路径是当前项目标准约定。若服务器现有目录规范不同，可以在脚本环境变量中覆盖。

### 15.3 初始模型选择

考虑到 `cisl113` 有 `8 x RTX 3090`，第一阶段建议按以下顺序推进：

1. `Qwen/Qwen3-0.6B`
2. `Qwen/Qwen3-1.7B`
3. `Qwen/Qwen3-4B`

原因：

- `0.6B` 最适合作为 tokenizer 扩展、mask、warm-up 数据链与最小训练闭环的联调模型
- `1.7B` 适合作为 warm-up 稳定性和 token efficiency 的中间档验证
- `4B` 可作为第一批较可信的论文复现规模

### 15.4 数据下载策略

必须遵守：

- 模型只在服务器下载
- 数据集只在服务器下载
- 本地工作区不保存模型或数据副本

第一批目标资产：

- warm-up 数据：`dolcivallone/dolci-think-sft`
- RL 数据：`dolcivallone/dolci-think-rl`
- baseline 模型：`Qwen/Qwen3-0.6B`

下载通道默认使用：

- `HF_ENDPOINT=https://hf-mirror.com`

如果服务器对 Hugging Face 镜像访问仍然受限，可再补一层 ModelScope 下载适配。

---

## 16. 实验性连续 Latent 分支

除论文原始的离散 abstract token 路线外，当前项目额外纳入一个实验性分支：

- 不生成离散 abstract tokens 作为下一步输入
- 每一步直接复用上一轮最后一层 hidden state
- 若 stop 预测不是 `<endabstract>`，则该 hidden state 直接作为下一步 `inputs_embeds`

### 16.1 核心假设

这一路线的核心假设是：

- 最后层 hidden state 自身可以作为一种连续推理载体
- 模型可以通过反复“hidden state 自回注入”形成一个短程连续 latent reasoning loop
- 最终答案生成不依赖离散 abstract codebook，而依赖 prompt + latent state trace

### 16.2 与论文方法的关系

这不是论文原始方法的等价实现，而是一个平行实验。

区别在于：

- 论文方法使用离散 abstract vocabulary
- 本实验直接使用连续 hidden states
- 论文依赖 constrained decoding
- 本实验依赖 stop token 判定和 embedding recycle

因此，实验报告中应明确区分：

- `discrete abstract-cot`
- `continuous recurrent latent`

### 16.3 SFT 设计

SFT 路线设计为：

1. prompt 先生成连续 latent trace
2. 将 `prompt embeddings + latent states + answer embeddings` 拼接
3. 只对 answer token 计算 loss

这样做的原因是：

- latent states 本身不是离散词表 token
- 难以直接做 token-level teacher forcing
- 先保证“latent trace 是否有助于 answer”这一主问题可测

后续可扩展项：

- stop prediction auxiliary loss
- latent consistency regularization
- detach / no-detach ablation

### 16.4 RL 设计

RL 路线设计为：

1. 先 rollout 连续 latent trace
2. 再基于 `prompt + latent trace` 生成 answer
3. reward 仍作用于最终 `(prompt, latent_trace, answer)` 轨迹

目前实现上保留了：

- latent trace 收集
- answer generation 接口
- 轨迹容器

后续需要再补：

- 连续 latent 轨迹的 policy update 定义
- 是否对 stop logits 施加 RL 梯度
- latent state 是否允许梯度穿过整段 rollout

### 16.5 风险

这一路线有几个明显风险：

1. hidden state 反复回注入后可能快速退化或发散
2. Qwen 类模型未必天然适配单步 `inputs_embeds` 递归调用
3. stop token 语义可能不稳定
4. 连续 latent 缺乏离散 codebook 的可审计性
5. RL 定义会比离散 token 路线更不直接

因此该分支应视为高风险高回报实验，而不是当前主线复现替代品。
