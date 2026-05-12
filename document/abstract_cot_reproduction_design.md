# Abstract-CoT 复现设计与当前状态

## 1. 项目目标

本项目目标是复现论文 *Thinking Without Words: Efficient Latent Reasoning with Abstract Chain-of-Thought* 的核心训练路径，并在此基础上搭建一个可继续扩展到 RL 和分析实验的工程骨架。

当前优先级：

1. 跑通离散 abstract token 路线的 Stage 1 warm-up
2. 在服务器上稳定跑多卡训练
3. 为后续 Stage 2 RL 和分析实验保留足够清晰的数据与模型接口

---

## 2. 当前仓库状态

仓库已经不是“只有文档没有实现”的状态。当前已有一套可运行的最小系统，位于 [abstract_cot](../abstract_cot)。

已落地的模块：

- tokenizer 扩展与 abstract vocab
- constrained abstract decoding
- bottleneck attention mask 规则
- warm-up 数据构造
- tokenization / collation / forward batch adapter
- 最小 SFT trainer
- Stage 1 warm-up 主循环
- 服务器环境脚本
- 原始 Dolci-Think-SFT-7B 到 `-cot` 数据集的预处理脚本
- 一个连续 latent 的实验性分支

当前主要配置和入口：

- 实验配置：[abstract_cot/configs/experiment/mvp_warmup.yaml](../abstract_cot/configs/experiment/mvp_warmup.yaml)
- Stage 1 入口：[abstract_cot/scripts/run_warmup_mvp.py](../abstract_cot/scripts/run_warmup_mvp.py)
- 服务器启动脚本：[abstract_cot/scripts/run_server_warmup.sh](../abstract_cot/scripts/run_server_warmup.sh)
- 预处理脚本：[abstract_cot/scripts/preprocess_dolci_think_sft.py](../abstract_cot/scripts/preprocess_dolci_think_sft.py)

---

## 3. 已实现的核心设计

### 3.1 Tokenizer 与抽象词表

已实现：

- `V_abs` 抽象 token 集合
- `<beginabstract>` / `<endabstract>`
- tokenizer 扩展与持久化
- model embedding resize

对应实现：

- [abstract_vocab.py](../abstract_cot/src/abstract_cot/tokenization/abstract_vocab.py)
- [tokenizer_extension.py](../abstract_cot/src/abstract_cot/tokenization/tokenizer_extension.py)
- [embedding_resize.py](../abstract_cot/src/abstract_cot/modeling/embedding_resize.py)

### 3.2 Constrained decoding

已实现：

- abstract mode 下只允许 `V_abs + <endabstract>`
- 最大 abstract 长度约束
- warm-up 中 on-policy trace 生成

当前 on-policy trace 生成已经进一步改成：

- batch 级 prompt encode
- tokenizer 左 padding
- batched autoregressive decode
- KV cache 增量生成

也就是说，当前 trace 生成不再按 sample 串行逐条 decode，而是整 batch 一起生成。

对应实现：

- [constrained_decoder.py](../abstract_cot/src/abstract_cot/decoding/constrained_decoder.py)
- [bottleneck_sft.py](../abstract_cot/src/abstract_cot/training/bottleneck_sft.py)

### 3.3 Bottleneck attention mask

目标约束：

- `Z` 可看 `X + C + previous Z`
- `Y` 可看 `X + Z + previous Y`
- `Y` 不可看 `C`

当前状态：

- 规则本身已实现并测试
- 当前训练主路径已不再依赖显式的 4D additive bottleneck mask
- bottleneck phase 改为两次标准 causal decoder forward 来等价实现约束

当前训练等价改写：

1. `Z` 子阶段输入 `[X; C; Z]`
   - 只在 `Z` token 上计算 loss
   - 因果 mask 天然保证 `Z` 只能看 `X + C + previous Z`

2. `Y` 子阶段输入 `[X; Z; Y]`
   - 只在 `Y` token 上计算 loss
   - 因果 mask 天然保证 `Y` 只能看 `X + Z + previous Y`

这样做的直接收益：

- 避免构造和传递完整 `B x 1 x L x L` attention bias
- bottleneck phase 回到标准 decoder attention 形式
- 可以继续使用 flash-attention / SDPA 一类的快速路径，而不是被 4D mask 迫使回退

对应实现：

- [attention_mask.py](../abstract_cot/src/abstract_cot/modeling/attention_mask.py)
- [forward_batch.py](../abstract_cot/src/abstract_cot/training/forward_batch.py)

### 3.4 Stage 1 warm-up

当前 Stage 1 已经按 round 驱动，而不是错误地用 `bottleneck_epochs/distill_epochs`。

当前流程：

```text
初始化 θ^(0)

for t = 1 ... T:
    遍历 D_t,1 的每个训练 batch
        if t == 1:
            当前 batch 使用随机 abstract trace
        else:
            当前 batch 先用当前模型在 x + c 条件下 constrained decoding 生成 trace

        bottleneck 训练拆成两次 forward:
            1. 用 [x; c; z]，只在 z 上算 loss
            2. 用 [x; z; y]，只在 y 上算 loss

    得到 θ_bar^(t)

    遍历 D_t,2 的每个训练 batch
        当前 batch 先用 θ_bar^(t) 在 x 条件下生成 prompt-only trace
        再用 [x; z'; y] 做 self-distillation，得到 θ^(t)
```

当前实现特点：

- `rounds` 是唯一的迭代控制参数
- `D_t,1` 和 `D_t,2` 从本轮数据中切分得到
- 每个 phase 使用 `torch.utils.data.DataLoader` 迭代 `SupervisedSample` batch
- bottleneck / distill 都改为按 batch 在线生成 trace，不再在 phase 开始前一次性预生成整轮 trace
- bottleneck phase 中 `Z` forward 与 `Y` forward 分开 backward，再统一 optimizer step
- 当前训练默认开启 gradient checkpointing
- 当前训练路径已支持 FSDP

### 3.5 当前训练侧的显存优化

为了让 warm-up 能在长序列和多卡场景下继续推进，当前已经接入以下训练实现：

1. Flash Attention 2 路径
   - 模型加载时显式请求 `flash_attention_2`
   - 启动时会打印后端诊断信息，检查环境是否真的可用

2. Sparse logits wrapper
   - 训练只在有监督 token 上计算 lm head
   - 不再为整段前缀 materialize 全量 `B x T x V` logits
   - 当前通过项目内 wrapper 实现，而不是改底层模型 `forward`

3. CUDA 显存观测
   - 支持关键阶段显存日志
   - 支持 PyTorch memory snapshot 采样

4. 最终导出
   - FSDP 下最终模型通过 full state dict gather 后导出
   - 最终模型权重保存为 `model.safetensors`
   - phase checkpoint 仍为 `.pt`

对应实现：

- [model_loader.py](../abstract_cot/src/abstract_cot/modeling/model_loader.py)
- [sparse_lm_wrapper.py](../abstract_cot/src/abstract_cot/modeling/sparse_lm_wrapper.py)
- [sft_trainer.py](../abstract_cot/src/abstract_cot/training/sft_trainer.py)
- [bottleneck_sft.py](../abstract_cot/src/abstract_cot/training/bottleneck_sft.py)

---

## 4. 数据设计

### 4.1 原始 warm-up 数据

当前只使用一个 warm-up 数据集：

- `allenai/Dolci-Think-SFT-7B`

原始样本格式主要是：

```python
{
  "messages": [...],
  "dataset_source": "...",
  "id": "..."
}
```

其中 assistant 内容形如：

```text
<think>
...
</think>

answer
```

### 4.2 预处理后的 warm-up 数据

由于训练时反复做 `messages -> prompt/cot/answer` 解析过慢，当前已增加离线预处理。

预处理输出：

- `server_assets/datasets/Dolci-Think-SFT-7B-cot`

预处理格式：

- 使用 `datasets.save_to_disk(...)`
- 训练时使用 `load_from_disk(...)`

预处理后列：

- `sample_id`
- `prompt`
- `cot`
- `answer`
- `task_type`

注意：

- 预处理阶段不做训练卡数相关的分发
- 训练时再根据实际 `world_size` 分配本轮样本
- 当前 warm-up 预处理已按长度阈值过滤，实际训练使用的是 `8k` 以下样本

对应实现：

- [preprocess_dolci_think_sft.py](../abstract_cot/scripts/preprocess_dolci_think_sft.py)
- [preprocess_warmup_data.sh](../abstract_cot/scripts/preprocess_warmup_data.sh)

---

## 5. 服务器工作流

当前推荐流程：

1. 本地 Git 管理代码
2. 用 `mutagen project start` 同步代码到服务器
3. 服务器用 `conda + uv` 管理环境
4. 模型和数据只在服务器下载
5. 先下载原始数据，再做 warm-up 预处理
6. 再运行 Stage 1 warm-up

关键脚本：

- 环境初始化：[bootstrap_server.sh](../abstract_cot/scripts/bootstrap_server.sh)
- 下载模型和原始数据：[download_qwen_and_data.sh](../abstract_cot/scripts/download_qwen_and_data.sh)
- 预处理 warm-up 数据：[preprocess_warmup_data.sh](../abstract_cot/scripts/preprocess_warmup_data.sh)
- 启动 warm-up：[run_server_warmup.sh](../abstract_cot/scripts/run_server_warmup.sh)

---

## 6. 已验证情况

### 6.1 本地静态验证

已多次通过：

- `python3 -m compileall abstract_cot/src abstract_cot/scripts`
- shell 脚本 `bash -n`

### 6.2 单机 CPU smoke run

已验证：

- 数据解析正确
- bottleneck / distill 两个 phase 可跑
- loss 会下降

### 6.3 单机 8 卡 FSDP smoke run

已验证：

- CUDA 正常可用
- FSDP 可启动
- 一轮最小 Stage 1 可以跑通
- summary 会正确输出 per-rank steps、global steps、loss
- 训练进度条、phase checkpoint、最终模型导出都已接入

### 6.4 当前已完成的一次可用训练

截至当前，已经按现有 warm-up 配置完成过一次：

- 基座模型：`Qwen3-0.6B`
- 数据：`Dolci-Think-SFT-7B-cot`
- 长度过滤：`8k` 以下样本

这说明当前工程链条已经不只是 smoke run，而是能完成一次实际 warm-up 训练。

但这个结果还不能视为论文复现完成。原因很明确：

- `0.6B` 模型规模太小，不能据此判断论文结论是否成立
- 后续至少需要推进到 `4B` 量级，才能开始验证 warm-up only 的有效性

一个已跑通的最小结果示意：

```python
{
  "num_samples_per_rank": 8,
  "rounds": 1,
  "batch_size_per_rank": 2,
  "global_effective_batch_size": 16,
  "round_summaries": [
    {
      "round": 1,
      "D_t1_size": 4,
      "D_t2_size": 4,
      "bottleneck": {"per_rank_num_steps": 2, "global_num_steps": 16, ...},
      "distill": {"per_rank_num_steps": 2, "global_num_steps": 16, ...}
    }
  ]
}
```

---

## 7. 当前主要问题

### 7.1 复现层面还没有触达目标模型规模

当前已经证明工程链能跑，但还没有触达论文复现真正关心的模型尺度。

当前缺口：

- 只完成过 `0.6B` 级别 warm-up
- 还没有完成 `4B` 及以上模型训练

这意味着当前还不能回答：

- warm-up only 是否在论文目标尺度上成立
- abstract reasoning 是否真的需要更大模型容量才会表现出来

### 7.2 数据加载路径仍然偏保守

虽然当前已经采用：

- `rank0 load_from_disk`
- 再 scatter 本轮样本

但这依然是“先可用、后优化”的方案。

后续可能还需要：

- dataset 流式化或更轻的索引式分发
- 更少的 Python object scatter
- 更明确的数据缓存和复用策略

### 7.3 当前还没有正式评测框架

目前训练侧已经基本成型，但评测链仍然空缺。

当前还没有系统化实现：

- warmup-only abstract CoT benchmark eval
- baseline 对比
- trace 输出与 answer 输出的统一记录
- metrics 聚合与按数据集汇总
- 可复现的 eval config / report 产物

---

## 8. 当前连续 latent 分支

除了论文原始离散 abstract token 路线，当前还接入了一个实验性连续 latent 想法：

- 取上一轮最后位置 hidden state
- 若 stop 预测不是 `<endabstract>`，则把该 hidden state 直接作为下一步输入 embedding

当前状态：

- 已有最小实现
- 已支持 SFT/RL 接口雏形
- 还没有接入主 warm-up 训练链

对应实现：

- [recurrent_latent.py](../abstract_cot/src/abstract_cot/continuous/recurrent_latent.py)
- [continuous_experiment.py](../abstract_cot/src/abstract_cot/training/continuous_experiment.py)

这个分支目前是实验性附加路线，不影响离散主线。

---

## 9. 下一步建议

当前最合理的优先级：

1. 推进 `4B` 及以上模型的 warm-up 训练
2. 建立 warmup-only 的正式 eval 框架
3. 先完成 baseline 与 abstract-CoT 的直接对比
4. 在评测链稳定后，再考虑 Stage 2 RL 和更复杂分析实验

不建议当前就做的事：

- 在没有 eval 框架前继续扩展复杂实验矩阵
- 提前把 RL 分支和主线 warm-up 混在一起
- 在 `0.6B` 结果上做过多结论性解释

---

## 10. 当前结论

当前项目已经从“论文 idea 整理”进入“训练链已跑通、接下来要做规模验证和评测框架”的阶段。

更准确地说：

- Stage 1 主流程已经实现
- 服务器训练链已经打通
- `0.6B + 8k以下数据` 已完成一次可用 warm-up 训练
- 当前主要缺口从“能不能训”转为“更大模型是否有效，以及如何系统评测”

因此，下一个对话最适合继续围绕：

- `4B` 量级 warm-up 训练
- warmup-only 的 baseline eval 框架
- benchmark 选择、数据适配与指标定义
- inference / report / trace dump 的统一接口

而不是再回到最早期的系统设计阶段。
