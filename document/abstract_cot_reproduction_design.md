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

当前 on-policy trace 生成已经接入 KV cache，避免每一步重算整段前缀。

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
- 可以继续使用 `sdpa` / flash-attention 一类的快速路径，而不是被 4D mask 迫使回退

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
- FSDP 路径已经可用

对应实现：

- [bottleneck_sft.py](../abstract_cot/src/abstract_cot/training/bottleneck_sft.py)
- [sft_trainer.py](../abstract_cot/src/abstract_cot/training/sft_trainer.py)
- [run_warmup_mvp.py](../abstract_cot/scripts/run_warmup_mvp.py)

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

### 7.1 大样本训练时仍然不稳定

当 `max_samples` 增大后，Stage 1 在第一轮 bottleneck phase 仍可能：

- 卡在第一个 step 前后
- 某个 rank 被 `SIGKILL`
- 导致 `torchrun` 终止其他进程

当前判断：

- 不是原先那种“每卡样本数不一致”的问题
- 更可能是数据加载和第一步前向的内存 / CPU 峰值问题

### 7.2 `load_from_disk` 仍需进一步优化

虽然当前已经改成：

- `rank0 load_from_disk`
- 再把本轮所需样本 scatter 给其他 rank

但这仍然不是最终高效方案。当前实现属于“先稳住内存，再谈吞吐”的过渡版本。

后续可能需要：

- 更明确的主进程数据服务式加载
- 更轻的 rank 间样本下发方式
- 更少的 Python object 传输

### 7.3 Bottleneck phase 仍然偏重

虽然 mask 已经不再以 `L x L bool` 常驻 feature，但 bottleneck phase 仍然比 distill phase 重很多。

原因：

- bottleneck attention 仍是自定义 4D additive mask 路径
- 很难走最优注意力 kernel
- 序列较长时仍可能产生较大 CPU/GPU 开销

### 7.4 还没有正式评测链

当前只有训练 loss 和 summary，没有：

- benchmark eval
- warm-up 前后输出对比
- abstract trace 质量检查脚本
- truncation / permutation / frequency 分析

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

1. 继续稳定 Stage 1 大样本训练
2. 解决多卡下数据加载与首步 bottleneck phase 的资源峰值问题
3. 增加 warm-up 后的最小 inference / eval 脚本
4. 在 Stage 1 稳定后再进入 Stage 2 RL

不建议当前就做的事：

- 直接扩大到更大模型规模
- 提前做复杂分析实验
- 把连续 latent 分支混入主训练路径

---

## 10. 当前结论

当前项目已经从“论文 idea 整理”进入“可运行但仍需稳定化”的阶段。

更准确地说：

- Stage 1 主流程已经实现
- 服务器训练链已经打通
- 小规模多卡 smoke run 已通过
- 主要风险从“功能缺失”转移到了“规模化训练时的数据与内存效率”

因此，下一个对话最适合继续围绕：

- Stage 1 大样本稳定性
- 数据加载优化
- bottleneck phase 的性能与内存问题
- 最小评测链

而不是再回到最早期的系统设计阶段。
