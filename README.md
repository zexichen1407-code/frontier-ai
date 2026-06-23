# frontier-ai — 进入前沿 AI 的项目优先之路

目标(已校准): **进中国前沿实验室(DeepSeek / Moonshot / Qwen / 字节 Seed / Zhipu / MiniMax)
或应用AI/Agent/推理工程岗,2–3 年;并在开源上成为无法被忽视的 AI builder。**
方法: Gabriel Petersson 式「先做、撞墙、补知识、靠作品爬团队阶梯」,但用可数值验证的标准卡深度。

> ⚠️ 压倒一切的前提:把主循环从「vibe coding / AI 辅助做 app」换成**深度练习**。
> 否则,诚实结论是「永远到不了前沿深度」。计时只在你换掉练习内容那一刻开始。

---

## AI-OFF 契约(贯穿所有 Stage)

- **允许**:看课、读论文/文档、Google、看完资料后自己默写、卡 30 分钟后找 Claude 要**提示**。
- **禁止**:让任何 AI 写/改你的「学习代码」,或把报错丢给 AI 换回改好的代码。
- 每次学习的检验:**「把模型关掉,我能重做一遍吗?」**
- Claude 的角色:**测试台 + 教练 + 验收官,不是 coder。** 每个 Stage 都有可数值验证的 done-when。

---

## 6 节点阶梯(~9–12 个月)

| | 节点 | 驱动项目 | 可验证的 done-when | 状态 |
|---|---|---|---|---|
| 0 | 杀死黑箱:手写反向传播 | 从零写 micrograd + MLP 训 two-moons | 梯度对 PyTorch <1e-6;accuracy >95% | ▶ 进行中 |
| 1 | 真预训练一个 Transformer | nanoGPT 复现 GPT-2(124M) | loss 达参考值;手算参数/FLOP/显存 误差<10% | ⬜ |
| 2 | 让它快:推理/GPU/精度 | 自写推理引擎(KV cache/投机解码) | 各项优化前后 tokens/s;说清 memory- vs compute-bound | ⬜ |
| 3 | 后训练:SFT→DPO→GRPO | 带 verifier 的任务上跑完整后训练 | pass-rate 逐级提升;手写 GRPO loss 与 TRL 一致 | ⬜ |
| 4 | 扩出去:多卡并行 | DP→FSDP→TP 训单卡放不下的模型 | scaling 效率曲线 + 正确瓶颈解释 | ⬜ |
| 5 | 复现者→研究者 | autoresearch 跑受控实验出原创小结果 | 公开 repo+writeup,带 baseline 和误差棒 | ⬜ |

**信号门槛(2026):** 套壳 app = 负信号。挑 1 个深做胜十个 app ——
modded-nanogpt speedrun 记录 / Prime Intellect 环境 / vLLM-SGLang merged PR / 命名新 eval。

**最高杠杆的一个动作:** 每周固定 **AI-off block** 从零手写;并在 3–6 个月内产出 **1 个外部可验证的深度作品**并公开。

---

## 当前

- **Stage 0 进行中** → 见 [`stage0-micrograd/README.md`](stage0-micrograd/README.md)
- 环境:Python 3.12 / numpy / sklearn / matplotlib / torch(CPU)已就绪。

(对应记忆:`frontier-ai-career-path`)
