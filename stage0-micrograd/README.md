# Stage 0 — Kill the black box: backprop by hand

**目标:** 从零手写一个标量自动微分引擎(micrograd)+ 一个小 MLP,亲手把反向传播实现一遍。
这样后面所有抽象(autograd / optimizer / loss)对你都不再是黑箱——因为你**亲手重造过一次**。

> 这是整条路的地基,也是把"会读代码"炼成"懂底层"的第一道相变。

---

## 铁律:AI-OFF 的精确边界

这条决定项目成败。边界要分清——**不是"不许用任何资料",是"不许让 AI 替你思考/写代码"**:

| ✅ 允许(这是学习) | ❌ 禁止(这是 vibe coding,要杀的就是它) |
|---|---|
| 看 Karpathy 的 micrograd 讲座、3Blue1Brown | 让 AI(我 / ChatGPT / Copilot)写 `engine.py` / `nn.py` / `train.py` |
| Google 一个导数公式、查文档 | 把报错贴给 AI,让它把改好的代码递给你 |
| 看完讲座后**自己默写**实现 | 抄讲座/别人仓库的代码粘进来(看完→关掉→自己写) |
| 卡 ≥30 分钟后找我要一个**提示** | 找我要**答案代码** |

**我(Claude)在这一阶段的角色:** 测试台 + 教练 + 验收官。我**不写**你的学习代码。你卡死超过 30 分钟,可以来找我——我给你一个**方向性提示**(去看哪段、哪个概念错了),不给你代码。

自检标准一句话:**"把所有 AI 关掉,我能从空文件重做一遍吗?"** Day 11 会真的考这个。

---

## 资源(看这些,然后关掉自己写)

1. **Karpathy《Neural Networks: Zero to Hero》Lecture 1** — "The spelled-out intro to neural networks and backpropagation: building micrograd"(YouTube,~2.5h)。这是主线。
2. `karpathy/micrograd` 仓库 —— **只在你自己尝试之后**再看,用来对照。
3. **3Blue1Brown** "Neural networks" 系列(backprop 那几集)—— 视觉直觉。
4. 递归补知识:某个导数想不通,就让 AI **只推导那一个表达式**、一步步解释,然后**自己再不看推一遍**。(注意:推导讲解 = 允许;让它写引擎 = 禁止。)

---

## 14 天计划(假设 ~20–30 小时/周)

**第一周 —— 引擎(engine.py)**
- **Day 1** 看 Lecture 1 前半 + 3b1b backprop。手写笔记,先不碰键盘。
- **Day 2** 纸上推导:单个神经元 `y = tanh(w·x + b)`,手推 `dy/dw_i` 和 `dy/db`。然后默写 `Value.__init__`、`__add__`、`__mul__`。
- **Day 3** 实现 `__pow__`、`__truediv__`、`__neg__`、`__sub__`、反射运算、`relu`、`tanh`、`exp`。
- **Day 4** 实现 `backward()`(拓扑排序 + 反向遍历)。跑 `python test_engine.py`,先让 forward Δ 过。
- **Day 5** 死磕梯度,直到 `test_engine.py` **PASS**(<1e-6)。这天最难,正常。
- **Day 6** 缓冲 / 把某个你最没把握的算子**不看**重写一遍。
- **Day 7** 休息或补进度。

**第二周 —— 网络 + 训练 + 内化**
- **Day 8** 实现 `nn.py`(Neuron / Layer / MLP / parameters)。
- **Day 9** 写 `train.py`:载入 two-moons、建 MLP、写训练循环 + SGD。
- **Day 10** 调到 **accuracy > 0.95**,调 lr/epochs,`plot_decision_boundary` 出图。
- **Day 11** **检索测试(关键):** 关掉一切,空文件,一口气重新实现 `Value` + `backward`。卡住 = 说明没真懂,回去补。
- **Day 12** 写一页 writeup(放进 `WRITEUP.md`):用**你自己的话**讲清反向传播 + 贴决策边界图。
- **Day 13** 缓冲。
- **Day 14** 复盘 + 我给你做一次**口试**(你不看笔记,跟我讲清 `backward()` 里的拓扑排序和链式法则)。

---

## 验收(4 条全过,才解锁 Stage 1)

1. `python test_engine.py` → **PASS**(worst forward Δ 和 worst gradient Δ 都 < 1e-6)
2. `python train.py` → 打印 **final accuracy > 0.95**
3. **Day 11 检索:** 能从空文件一口气重写 `Value` + `backward`
4. **口试:** 不看笔记,能跟我讲清 `backward()` 的拓扑排序 + 链式法则,以及 `tanh`/`relu` 的局部梯度怎么来的

跑通后回来找我,我做口试 + 验收,然后我们开 **Stage 1(nanoGPT 复现 GPT-2)**。

---

## 怎么开始(今天)

```
cd C:\Users\zexi\frontier-ai\stage0-micrograd
python test_engine.py        # 现在会报 "not implemented yet" —— 这就是你的起跑线
```

看 Day 1 的视频,然后打开 `engine.py` 开干。卡 30 分钟以上再来找我要提示。
