---
trigger: always_on
---

请为我设计并生成一套全面、系统、由浅入深的强化学习（Reinforcement Learning, RL）学习内容，直接适配填充到教学/演示/知识管理界面中使用。内容必须以以下最权威来源为主要依据：

《Reinforcement Learning: An Introduction》（Sutton & Barto, 第2版，在线免费版 http://incompleteideas.net/book/the-book-2nd.html）
《Reinforcement Learning: Theory and Algorithms》（Agarwal, Jiang, Kakade, Sun 等，https://rltheorybook.github.io/）
《A Course in Reinforcement Learning》（Bertsekas, 2024–2025版）
OpenAI Spinning Up in Deep RL（https://spinningup.openai.com/）
经典课程大纲：Stanford CS234、Berkeley Deep RL、Georgia Tech CS7642、MIT 相关讲义（2024–2025最新版）
最新会议论文与综述（NeurIPS/ICLR/ICML 2024–2025 RL oral/spotlight/highlight 论文、RLChina 2025 workshop 等）

严禁凭空捏造算法、公式、收敛证明或已过时/失效的变体。所有数学推导、伪代码、复杂度分析、实验设置均应与上述来源一致，并适当注明参考章节/论文/链接。
总体要求

内容深度与广度：从零基础（马尔可夫决策过程入门、基本概念）开始，逐步推进到中级（表格方法、函数逼近）、深度强化学习（policy gradient、actor-critic、model-based）、高级理论（收敛性、样本复杂度）、前沿方向（RLHF、DPO、offline RL、multi-agent、meta-RL、reasoning agents、绿色/社会对齐 RL、大模型时代 RL）。宁可内容存在适度冗余与多视角重复讲解（不同教材/论文观点对比），也绝不允许遗漏核心知识点与当前前沿。同一主题可在入门、中级、高级章节多次出现，但每次侧重（直观解释、数学证明、代码实现、实验规模、最新变体）需明显递增。
讲解形式与要求：
每个重要概念/算法需包含：
正式定义、动机与历史背景
与其他方法对比（on-policy vs off-policy、value-based vs policy-based、model-free vs model-based 等）
数学分析（Bellman 方程、价值函数定义、策略梯度定理、收敛性证明、样本复杂度界，必要时使用 LaTeX 公式）
正确性/最优性/遗憾界证明（循环不变式、Martingale、concentration 不等式等）
典型应用场景（游戏、机器人、推荐、LLM 对齐、金融、自动驾驶等）

必须提供大量伪代码 + 可直接运行的代码示例（优先 Python + PyTorch/Gymnasium/MuJoCo/Procgen 等环境；包含完整 import、超参、训练循环、评估曲线），并附上预期输出（reward curve、episode length、胜率、样本效率对比等）。
对于复杂或前沿机制（例如：advantage estimation 与 GAE、PPO clip 机制、SAC entropy 正则、Dreamer world model 滚动、RLHF 中的 reward modeling、DPO 隐式奖励、offline RL 的分布修正、multi-agent 的 centralized training decentralized execution、meta-RL 的任务分布采样、reasoning-time RL 的 search vs learning 权衡等），强烈建议采用交互式动画、动态示意图、步进可视化进行讲解（例如：价值迭代收敛动画、策略梯度蒙特卡洛采样过程、actor-critic bootstrapping、PPO 比率裁剪边界、world model 预测轨迹 rollout、preference dataset 中的 Bradley-Terry 模型、Pareto front 逼近、多智能体通信与博弈树展开等）。若当前界面支持嵌入动画、Mermaid 图、可交互 Gym 环境可视化或类似组件，请优先采用此类形式。

结构要求：
首先生成一份非常详细的分级目录（建议采用 Chapter 0、Chapter 1 … 形式），层级至少到三级（大章 → 小节 → 具体算法/证明/变体/前沿扩展）。
目录需清晰体现由简入深 + 前沿跟踪：基础 MDP 与经典方法 → 表格 RL → 函数逼近与深度 RL → 策略优化 → 高级 actor-critic 与 model-based → 探索与样本效率 → 多目标/离线/元学习 → 多智能体 RL → 大模型时代 RL（RLHF、DPO、reasoning agents） → 理论前沿与可靠性 → 绿色/社会对齐/实际部署。
目录生成后，一个章节一个章节地详细展开内容（不要一次性输出全部），每个章节内部建议再细分小节，并保持相对统一的讲解结构：概念与动机 → 数学推导与证明 → 伪代码 → 实现代码（含环境与超参） → 实验结果与曲线 → 常见问题、失效模式与调试 → 最新变体与前沿论文 → 扩展阅读（Sutton/Barto 章节、论文 arXiv 链接）。

重点覆盖但不限于以下前沿与高级主题（应有独立或深度章节，优先引用 2024–2025 最新进展）：
基础：MDP、Bellman 最优性、动态规划、Monte Carlo、TD(λ)、SARSA、Q-learning
策略梯度：REINFORCE、baseline、Actor-Critic、A2C/A3C、GAE
近端优化：TRPO、PPO、Trust Region 方法、Natural PG
最大熵 RL：Soft Actor-Critic (SAC)、最大熵框架
Model-based：Dyna、MBPO、DreamerV3、世界模型滚动与想象
探索：count-based、curiosity、random network distillation、Go-Explore、UER
Offline RL：BCQ、CQL、TD3+BC、保守 Q-learning、分布修正
RLHF 与对齐：reward modeling、PPO + KL、正则化、DPO、KTO、SPIN、迭代 DPO
多智能体：MARL、centralized vs decentralized、CTDE、QMIX、MAPPO、self-play、emergent behaviors
元强化学习：MAML、PEARL、任务分布、few-shot adaptation
推理时代 RL：process reward vs outcome reward、search-augmented RL、o1-like reasoning-time scaling
多目标 RL（MORL）：Pareto front 逼近、scalarization、vectorized value function
绿色 RL 与可靠性：样本效率、计算碳足迹、robust RL、对抗攻击、分布漂移
实际系统：机器人（Dexterity、locomotion）、游戏（Atari、StarCraft、Procgen）、LLM agent、自动驾驶

其他细节偏好：
优先使用现代环境与库：Gymnasium、MuJoCo、Atari、Procgen、DM Control、PettingZoo（MARL）、HuggingFace RL 工具等。
在合适位置插入小练习（手推公式、改写算法）、复现建议（Spinning Up 风格实验）、思考题（为什么 PPO clip 有效？DPO 如何避免显式 reward？offline RL 的 deadly triad 如何缓解？）。
鼓励展示性能对比表格（样本效率、最终分数、稳定性）、收敛曲线、ablation study。
语言风格：正式、严谨、结构清晰、逻辑严密，如同撰写给 AI 研究员、博士生或工业界强化学习工程师的高质量内部培训教材。


请先输出详细的章节大纲（包含 Chapter 编号、章节标题、二级/三级子标题），待我确认或提出修改意见后再逐章详细展开具体内容。
（可根据 Sutton & Barto 第2版最新勘误、2025年 NeurIPS/ICLR/ICML RL 论文趋势、OpenAI/DeepMind/Google 最新博客与技术报告进行适度微调，但核心由浅入深结构与前沿跟踪要求保持不变。）