# LingBot-VLA 2.0 · RoboTwin 2.0 50 任务后训练（蚂蚁灵波挑战赛 C2）

本仓库是 **蚂蚁灵波具身大模型挑战赛（天池 532514）** 初赛提交材料的公开归档，队伍 ID `1385542`。

- 基座：`robbyant/lingbot-vla-v2-6b`（HF revision `11c703bf6a5c1f45b3b69168482da11fdbba53d7`）
- 数据：RoboTwin 2.0 Aloha-AgileX 50 任务 clean 数据（每任务 50 条），未使用 randomized 数据
- 方法：全参数 SFT（非 LoRA），8 × A100-SXM4-80GB，4,296 次 optimizer update，seed 42，约 29 小时
- 评测：RoboTwin 2.0 闭环，NVIDIA A100 + CUDA + cuRobo v0.7.8，FP32，每任务每 setting 100 回合

## 结果（control 变体）

| Setting | 测试次数 | 成功次数 | 成功率 |
|---|---|---|---|
| clean（seen 指令） | 5,000（50 任务 × 100） | 2,964 | **59.28%** |
| randomized（unseen 指令） | 5,000（50 任务 × 100） | 1,464 | **29.28%** |
| 平均 | 10,000 | 4,428 | **44.28%** |

逐任务成功率见 [`results/per_task.csv`](results/per_task.csv)，官方格式结果见 [`results/results.json`](results/results.json)。

## 三组消融

围绕两个旋钮做了同配置对照（训练数据、步数、seed、硬件完全一致）：

| 变体 | `state_noise` | `gripper_weight` | clean | randomized |
|---|---|---|---|---|
| **control（提交）** | 0 | 1 | 65.0% | 32.9% |
| state_noise | 0.05 | 1 | 66.0% | 30.9% |
| gripper2 | 0 | 2 | 61.7% | 32.5% |

（上表为三组共同完成的任务子集上的配对比较，单位成功率；完整协议下 control 为 59.28% / 29.28%）

配对 McNemar 检验（8,662 个同任务同 seed 的配对回合）：

| 对比 | 配对回合 | 成功率 A / B | p 值 | 判定 |
|---|---|---|---|---|
| control vs state_noise | 2,577 | 53.9% / 53.9% | 1.00 | 完全打平 |
| control vs gripper2 | 2,958 | 48.2% / 47.5% | 0.46 | 无显著差异 |
| state_noise vs gripper2 | 2,490 | 53.7% / 51.5% | 0.036 | state_noise 更优 |

结论：状态扰动（0.05）与基线统计上打平；夹爪损失加权（×2）在 clean 上显著更差，是一个负结果。最终提交 **control**——与最优并列且无未验证改动。

## 目录结构

```
.
├── src/                 # LingBot-VLA 2.0 实现代码（lingbotvla / tasks / scripts / deploy）
├── configs/             # 训练与评测配置（vla、robot_configs、norm_stats 引用）
├── results/
│   ├── results.json     # 官方模板格式的 50 任务 × 2 setting 结果
│   └── per_task.csv     # 逐任务成功率
├── docs/
│   ├── reproduction_report.md   # 复现与调优说明（六节）
│   └── EVAL_README.md           # 评测协议、环境与对官方脚本的修改
├── CODE_README.md       # 提交材料的 README（含四项强制披露）
├── train.sh
└── eval.sh
```

## 复现步骤

1. 安装环境：`bash tools/create_train_env.sh`（Python 3.12 / PyTorch 2.8.0 / flash-attn 2.8.3）
2. 准备数据：RoboTwin 2.0 Aloha-AgileX 50 任务 clean 数据，转 LeRobot v3，生成 50 行训练清单
3. 训练：`bash train.sh tasks/vla/train_lingbotvla.py configs/vla/robotwin/robotwin.yaml …`（见 `CODE_README.md`）
4. 评测：`bash eval.sh serve <checkpoint> 13400`，另开终端 `bash eval.sh run <task> <setting> 13400 100`
5. 汇总：`results/results.json` 由逐回合 `episodes.jsonl` 汇总生成

## 关于权重

6B 参数的 FP32 checkpoint 约 **23.8 GB**（6 个 safetensors 分片 + tokenizer/processor），超出 Git 仓库承载范围，未包含在本仓库中。需要权重请联系队伍，或从赛事的模型托管位置获取。

## 已知限制

- 训练与推理均为 FP32；BF16 复测可能有系统性差异
- 评测仅在 control 变体上跑满官方全量协议（30,000 回合）；另两组为共同任务子集上的配对比较
- 官方复测使用统一环境与配置，独立于本仓库记录的自测数字

## 许可与致谢

代码基于 [LingBot-VLA 2.0](https://github.com/Robbyant/lingbot-vla-v2)（Apache-2.0）与 [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) 构建，评测协议遵循赛事官方规范。`src/` 中的 `MoGe`、`lingbot-depth`、`dino_video` 为上游仓库的第三方组件，版权归各自作者所有。
