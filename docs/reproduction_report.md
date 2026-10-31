# 复现与调优说明（reproduction_report）

> 对应提交权重：`03_模型权重/checkpoint/`（变体 `control`）
> 队伍 ID：1385542

---

## 1. 训练方式

从 `robbyant/lingbot-vla-v2-6b`（HF revision `11c703bf6a5c1f45b3b69168482da11fdbba53d7`）初始化，
在 RoboTwin 2.0 Aloha-AgileX 50 任务 clean 数据上做**全参数 SFT**（非 LoRA），采用 FSDP2 数据并行，
`ckpt_manager=dcp` 存盘并导出为 HF safetensors 格式。

- 数据：50 任务 × 50 条 clean 示范（`lingbot-c2-clean50-repaired-20261022:2`），未使用任何 randomized 数据
- 硬件：8 × NVIDIA A100-SXM4-80GB
- 规模：4,296 个 optimizer update，global batch 1024（micro 8/GPU × grad accum 16），seed 42
- 墙钟：约 29 小时
- 精度：`enable_fp32=true`（action expert 使用 FP32）

## 2. 评测方式

RoboTwin 2.0 闭环评测，模型以 WebSocket 策略服务运行，评测客户端在仿真中执行动作并调用任务内置
`check_success()` 判定成败（不做人工复核）。

- 任务：Aloha-AgileX 50 任务
- Setting：`demo_clean`（seen 指令）与 `demo_randomized`（unseen 指令 + 随机化场景）
- 回合数：每任务每 setting 目标 100 回合
- 种子：三组权重使用**同一套** episode 种子（`100001` 起递增），因此可做逐回合配对比较
- 推理：FP32，`use_length=25`，`expert_check=true`

**评测完成度**：全量协议 50 任务 × 2 setting × 100 回合 = 30,000 回合，**全部完成**。

| Setting | 测试次数 | 成功次数 | 成功率 |
|---|---|---|---|
| clean（seen 指令） | 5,000（50 任务 × 100） | 2,964 | **59.28%** |
| randomized（unseen 指令） | 5,000（50 任务 × 100） | 1,464 | **29.28%** |
| 两场景平均 | — | — | **44.28%** |

每个任务的 `attempts` 均为 100，逐回合明细（100 个 jsonl，每任务 100 条不重复、按序 seed）随评测交付记录留存。

## 3. 参数配置

| 类别 | 配置 |
|---|---|
| 模型 | `LingbotVLAV2Config`，`post_training=true`，`adanorm_time=true`，`moe_implementation=fused` |
| MoE | 32 experts，top-k 4，`token_moe_layers` 0–35，`routed_scaling_factor=4.0` |
| 损失 | flow-matching（`loss_type=fm`），序列级辅助损失 1e-3，router z-loss 1e-4 |
| 归一化 | per-joint `bounds_99_woclip`，共用 `c2_all_clean_norm.json`（SHA-256 `0883c665…b486`） |
| 并行 | FSDP2，`vlm_fsdp=true`，`enable_full_shard=false` |
| 注意力 | `flex_cached`（联合注意力），VLM/ViT 用 `flash_attention_2` |
| 优化 | lr 1e-4，constant decay |

完整配置见 `02_代码材料/configs/vla/robotwin/robotwin.yaml`。

## 4. 模型改动

**无结构改动**：主干、视觉编码器、action expert 与 MoE 结构均与基座一致，仅做全参数后训练。
未引入外部数据、未引入新的预训练模块；基座自带的深度/视频双教师蒸馏配方保持原样（教师全程冻结）。

## 5. 调优思路与消融结果

围绕两个旋钮做了三组同配置消融（训练数据、步数、seed、硬件完全一致）：

| 变体 | `state_noise` | `gripper_weight` |
|---|---|---|
| **control（提交）** | 0 | 1 |
| state_noise | 0.05 | 1 |
| gripper2 | 0 | 2 |

**结论（基于 8,662 个配对回合的 McNemar 检验）：**

| 对比 | 配对回合 | 成功率 A / B | p 值 | 判定 |
|---|---|---|---|---|
| control vs state_noise | 2,577 | 53.9% / 53.9% | 1.00 | 完全打平 |
| control vs gripper2 | 2,958 | 48.2% / 47.5% | 0.46 | 无显著差异 |
| state_noise vs gripper2 | 2,490 | 53.7% / 51.5% | 0.036 | state_noise 更优 |

分场景：

| 场景 | control | state_noise | gripper2 |
|---|---|---|---|
| demo_clean | 65.0% | 66.0% | 61.7% |
| demo_randomized | 32.9% | 30.9% | 32.5% |

- `state_noise`（proprioception 加性噪声 0.05）与 control 在总体和两个场景上都无显著差异（p ≥ 0.22）
- `gripper2`（夹爪损失加权 ×2）在 clean 上显著劣于 state_noise（p=0.0014），总体与 control 的差距未达显著
- 因此选择 **control**：与最优并列、无未验证改动、迁移风险最低

## 6. 复现步骤

1. 安装环境：`bash tools/create_train_env.sh`（Python 3.12 / PyTorch 2.8.0 / flash-attn 2.8.3）
2. 准备数据：RoboTwin 2.0 Aloha-AgileX 50 任务 clean 数据，转 LeRobot v3，生成 50 行训练清单
3. 计算/复用归一化统计：`assets/norm_stats/robotwin.json`（本次使用交付文件 `c2_all_clean_norm.json`）
4. 训练：`bash train.sh tasks/vla/train_lingbotvla.py configs/vla/robotwin/robotwin.yaml ...`（见 README 第 4 节）
5. 评测：`bash eval.sh serve 03_模型权重/checkpoint 13400`，另开终端 `bash eval.sh run <task> <setting> 13400 100`
6. 汇总：把每个任务的结果写进 `results.json`（本队工具 `make_results_json.py` 可直接从 run 目录生成）

## 7. 已知限制

- 训练与推理均为 FP32；若用 BF16 复测，成功率可能与本报告数字有系统性差异
- 三组消融的配对比较基于三组共同完成的任务子集，样本量见第 5 节表格
- 评测所用末端位姿规划器配置需与官方复测环境对齐（见 README 第 6 节）
- 官方复测为统一环境、统一配置的独立评测，本报告的 `results.json` 仅作初赛提交依据
