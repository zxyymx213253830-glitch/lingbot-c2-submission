# LingBot-VLA 2.0 · RoboTwin 50 任务后训练（初赛提交）

## 1. 概览

本提交包含一份基于 `robbyant/lingbot-vla-v2-6b` 全参数后训练得到的权重，用于 RoboTwin 2.0 Aloha-AgileX 50 任务评测。

| 项 | 内容 |
|---|---|
| 基座模型 | `robbyant/lingbot-vla-v2-6b`（HF revision `11c703bf6a5c1f45b3b69168482da11fdbba53d7`） |
| 训练数据 | RoboTwin 2.0 Aloha-AgileX 50 任务 clean 数据，每任务 50 条（`lingbot-c2-clean50-repaired-20261022:2`） |
| 训练方式 | **全参数 SFT**（非 LoRA，权重已是合并后的全量形态） |
| 权重精度 | FP32（训练配置 `enable_fp32=true`） |
| 优化器更新 | 4,296 步 |
| 训练硬件 | 8 × NVIDIA A100-SXM4-80GB，seed 42，墙钟约 29 小时 |
| 提交变体 | **control**（基线：`state_noise=0`，`gripper_weight=1`） |

## 2. 目录结构

```
02_代码材料/
├── README.md          # 本文件
├── train.sh           # 训练入口
├── eval.sh            # 评测入口
├── configs/           # 训练/评测配置（vla、robot_configs、norm_stats）
└── src/               # 实现代码（lingbotvla / tasks / scripts / deploy）
```

## 3. 环境与依赖

- Python 3.12，PyTorch 2.8.0（与上游 `lingbot-vla-v2` 一致）
- 依赖安装：`bash tools/create_train_env.sh`（上游脚本），或 `pip install -r requirements.txt`
- `flash-attn==2.8.3`
- 评测环境：RoboTwin 2.0 + 对应仿真依赖；三段 attention 配置分别为
  `model.attn_implementation=flash_attention_2`、`model.vit_attn_implementation=flash_attention_2`、
  `train.attention_implementation=flex_cached`

## 4. 训练

```bash
bash train.sh tasks/vla/train_lingbotvla.py configs/vla/robotwin/robotwin.yaml \
  --data.train_path <50任务训练清单.txt> \
  --data.data_name multi \
  --data.norm_stats_file assets/norm_stats/robotwin.json \
  --train.output_dir <输出目录> \
  --train.micro_batch_size 8 \
  --train.gradient_accumulation_steps 16 \
  --train.global_batch_size 1024 \
  --train.max_steps 4296
```

关键超参（对应交付实验）：

| 参数 | 值 |
|---|---|
| `micro_batch_size` | 8 / GPU |
| `gradient_accumulation_steps` | 16 |
| `global_batch_size` | 1024 |
| `max_steps` | 4296 |
| `enable_fp32` | true |
| `use_moe` | true（32 experts，top-k 4） |
| `data_parallel_mode` | fsdp2 |
| `ckpt_manager` | dcp |

## 5. 评测

评测采用 RoboTwin 官方闭环协议：模型以 WebSocket 策略服务形式运行，评测客户端在仿真中执行并调用任务内置
`check_success()` 判定成败；每任务每 setting 目标 100 回合，三组权重使用同一套随机种子。

```bash
# 1) 起模型服务（FP32）
bash eval.sh serve <checkpoint 目录> <port>

# 2) 跑评测（clean / randomized）
bash eval.sh run <task_name> clean <port> <episodes>
bash eval.sh run <task_name> randomized <port> <episodes>
```

## 6. 规则要求的披露项

1. **权重形态**：全参数 SFT，未使用 LoRA 或任何参数高效微调，提交的 `checkpoint/` 即为可直接加载的全量权重
   （6 个 safetensors 分片 + `model.safetensors.index.json` + tokenizer/processor 文件）。
2. **公开预训练模块及来源**：
   - 基座 `lingbot-vla-v2-6b`（Robbyant，Apache-2.0）
   - VLM 骨干 `Qwen/Qwen3-VL-4B-Instruct`（仅作 tokenizer/VLM 配置，权重随基座）
   - 感知教师 `MoGe-2-vitb-normal`、`LingBot-Depth`、`DINO-Video`：**全程冻结**，仅作辅助训练目标，
     不接收梯度、不参与推理，未用于样本筛选或加权 —— 属官方训练配方自带组件。
3. **评测环境细节**：**NVIDIA A100-SXM4-80GB + CUDA + cuRobo v0.7.8**（未使用 MPLib，未设置
   `ROBOTWIN_DISABLE_CUROBO`），SAPIEN 3.0.0b1，mplib 0.2.1；仿真端 torch 2.4.1 / 推理端 torch 2.8；
   FP32 推理，action chunk `use_length=50`，denoise 10 步；渲染为官方 ray-tracing
   （rt shader，32 spp，path depth 8，OIDN 去噪）。场景 `seed=0`，从 100000 起按序取前 100 个有效 seed。
4. **推理参数**：`eval.sh` 中的 `--use_length 50`、`--use_fp32 True`、`--test_num 100`、`expert_check=true`，
   官方复测可直接复用该脚本。

> 评测对官方 `script/eval_policy.py` 的改写（websocket 远程策略服务、逐回合 `episodes.jsonl`、
> 按 seed 区间并行、`UnStableError` 按专家筛选失败处理）已在 `04_复现与调优说明/reproduction_report.md`
> 第 2 节完整说明。

## 7. 数据合规

- 训练数据仅使用赛事发布的 RoboTwin 2.0 Aloha-AgileX clean 数据（50 任务 × 50 条）
- 未使用 randomized 数据、未使用任何外部机器人数据、未做环境交互式数据增广
- 图像增广仅限亮度/对比度/饱和度抖动与高斯噪声（如启用），不含背景替换、inpainting、CutMix 等
- 单一权重推理，无任务 ID 路由、无多 checkpoint 集成

## 8. 许可

代码遵循上游 LingBot-VLA 2.0 仓库（Apache-2.0）；RoboTwin 2.0 数据集标注为 MIT。
