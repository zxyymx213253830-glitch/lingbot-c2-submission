# 评测复现说明（LingBot-VLA 2.0 C2，RoboTwin 2.0 Aloha-AgileX 50 任务）

## 评测协议
- 50 个任务 × 2 个 setting：`clean`（demo_clean，seen 指令）、`randomized`（demo_randomized，unseen 指令），每个任务 100 次 policy 测试。
- 场景 seed：`seed=0`，场景从 100000 开始按顺序取，先用专家轨迹筛选，取前 100 个有效 seed（与官方 `eval_policy.py` 规则一致）。
- 推理：FP32，action chunk 50（`--use_length 50`），denoise 10 步，单份权重，不做多 seed 投票，也不按任务路由。
- 渲染：官方光线追踪（`rt` shader，32 spp，path depth 8，OIDN 去噪），不做简化。
- 评测环境：NVIDIA A100-SXM4-80GB，CUDA + cuRobo v0.7.8，SAPIEN 3.0.0b1，mplib 0.2.1，torch 2.4.1（仿真端）/ torch 2.8（推理端），Python 3.10 / 3.12。
- RoboTwin 资产：HuggingFace `TianxingChen/RoboTwin2.0` revision `3dc3b798668feb99ac61cc9086d84cbcc3d79186`（background_texture / embodiments / objects，均做了 SHA-256 校验）。

## 运行方式
1. 推理服务：`python -m deploy.lingbot_vla_v2_policy --model_path <ckpt> --port <port> --use_length 50 --use_bf16 False --use_fp32 True --use_compile True`，环境变量 `LINGBOT_DENOISE_STEPS=10`。
2. 仿真客户端：`script/eval_policy_client_lingbotvla.py --config policy/ACT/deploy_policy.yml --overrides --task_name <task> --task_config demo_clean|demo_randomized --seed 0 --instruction_type seen|unseen --host <host> --port <port> ...`
3. 逐回合结果写入 `episodes.jsonl`（seed、成功与否、指令）；`results.json` 按官方模板汇总。

## 对 RoboTwin 官方评测代码的修改
文件：`script/eval_policy_client_lingbotvla.py`（由官方 `script/eval_policy.py` 改写）
1. 策略通过 websocket 连接远程推理服务（`WebsocketClientPolicy(host, port)`），不在仿真进程内加载模型。
2. 每个仿真进程使用独立的 warp / torch kernel 缓存目录，避免并发编译冲突；不改变任何仿真参数。
3. 逐回合写 `episodes.jsonl`，支持从已记录的最后一个 seed 续跑。
4. 为并行加速，同一任务按场景 seed 区间切成若干段，在多张 GPU 上并行评测（环境变量 `C2_SEED_START` / `C2_SEED_END`）。每个 seed 的场景只由 seed 决定，与是否切分无关。合并时按 seed 排序，取从 100000 起的前 100 个有效 seed，结果与顺序评测使用的场景集合一致。
5. 个别 seed 通过了专家筛选，但在为 policy 回合重建场景时触发 `UnStableError`（物体不稳定）。官方脚本遇到这种情况会直接崩溃退出。这里按专家筛选失败的 seed 处理：跳过该 seed、不计入 100 次，日志记为 `POLICY_SETUP_UNSTABLE`。例：`dump_bin_bigbin` randomized 的 seed 100062。

官方安装步骤中的标准补丁照常应用：mplib `planner.py` 去掉 `or collide` 判断；SAPIEN `urdf_loader.py` 以 utf-8 读取 URDF。

## 文件说明
- `results.json`：官方模板格式的评测结果。
- `results_summary.json` / `per_task.csv`：逐任务成功率和汇总。
- `episodes/<setting>__<task>.jsonl`：逐回合明细，可用于核对。
