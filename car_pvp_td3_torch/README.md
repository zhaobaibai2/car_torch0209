# car_pvp_td3_torch（PVP-TD3 实车节点）

本包用于将 **PVP-TD3（Torch 版本）** 迁移到实车节点上运行，接口、CAN 编码、安全策略、日志指标与 `car_dacer_torch` 对齐，便于对比实验。算法/网络/buffer 已内置在包内，不依赖外部 `pvp` 源码或 SB3。

## 1. 节点与话题

- 订阅：`surrounding_info_data`（`car_interfaces/msg/SurroundingInfoInterface`）
- 发布：`car_pvp_td3_data`（`car_interfaces/msg/CarRLInterface`）
- CAN：socketcan (`can1` / `can0`)，编码与 `car_dacer_torch` 一致

## 2. 目录结构

```
car_pvp_td3_torch/
├── car_pvp_td3_torch/
│   ├── car_pvp_td3_torch.py            # 实车主节点
│   ├── car_pvp_td3_torch_eval.py       # 评估节点（det/noisy）
│   ├── realcar_utils.py                # 观测构造/动作映射/CAN 打包
│   ├── segment_recorder.py             # 分段采集保存
│   ├── safety_manager.py               # 安全策略
│   └── config.yaml                     # 配置文件
├── package.xml
├── setup.py
├── setup.cfg
└── README.md
```

## 3. 构建与运行

```bash
colcon build --packages-select car_pvp_td3_torch
source install/setup.bash

# 实车主节点
ros2 run car_pvp_td3_torch car_pvp_td3_torch

# 评估节点（det/noisy）
ros2 run car_pvp_td3_torch car_pvp_td3_torch_eval_det
ros2 run car_pvp_td3_torch car_pvp_td3_torch_eval_noisy
```

## 4. 配置说明（config.yaml）

核心参数：

- `env.state_dim / env.action_dim`：与实车观测维度一致（默认 344/2）
- `algo.*`：PVP-TD3 超参（buffer_size、tau、gamma、policy_delay、target_noise 等）
- `training.*`：在线更新开关与频率
- `eval.model_dir / eval.model_path`：加载 `.pt/.pth` 模型
- `eval.deterministic`：是否确定性推理
- `eval.noise_std`：评估噪声强度（noisy 入口使用）
- `hardware.dry_run`：为 true 时不发送 CAN

## 5. 记录与指标（TensorBoard + CSV）

保存目录：`/home/nvidia/Autodrive/results/PVP_TD3_CAR_时间戳/`

**TensorBoard** 指标（训练/在线更新阶段）：
- `takeover_rate`
- `Timing/process_time_ms` / `Timing/infer_time_ms`
- `Vehicle/carspeed` / `Vehicle/error_yaw` / `Vehicle/error_distance` / `Vehicle/intervention`
- `Action/human_x/y` / `Action/model_x/y` / `Action/sent_throttle/brake/steer`
- `Safety/emergency_brakes` / `Safety/steer_violations` / `Safety/safety_override`
- `Comfort/action_smooth_l1`
- `Time/elapsed_s`

**评估节点 CSV**（eval_logs/*.csv）额外包含：
- `action_raw_*` / `action_sent_*`
- CAN bytes
- `action_smooth_l1`

## 6. 分段采集（按 P 键）

按 `P` 暂停/恢复：
- 暂停时自动保存分段数据到 `buffers/segments/segment_*.pkl`

## 7. 常见问题

1) **CAN 报错**
- 检查 `hardware.can_interface` 是否存在（`ip link show`）

2) **模型加载失败**
- 确认 `eval.model_path` 或 `eval.model_dir` 存在 `.pt/.pth` 模型文件
