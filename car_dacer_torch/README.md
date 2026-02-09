# car_dacer_torch（中文说明）

这是一个基于 **PyTorch**（不依赖 JAX）实现的 **PVP-DACER** 实车 ROS2 节点，接口与现有车端代码一致：

- 订阅：`surrounding_info_data`（`car_interfaces/msg/SurroundingInfoInterface`）
- 发布：`car_dacer_data`（`car_interfaces/msg/CarRLInterface`，与 car_rl/car_dacer 同名）
- CAN 编码：与 car_rl/car_bc 相同

## 1. 解决什么问题
JAX 版 `car_dacer` 在 Orin 上可能因 XLA/JAX 运行时或 GPU 内存分配导致 segfault。本包用 **纯 PyTorch**，无 JIT/XLA，并尝试限制 GPU 占用，降低崩溃风险。

## 2. 目录结构
```
car_dacer_torch/
  package.xml
  setup.py
  setup.cfg
  resource/
  car_dacer_torch/
    __init__.py
    config.yaml
    car_dacer_torch.py
    safety_manager.py
    torch_algorithm.py
    torch_diffusion.py
    torch_networks.py
    torch_replay_buffer.py
```

## 3. 主要文件与作用
### `car_dacer_torch/car_dacer_torch.py`
- ROS2 节点：订阅/发布/定时器，初始化 CAN、缓冲区、阶段管理、安全管理、Torch 网络与算法。
- `get_state()`：把 `SurroundingInfoInterface` 组装成 344 维观测，归一化方式与 car_rl/car_dacer 一致。
- `send_action()`：归一化动作映射油门/制动/转向，安全未通过则紧急制动；`hardware.dry_run=true` 时不发 CAN。
- `_sample_one_step_delay()`：实现 one-step delay，对齐 PVP 经验（上一时刻 AI 动作 + 当前行为/人类动作 + stop_td 接管边界）。
- `train_algorithm()`：Phase2 扩散 BC 离线训练；Phase3 在线 PVP（可混采 human+pvp）。

### `car_dacer_torch/torch_algorithm.py`
- `PVPDACERTorch.train_pvp`：TD 用行为动作 `a_b`；TD mask 包含 stop_td，若 `reward_free=true` 还会屏蔽干预段；PV 用人类/AI 动作；策略损失 = 非干预 RL + 干预段 diffusion BC boost；`fix_alpha=true` 时不更新 alpha。
- `train_offline_bc`：Phase2 扩散 BC 预训练。

### `car_dacer_torch/torch_networks.py`
- `DACERTorchAgent`：扩散策略、双分布式 Q、目标 Q、`log_alpha`；`get_action` 训练可反传，推理由上层 no_grad 包裹。

### `car_dacer_torch/torch_diffusion.py`
- 扩散采样/去噪/加权 diffusion loss（干预段 BC boost）。`p_sample` 可反传，`p_sample_deterministic` 用于无噪声推理。

### `car_dacer_torch/torch_replay_buffer.py`
- `TorchPVPBuffer`：`human_buffer`(Phase2)、`pvp_buffer`(Phase3)，支持混采；`PhaseManager` 管理阶段与阈值。

### `car_dacer_torch/safety_manager.py`
- 转向角越界、距离/车速紧急制动检查与统计。

## 4. 配置（`config.yaml`）关键项
- `hardware.use_gpu` / `cuda_mem_fraction`：启用 CUDA 与显存上限（best-effort）。
- `hardware.dry_run`：调试关闭 CAN。
- `training.use_one_step_delay`：开启 one-step delay。
- `training.phase1_collect_only_intervention`：Phase1 只收接管样本。
- `training.phase3_human_mix_ratio`：Phase3 混合 demo 比例。
- `algorithm.phase3_use_bc_boost`：干预段 BC boost。
- `algorithm.fix_alpha`：固定 alpha。

## 5. 构建与运行
```bash
colcon build --packages-select car_dacer_torch
source install/setup.bash
ros2 run car_dacer_torch car_dacer_torch
```

## 6. 实车部署建议
1) 先设 `hardware.dry_run: true`，确认订阅/发布正常；
2) 检查观测维度是否始终 344；
3) 确认安全阈值后再将 `dry_run` 设为 false 发 CAN。

## 7. 提示
- 不依赖 JAX。
- 算法语义对齐原 JAX 版 `pvp_dacer`：行为动作 TD、干预 PV、干预 BC boost。

