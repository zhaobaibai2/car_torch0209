#!/usr/bin/env python3
"""
独立的buffer加载脚本
用于加载已有的buffer数据并直接进入Phase2训练
"""

import os
import sys
import yaml
import pickle
import torch
import numpy as np
from collections import deque

# 添加路径以导入你的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from torch_replay_buffer import TorchPVPBuffer, Experience, PhaseManager
from torch_networks import DACERTorchAgent, DACERActionConfig
from torch_algorithm import PVPDACERTorch, TorchDACERConfig


def load_config(config_path="config.yaml"):
    """加载配置文件"""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_buffer_from_paths(buffer_paths, config):
    """从多个路径加载并合并buffer数据"""
    print(f"[加载] 正在加载 {len(buffer_paths)} 个buffer文件...")
    
    # 创建buffer
    buffer_max_size = int(config['training']['replay_batch_size'] * 100)
    buffer = TorchPVPBuffer(
        max_size=buffer_max_size,
        obs_dim=config['env']['state_dim'],
        act_dim=config['env']['action_dim'],
    )
    
    total_human = 0
    total_pvp = 0
    loaded_count = 0
    
    for buffer_path in buffer_paths:
        try:
            print(f"[加载] 正在加载: {os.path.basename(buffer_path)}")
            
            # 创建临时buffer来加载单个文件
            temp_buffer = TorchPVPBuffer(
                max_size=buffer_max_size,
                obs_dim=config['env']['state_dim'],
                act_dim=config['env']['action_dim'],
            )
            temp_buffer.load(buffer_path)
            
            # 合并到主buffer
            for exp in list(temp_buffer.human_buffer):
                buffer.add_human(exp)
                total_human += 1
            
            for exp in list(temp_buffer.pvp_buffer):
                buffer.add_pvp(exp)
                total_pvp += 1
            
            loaded_count += 1
            print(f"[加载] 完成: {os.path.basename(buffer_path)} "
                  f"(Human: {len(temp_buffer.human_buffer)}, PVP: {len(temp_buffer.pvp_buffer)})")
            
        except Exception as e:
            print(f"[错误] 文件加载失败 {buffer_path}: {e}")
            continue
    
    stats = buffer.get_statistics()
    print(f"[加载] 总结: 加载了 {loaded_count}/{len(buffer_paths)} 个文件 "
          f"(总Human: {total_human}, 总PVP: {total_pvp})")
    print(f"[加载] Buffer状态 - Human: {stats['human_size']}, PVP: {stats['pvp_size']}")
    
    return buffer


def find_buffer_files_in_directory(directory):
    """在目录中查找所有buffer文件"""
    import glob
    
    buffer_files = glob.glob(os.path.join(directory, "buffer_*.pkl"))
    segment_files = glob.glob(os.path.join(directory, "segment_*.pkl"))
    
    all_files = buffer_files + segment_files
    all_files.sort()  # 按文件名排序
    
    return all_files


def auto_find_latest_buffer():
    """自动搜索最新的buffer文件"""
    import glob
    
    # 搜索常见的buffer目录
    search_paths = [
        "/home/nvidia/Autodrive/results/DACER_TORCH_CAR_*/buffers/",
        "/home/nvidia/Autodrive/results/*/buffers/",
        "./buffers/",
        "../buffers/",
    ]
    
    for search_path in search_paths:
        expanded_paths = glob.glob(search_path)
        for path in expanded_paths:
            if os.path.exists(path):
                buffer_files = glob.glob(os.path.join(path, "buffer_*.pkl"))
                if buffer_files:
                    # 按修改时间排序，选择最新的
                    latest_file = max(buffer_files, key=os.path.getmtime)
                    return [latest_file]
    
    return []


def setup_algorithm_and_agent(config):
    """设置算法和智能体"""
    # 设备配置
    hw = config.get('hardware', {})
    use_gpu = bool(hw.get('use_gpu', True))
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    
    # 网络配置
    net_cfg = config.get('network', {})
    alg_cfg = config.get('algorithm', {})
    
    action_cfg = DACERActionConfig(
        init_alpha=float(alg_cfg.get('init_alpha', 0.1)),
        action_noise_scale=float(alg_cfg.get('action_noise_scale', 0.05)),
    )
    
    agent = DACERTorchAgent(
        obs_dim=config['env']['state_dim'],
        act_dim=config['env']['action_dim'],
        hidden_dims=tuple(net_cfg.get('hidden_dims', [256, 256, 256])),
        diffusion_hidden_dims=tuple(net_cfg.get('diffusion_hidden_dims', [256, 256, 256])),
        num_timesteps=int(alg_cfg.get('num_timesteps', 20)),
        target_entropy=float(alg_cfg.get('target_entropy', -2.0)),
        time_dim=int(net_cfg.get('time_dim', 16)),
        activation=str(net_cfg.get('activation', 'relu')),
        use_layer_norm=bool(net_cfg.get('use_layer_norm', True)),
        action_cfg=action_cfg,
        device=device,
    ).to(device)
    
    cfg = TorchDACERConfig(
        gamma=float(alg_cfg.get('gamma', 0.99)),
        tau=float(alg_cfg.get('tau', 0.005)),
        lr=float(alg_cfg.get('lr', 1e-4)),
        alpha_lr=float(alg_cfg.get('alpha_lr', 3e-2)),
        delay_update=int(alg_cfg.get('delay_update', 1)),
        delay_alpha_update=int(alg_cfg.get('delay_alpha_update', 1000)),
        reward_scale=float(alg_cfg.get('reward_scale', 1.0)),
        lambda_pv=float(alg_cfg.get('lambda_pv', 1.0)),
        B=float(alg_cfg.get('B', 1.0)),
        lambda_bc=float(alg_cfg.get('lambda_bc', 5.0)),
        reward_free=bool(alg_cfg.get('reward_free', True)),
        phase3_use_bc_boost=bool(alg_cfg.get('phase3_use_bc_boost', True)),
        fix_alpha=bool(alg_cfg.get('fix_alpha', True)),
        target_entropy=float(alg_cfg.get('target_entropy', -2.0)),
    )
    
    algorithm = PVPDACERTorch(agent, cfg, device=device)
    
    return algorithm, device


def run_phase2_training(config, buffer_input=None, max_updates=None):
    """运行Phase2离线BC训练"""
    print("=== 开始Phase2离线BC训练 ===")
    
    # 加载配置和数据
    config = load_config()
    
    # 确定buffer加载方式
    if buffer_input is None:
        # 从配置文件加载
        demo_data_path = config.get('training', {}).get('demo_data_path')
        demo_data_dir = config.get('training', {}).get('demo_data_dir')
        
        if demo_data_path:
            buffer_paths = [demo_data_path]
        elif demo_data_dir:
            buffer_paths = find_buffer_files_in_directory(demo_data_dir)
        else:
            buffer_paths = auto_find_latest_buffer()
        
        if not buffer_paths:
            print("[错误] 未找到任何buffer文件")
            return
    elif isinstance(buffer_input, str):
        # 单个文件路径
        buffer_paths = [buffer_input]
    elif isinstance(buffer_input, list):
        # 文件路径列表
        buffer_paths = buffer_input
    else:
        print("[错误] 无效的buffer输入参数")
        return
    
    # 加载buffer数据
    buffer = load_buffer_from_paths(buffer_paths, config)
    
    # 检查是否有足够数据
    if buffer.human_size == 0:
        print("[错误] Buffer中没有human数据，无法进行BC训练")
        return
    
    algorithm, device = setup_algorithm_and_agent(config)
    
    # 设置Phase管理器
    phase_manager = PhaseManager()
    phase_manager.current_phase = 2
    phase_manager.phase1_episodes = phase_manager.phase1_threshold  # 标记Phase1已完成
    
    # 训练参数
    batch_size = int(config['training']['replay_batch_size'])
    bc_updates_per_iter = int(config['training']['phase2_bc_updates_per_iter'])
    phase2_threshold = int(config['training']['phase2_updates'])
    
    if max_updates:
        phase2_threshold = max_updates
    
    print(f"[训练] Batch Size: {batch_size}, BC Updates/Iter: {bc_updates_per_iter}")
    print(f"[训练] 目标总更新次数: {phase2_threshold}")
    
    # 开始训练循环
    iteration = 0
    while phase_manager.phase2_updates < phase2_threshold:
        # 检查buffer是否有足够数据
        if buffer.human_size < int(config['training']['buffer_warm_size']):
            print(f"[等待] Buffer数据不足: {buffer.human_size} < {config['training']['buffer_warm_size']}")
            break
        
        # 采样human数据进行BC训练
        human_exps = buffer.sample_human(batch_size)
        if not human_exps:
            continue
        
        # 准备训练数据
        obs = torch.as_tensor(np.stack([e.obs for e in human_exps], axis=0), device=device, dtype=torch.float32)
        act = torch.as_tensor(np.stack([e.actions_behavior for e in human_exps], axis=0), device=device, dtype=torch.float32)
        
        # 执行BC更新
        for _ in range(bc_updates_per_iter):
            metrics = algorithm.train_offline_bc(obs, act)
            phase_manager.update_progress(updates=1)
        
        iteration += 1
        
        # 打印进度
        if iteration % 10 == 0:
            progress = phase_manager.phase2_updates / phase2_threshold * 100
            bc_loss = metrics.get('bc_loss', 0.0)
            print(f"[进度] Iter: {iteration} | Phase2 Updates: {phase_manager.phase2_updates}/{phase2_threshold} ({progress:.1f}%) | BC Loss: {bc_loss:.4f}")
        
        # 保存检查点
        if iteration % 1000 == 0:
            save_path = f"phase2_checkpoint_iter_{iteration}.pt"
            algorithm.save(save_path)
            print(f"[保存] 检查点已保存: {save_path}")
    
    print(f"[完成] Phase2训练结束! 总更新次数: {phase_manager.phase2_updates}")
    
    # 保存最终模型
    final_model_path = "phase2_final_model.pt"
    algorithm.save(final_model_path)
    print(f"[保存] 最终模型已保存: {final_model_path}")


if __name__ == "__main__":
    # 使用示例
    if len(sys.argv) < 2:
        print("用法:")
        print("  python load_buffer_and_train.py <buffer_path> [max_updates]")
        print("  python load_buffer_and_train.py <buffer_dir> [max_updates]")
        print("  python load_buffer_and_train.py auto [max_updates]")
        print("")
        print("示例:")
        print("  # 加载单个buffer文件")
        print("  python load_buffer_and_train.py /path/to/buffer.pkl")
        print("  # 批量加载目录下所有buffer文件")
        print("  python load_buffer_and_train.py /path/to/buffers/")
        print("  # 自动搜索最新的buffer文件")
        print("  python load_buffer_and_train.py auto")
        print("  # 限制最大更新次数")
        print("  python load_buffer_and_train.py /path/to/buffer.pkl 10000")
        sys.exit(1)
    
    buffer_input = sys.argv[1]
    max_updates = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if buffer_input == "auto":
        buffer_input = None  # 让脚本自动搜索
    elif os.path.isdir(buffer_input):
        # 是目录，批量加载
        print(f"[批量加载] 指定目录: {buffer_input}")
    elif os.path.isfile(buffer_input):
        # 是文件，单个加载
        print(f"[单文件加载] 指定文件: {buffer_input}")
    else:
        print(f"[错误] 路径不存在: {buffer_input}")
        sys.exit(1)
    
    # 加载配置
    config = load_config()
    
    # 开始训练
    run_phase2_training(config, buffer_input, max_updates)
