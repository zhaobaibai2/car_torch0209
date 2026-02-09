import datetime
import os
import sys
import time
import json
import copy
import pickle
import yaml
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from car_interfaces.msg import *
import can
import torch
from pynput import keyboard

from .torch_networks import DACERTorchAgent, DACERActionConfig
from .torch_algorithm import PVPDACERTorch, TorchDACERConfig
from .torch_replay_buffer import Experience, TorchPVPBuffer, PhaseManager
from .safety_manager import SafetyManager


OBS_DIM = 344
PI = 3.1415926535

ros_node_name = "car_dacer_torch"

sys.path.append(os.getcwd() + "/src/utils/")
sys.path.append(os.getcwd() + "/src/%s/%s/"%(ros_node_name, ros_node_name))
sys.path.append(os.getcwd() + "/src/%s/%s/utils"%(ros_node_name, ros_node_name))

import tjitools


class CarDACERTorch(Node):
    def __init__(self):
        super().__init__(ros_node_name)

        self.config = self._load_config()

        self.subSurrounding = self.create_subscription(
            SurroundingInfoInterface, "surrounding_info_data", self.sub_callback_surrounding, 1
        )

        self.pubCarDACER = self.create_publisher(
            CarRLInterface, "car_dacer_data", 10
        )
        self.timerCarDACER = self.create_timer(0.1, self.pub_callback_car_dacer_torch)

        self.carControlBus = can.interface.Bus(
            channel=self.config['hardware']['can_interface'],
            bustype='socketcan'
        )

        self.iteration = 0
        self.rcvMsgSurroundingInfo = None
        self.takeover_recorder = deque(maxlen=2000)
        self._prev_intervention = 0.0

        training_cfg = self.config.get('training', {})
        self.use_one_step_delay = bool(training_cfg.get('use_one_step_delay', False))
        self.phase1_collect_only_intervention = bool(training_cfg.get('phase1_collect_only_intervention', False))
        self.phase3_human_mix_ratio = float(training_cfg.get('phase3_human_mix_ratio', 0.5))
        self.phase2_bc_updates_per_iter = int(training_cfg.get('phase2_bc_updates_per_iter', 1))
        self.log_detail_interval = int(training_cfg.get('log_detail_interval', 10))  # 详细日志打印间隔，防止刷屏

        self._prev_state = None
        self._prev_action_novice = None

        # 分段数据缓存：用于按 P 暂停时立即落盘当前段
        self.segment_human_buffer = []
        self.segment_pvp_buffer = []
        self.segment_idx = 0

        # 按键 P 暂停/恢复数据采集（参考 car_rl）
        self.is_paused = False
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        tjitools.ros_log(self.get_name(), "Keyboard listener started. Press 'P' to Pause/Resume.")

        self._init_logging()
        self._init_safety()
        self._init_algo_and_buffer()

        tjitools.ros_log(self.get_name(), f"Start Node: {self.get_name()}")

    def _load_config(self) -> dict:
        """Load configuration from YAML file（保持你原来的路径方式）"""
        config_path = os.getcwd() + "/src/%s/%s/"%(ros_node_name, ros_node_name) + "config.yaml"
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        base_save = "/home/nvidia/Autodrive"
        save_folder = os.path.join(
            base_save + "/results/",
            "DACER_TORCH_CAR_" + datetime.datetime.now().strftime("%y%m%d-%H%M%S"),
        )
        cfg["save_folder"] = save_folder
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(os.path.join(save_folder, "models"), exist_ok=True)
        os.makedirs(os.path.join(save_folder, "buffers"), exist_ok=True)

        with open(os.path.join(save_folder, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4)

        return cfg

    def _init_logging(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(
                log_dir=self.config["save_folder"],
                flush_secs=self.config.get('logging', {}).get('tensorboard_flush_secs', 20),
            )
            self.use_tensorboard = True
        except Exception:
            self.writer = None
            self.use_tensorboard = False

        self.training_metrics = {}

    def _init_safety(self):
        self.safety_manager = SafetyManager(self.config.get('safety', {}))

    def _init_algo_and_buffer(self):
        hw = self.config.get('hardware', {})
        use_gpu = bool(hw.get('use_gpu', True))
        cuda_mem_fraction = float(hw.get('cuda_mem_fraction', 0.7))

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            try:
                torch.cuda.set_per_process_memory_fraction(cuda_mem_fraction, device=self.device)
            except Exception:
                pass
        else:
            self.device = torch.device('cpu')

        net_cfg = self.config.get('network', {})
        alg_cfg = self.config.get('algorithm', {})

        action_cfg = DACERActionConfig(
            init_alpha=float(alg_cfg.get('init_alpha', 0.1)),
            action_noise_scale=float(alg_cfg.get('action_noise_scale', 0.05)),
        )

        self.agent = DACERTorchAgent(
            obs_dim=self.config['env']['state_dim'],
            act_dim=self.config['env']['action_dim'],
            hidden_dims=tuple(net_cfg.get('hidden_dims', [256, 256, 256])),
            diffusion_hidden_dims=tuple(net_cfg.get('diffusion_hidden_dims', [256, 256, 256])),
            num_timesteps=int(alg_cfg.get('num_timesteps', 20)),
            target_entropy=float(alg_cfg.get('target_entropy', -2.0)),
            time_dim=int(net_cfg.get('time_dim', 16)),
            activation=str(net_cfg.get('activation', 'relu')),
            use_layer_norm=bool(net_cfg.get('use_layer_norm', True)),
            action_cfg=action_cfg,
            device=self.device,
        ).to(self.device)

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

        self.algorithm = PVPDACERTorch(self.agent, cfg, device=self.device)

        buffer_max_size = int(self.config['training']['replay_batch_size'] * 100)
        self.buffer = TorchPVPBuffer(
            max_size=buffer_max_size,
            obs_dim=self.config['env']['state_dim'],
            act_dim=self.config['env']['action_dim'],
        )

        self.phase_manager = PhaseManager()
        self.phase_manager.phase1_threshold = int(self.config['training']['phase1_episodes'])
        self.phase_manager.phase2_threshold = int(self.config['training']['phase2_updates'])
        
        # 加载已有buffer数据（如果配置了路径）
        self._load_existing_buffer()

    def _load_existing_buffer(self):
        """加载已有buffer数据并直接进入Phase2"""
        demo_data_path = self.config.get('training', {}).get('demo_data_path')
        demo_data_dir = self.config.get('training', {}).get('demo_data_dir')
        
        loaded_any = False
        
        # 方案1：加载单个文件
        if demo_data_path and os.path.exists(demo_data_path):
            try:
                self.get_logger().info(f"[加载Buffer] 正在加载单个文件: {demo_data_path}")
                self.buffer.load(demo_data_path)
                loaded_any = True
            except Exception as e:
                self.get_logger().error(f"[加载Buffer] 单文件加载失败: {e}")
        
        # 方案2：批量加载目录下的所有buffer文件
        elif demo_data_dir and os.path.exists(demo_data_dir):
            try:
                self.get_logger().info(f"[加载Buffer] 正在批量加载目录: {demo_data_dir}")
                self._load_multiple_buffers(demo_data_dir)
                loaded_any = True
            except Exception as e:
                self.get_logger().error(f"[加载Buffer] 批量加载失败: {e}")
        
        # 方案3：自动搜索最近的buffer文件
        else:
            try:
                self.get_logger().info("[加载Buffer] 未指定路径，尝试自动搜索最近的buffer文件...")
                found_path = self._auto_find_latest_buffer()
                if found_path:
                    self.buffer.load(found_path)
                    loaded_any = True
                    self.get_logger().info(f"[加载Buffer] 自动找到并加载: {found_path}")
                else:
                    self.get_logger().info("[加载Buffer] 未找到已有的buffer文件")
            except Exception as e:
                self.get_logger().error(f"[加载Buffer] 自动搜索失败: {e}")
        
        # 如果成功加载了数据，设置为Phase2
        if loaded_any:
            self.phase_manager.current_phase = 2
            self.phase_manager.phase1_episodes = self.phase_manager.phase1_threshold  # 标记Phase1已完成
            
            buffer_stats = self.buffer.get_statistics()
            self.get_logger().info(
                f"[加载Buffer] 加载完成 - Human: {buffer_stats['human_size']} "
                f"PVP: {buffer_stats['pvp_size']} | 直接进入Phase2离线更新"
            )
            
            tjitools.ros_log(self.get_name(), "Loaded existing buffer(s) and entered Phase2")
        else:
            self.get_logger().info("[初始化] 未加载任何buffer数据，从Phase1开始")

    def _load_multiple_buffers(self, directory: str):
        """批量加载目录下的所有buffer文件"""
        import glob
        
        # 搜索所有buffer_*.pkl文件
        pattern = os.path.join(directory, "buffer_*.pkl")
        buffer_files = glob.glob(pattern)
        
        # 也搜索segment文件（如果有的话）
        segment_pattern = os.path.join(directory, "segment_*.pkl")
        segment_files = glob.glob(segment_pattern)
        
        all_files = buffer_files + segment_files
        
        if not all_files:
            self.get_logger().warning(f"[批量加载] 目录中未找到buffer文件: {directory}")
            return
        
        # 按文件名排序（通常包含时间戳）
        all_files.sort()
        
        self.get_logger().info(f"[批量加载] 找到 {len(all_files)} 个文件")
        
        loaded_count = 0
        total_human = 0
        total_pvp = 0
        
        for file_path in all_files:
            try:
                self.get_logger().info(f"[批量加载] 正在加载: {os.path.basename(file_path)}")
                
                # 创建临时buffer来加载单个文件
                temp_buffer = TorchPVPBuffer(
                    max_size=self.buffer.max_size,
                    obs_dim=self.buffer.obs_dim,
                    act_dim=self.buffer.act_dim,
                )
                temp_buffer.load(file_path)
                
                # 合并到主buffer
                for exp in list(temp_buffer.human_buffer):
                    self.buffer.add_human(exp)
                    total_human += 1
                
                for exp in list(temp_buffer.pvp_buffer):
                    self.buffer.add_pvp(exp)
                    total_pvp += 1
                
                loaded_count += 1
                self.get_logger().info(f"[批量加载] 完成: {os.path.basename(file_path)} "
                                     f"(Human: {len(temp_buffer.human_buffer)}, PVP: {len(temp_buffer.pvp_buffer)})")
                
            except Exception as e:
                self.get_logger().error(f"[批量加载] 文件加载失败 {file_path}: {e}")
                continue
        
        self.get_logger().info(f"[批量加载] 总结: 加载了 {loaded_count}/{len(all_files)} 个文件 "
                             f"(总Human: {total_human}, 总PVP: {total_pvp})")

    def _auto_find_latest_buffer(self) -> str:
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
                        return latest_file
        
        return None

    def on_press(self, key):
        """监听按键：P 键切换暂停状态。"""
        try:
            if key.char == 'p' or key.char == 'P':
                self.is_paused = not self.is_paused
                status = "PAUSED" if self.is_paused else "RESUMED"
                tjitools.ros_log(self.get_name(), f"System {status}")

                # 当切换到暂停时，立即把当前段数据保存下来
                if self.is_paused:
                    self._save_segment_data()
        except AttributeError:
            pass

    def _log_step_detail(self, intervention: float, action_behavior: np.ndarray, action_novice: np.ndarray):
        """打印采样输入/模型输出/人工动作，便于对齐数据。"""
        msg = self.rcvMsgSurroundingInfo
        throttle_h, brake_h, steer_h = (
            msg.throttle_percentage,
            msg.braking_percentage,
            msg.steerangle,
        )

        th_pred, br_pred, steer_pred = self.process_action(action_novice)
        state_preview = {
            "iter": self.iteration,
            "yaw_err": float(msg.error_yaw),
            "dist_err": float(msg.error_distance),
            "speed": float(msg.carspeed),
            "run_mode": int(msg.car_run_mode),
        }

        self.get_logger().info(
            "[采样明细] {iter} | run_mode:{run_mode} inter:{inter:.1f}\n"
            "  state(yaw:{yaw_err:.2f}, dist:{dist_err:.2f}, v:{speed:.2f})\n"
            "  Human(th:{th_h:.0f}, br:{br_h:.0f}, steer:{st_h:.1f}) | Model(th:{th_p:.0f}, br:{br_p:.0f}, steer:{st_p:.1f})".format(
                inter=intervention,
                th_h=throttle_h,
                br_h=brake_h,
                st_h=steer_h,
                th_p=th_pred,
                br_p=br_pred,
                st_p=steer_pred,
                **state_preview,
            )
        )

    def _save_segment_data(self):
        """将当前分段的采样数据落盘，便于按场景切片保存。"""
        if not self.segment_human_buffer and not self.segment_pvp_buffer:
            self.get_logger().info("[P段保存] 当前段没有数据可保存，跳过。")
            return

        buffers_dir = os.path.join(self.config['save_folder'], 'buffers')
        os.makedirs(buffers_dir, exist_ok=True)

        file_name = f"segment_{self.segment_idx:04d}_iter_{self.iteration:08d}.pkl"
        file_path = os.path.join(buffers_dir, file_name)

        payload = {
            'human': list(self.segment_human_buffer),
            'pvp': list(self.segment_pvp_buffer),
            'iteration': self.iteration,
            'phase': self.phase_manager.current_phase,
        }

        try:
            with open(file_path, 'wb') as f:
                pickle.dump(payload, f)
            self.get_logger().info(f"[P段保存] 已保存分段数据 -> {file_path}")
            tjitools.ros_log(self.get_name(), f"Segment saved: {file_name}")
            self.segment_idx += 1
        except Exception as e:
            self.get_logger().error(f"[P段保存] 保存失败: {e}")

        # 重置当前段缓存
        self.segment_human_buffer.clear()
        self.segment_pvp_buffer.clear()

    def sub_callback_surrounding(self, msgSurroundingInfo: SurroundingInfoInterface):
        self.rcvMsgSurroundingInfo = msgSurroundingInfo

    def get_state(self) -> np.ndarray:
        if self.rcvMsgSurroundingInfo is None:
            return np.zeros(OBS_DIM, dtype=np.float32)

        state = []

        errorYaw = float(self.rcvMsgSurroundingInfo.error_yaw) / PI
        errorYaw = np.clip(errorYaw, -1.0, 1.0)
        state.append(errorYaw)

        errorDistance = float(self.rcvMsgSurroundingInfo.error_distance) / 5.0
        errorDistance = np.clip(errorDistance, -1.0, 1.0)
        state.append(errorDistance)

        carspeed = float(self.rcvMsgSurroundingInfo.carspeed) / 15.0
        carspeed = np.clip(carspeed, 0.0, 1.0)
        state.append(carspeed)

        turn_state = float(self.rcvMsgSurroundingInfo.turn_signals)
        turn_state = (turn_state + 1.0) / 2.0
        state.append(turn_state)

        radar_data = list(self.rcvMsgSurroundingInfo.surroundinginfo)
        expected_radar_len = 240
        if len(radar_data) < expected_radar_len:
            radar_data.extend([1.0] * (expected_radar_len - len(radar_data)))
        elif len(radar_data) > expected_radar_len:
            radar_data = radar_data[:expected_radar_len]
        state.extend(radar_data)

        path = list(self.rcvMsgSurroundingInfo.path_rfu)
        expected_path_len = 100
        if len(path) < expected_path_len:
            path.extend([0.0] * (expected_path_len - len(path)))
        elif len(path) > expected_path_len:
            path = path[:expected_path_len]
        path = np.array(path, dtype=np.float32) / 30.0
        path = np.clip(path, -1.0, 1.0)
        state.extend(list(path))

        final_state = np.array(state, dtype=np.float32)
        if len(final_state) != OBS_DIM:
            fixed = np.zeros(OBS_DIM, dtype=np.float32)
            n = min(len(final_state), OBS_DIM)
            fixed[:n] = final_state[:n]
            final_state = fixed

        return final_state

    def process_action(self, action: np.ndarray) -> tuple:
        x = float(action[0])
        y = float(action[1])

        if x > 0:
            throttle_percentage = int(round(x * 100))
            braking_percentage = 0
        else:
            throttle_percentage = 0
            braking_percentage = int(round(abs(x) * 100))

        steering_angle = y * 200.0
        return throttle_percentage, braking_percentage, steering_angle

    def send_action(self, action: np.ndarray):
        hw = self.config.get('hardware', {})
        if bool(hw.get('dry_run', False)):
            return

        throttle_percentage, braking_percentage, steering_angle = self.process_action(action)

        state = self.get_state()
        if not self.safety_manager.check_action_safety(action, state):
            throttle_percentage = 0
            braking_percentage = 100
            steering_angle = 0.0

        gearpos = 0x03
        enableSignal = 1
        ultrasonicSwitch = 0
        dippedHeadlight = 0
        contourLamp = 0

        brakeEnable = 1 if braking_percentage != 0 else 0
        alarmLamp = 0
        turnSignalControl = 0
        appInsert = 0

        Byte7 = appInsert << 7 | turnSignalControl << 1 | alarmLamp
        Byte6 = int(braking_percentage) << 1 | brakeEnable
        Byte5 = (int(steering_angle) & 0xFF00) >> 8
        Byte4 = int(steering_angle) & 0x00FF
        Byte3 = 0
        Byte2 = ((int(throttle_percentage) * 10) & 0xFF00) >> 8
        Byte1 = (int(throttle_percentage) * 10) & 0x00FF
        Byte0 = gearpos << 6 | enableSignal << 5 | ultrasonicSwitch << 4 | dippedHeadlight << 1 | contourLamp

        canData = [Byte0, Byte1, Byte2, Byte3, Byte4, Byte5, Byte6, Byte7]
        carControlMsg = can.Message(
            arbitration_id=self.config['hardware']['can_id'],
            data=canData,
            extended_id=False,
        )
        self.carControlBus.send(carControlMsg)

    def reverse_process_action(self, throttle_percentage: float, braking_percentage: float, steering_angle: float) -> np.ndarray:
        if braking_percentage != 0:
            x = -float(braking_percentage / 100)
        else:
            x = float(throttle_percentage / 100)
        x = np.clip(x, -1.0, 1.0)

        y = float(steering_angle) / 200.0
        y = np.clip(y, -1.0, 1.0)

        return np.array([x, y], dtype=np.float32)

    @torch.no_grad()
    def _compute_action(self, state: np.ndarray) -> np.ndarray:
        obs = torch.from_numpy(np.expand_dims(state, axis=0).astype(np.float32)).to(self.device)
        action = self.algorithm.get_action(obs, deterministic=False)
        action = action.detach().cpu().numpy().flatten().astype(np.float32)
        return np.clip(action, -1.0, 1.0)

    def _sample_one_step_delay(self):
        state = self.get_state()
        batch_data = []

        intervention = 0.0 if int(self.rcvMsgSurroundingInfo.car_run_mode) == 1 else 1.0
        self.takeover_recorder.append(intervention)

        action_behavior = self.reverse_process_action(
            self.rcvMsgSurroundingInfo.throttle_percentage,
            self.rcvMsgSurroundingInfo.braking_percentage,
            self.rcvMsgSurroundingInfo.steerangle,
        )

        takeover_start = (intervention == 1.0 and self._prev_intervention == 0.0)
        takeover_end = (intervention == 0.0 and self._prev_intervention == 1.0)
        stop_td = 1.0 if (takeover_start or takeover_end) else 0.0

        if self._prev_state is not None and self._prev_action_novice is not None:
            exp = Experience.from_pvp(
                obs=self._prev_state,
                next_obs=state,
                reward=0.0,
                done=False,
                a_behavior=action_behavior,
                a_novice=self._prev_action_novice,
                a_human=action_behavior,
                intervention=intervention,
                stop_td=stop_td,
            )
            batch_data.append(exp)

        action_novice = self._compute_action(state)
        self.send_action(action_novice)

        self._prev_state = state.copy()
        self._prev_action_novice = action_novice.copy()
        self._prev_intervention = intervention

        return batch_data

    def train_algorithm(self) -> dict:
        stats = self.buffer.get_statistics()

        if self.phase_manager.current_phase == 1:
            return {}

        if self.phase_manager.current_phase == 2:
            bs = int(self.config['training']['replay_batch_size'])
            human_exps = self.buffer.sample_human(bs)
            if not human_exps:
                return {}

            obs = torch.as_tensor(np.stack([e.obs for e in human_exps], axis=0), device=self.device, dtype=torch.float32)
            act = torch.as_tensor(np.stack([e.actions_behavior for e in human_exps], axis=0), device=self.device, dtype=torch.float32)

            bc_updates = max(1, self.phase2_bc_updates_per_iter)
            metrics = {}
            for _ in range(bc_updates):
                m = self.algorithm.train_offline_bc(obs, act)
                # 覆盖为最新一次，数值相近无需平均，减少额外开销
                metrics.update(m)
                self.phase_manager.update_progress(updates=1)
# [新增] 打印 Phase 2 的 BC Loss
            self.get_logger().info(f"Phase2 Training | BC Loss: {metrics.get('bc_loss', 0.0):.4f} | Progress: {self.phase_manager.phase2_updates}/{self.phase_manager.phase2_threshold}")


            if self.phase_manager.should_transition_to_pvp():
                old = self.phase_manager.current_phase
                self.phase_manager.transition_to_pvp()
                info = self.phase_manager.get_phase_info()
                self.get_logger().info(f"=== PHASE TRANSITION: {old} -> {info['current_phase']} ({info['phase_name']}) ===")

            return metrics

        if self.phase_manager.current_phase == 3:
            bs = int(self.config['training']['replay_batch_size'])
            exps = self.buffer.sample_pvp_mixed(bs, human_ratio=self.phase3_human_mix_ratio)
            batch = self.buffer.to_pvp_batch(exps, device=self.device)
            if batch is None:
                return {}

            metrics = self.algorithm.train_pvp(batch)
            self.phase_manager.update_progress(iterations=1)
# [新增] 打印 Phase 3 的关键 Loss
            self.get_logger().info(f"Phase3 PVP | Q1 Loss: {metrics.get('train/q1_loss', 0):.4f} | Policy Loss: {metrics.get('train/policy_loss', 0):.4f}")

            return metrics

        return {}

    def _save_demo_data(self) -> str:
        path = os.path.join(self.config['save_folder'], 'buffers', f"demo_data_{self.iteration:08d}.pkl")
        data = list(self.buffer.human_buffer)
        with open(path, 'wb') as f:
            import pickle
            pickle.dump(data, f)
        return path

    def _save_model(self):
        model_path = os.path.join(self.config['save_folder'], 'models', f"dacer_torch_{self.iteration:08d}.pt")
        try:
            self.algorithm.save(model_path)
        except Exception as e:
            self.get_logger().error(f"Failed to save model: {e}")

        buffer_path = os.path.join(self.config['save_folder'], 'buffers', f"buffer_{self.iteration:08d}.pkl")
        try:
            self.buffer.save(buffer_path)
        except Exception as e:
            self.get_logger().error(f"Failed to save buffer: {e}")

    def _log_metrics(self):
        buffer_stats = self.buffer.get_statistics()
        phase_info = self.phase_manager.get_phase_info()
        safety_stats = self.safety_manager.get_safety_stats()

        if len(self.takeover_recorder) > 0:
            takeover_rate = float(np.mean(np.array(self.takeover_recorder)) * 100.0)
        else:
            takeover_rate = 0.0

# [新增] 打印统计摘要
        self.get_logger().info(
            f"\n=== Stats @ Iter {self.iteration} ===\n"
            f"  Buffer (Human/PVP): {buffer_stats['human_size']}/{buffer_stats['pvp_size']}\n"
            f"  Safety (Brakes/Steer): {safety_stats['emergency_brakes']}/{safety_stats['steer_violations']}\n"
            f"  Takeover Rate: {takeover_rate:.2f}%\n"
            f"=============================="
        )

        if self.use_tensorboard:
            for k, v in self.training_metrics.items():
                try:
                    self.writer.add_scalar(k, float(v), self.iteration)
                except Exception:
                    pass

            self.writer.add_scalar('Buffer/human_size', buffer_stats['human_size'], self.iteration)
            self.writer.add_scalar('Buffer/pvp_size', buffer_stats['pvp_size'], self.iteration)
            self.writer.add_scalar('Phase/current', phase_info['current_phase'], self.iteration)
            self.writer.add_scalar('Safety/emergency_brakes', safety_stats['emergency_brakes'], self.iteration)
            self.writer.add_scalar('Safety/steer_violations', safety_stats['steer_violations'], self.iteration)
            self.writer.add_scalar('takeover_rate', takeover_rate, self.iteration)

    def pub_callback_car_dacer_torch(self):
        msg = CarRLInterface()
        now_ts = time.time()
        msg.timestamp = now_ts

        if self.rcvMsgSurroundingInfo is None:
            return

        # 暂停时不采样/不训练/不发送动作，便于阶段一按场景采集
        if self.is_paused:
            return

        self.iteration += 1


# [新增] 打印当前 Iter 和 Phase，方便确认程序在跑
        if self.iteration % 10 == 0:  # 防止刷屏太快，每10次打印一次
            phase_name = self.phase_manager.get_phase_info()['phase_name']
            self.get_logger().info(f"Iter: {self.iteration} | Phase: {self.phase_manager.current_phase} ({phase_name})")



        if int(self.rcvMsgSurroundingInfo.gearpos) != 2:
            # Phase2 仅离线 BC 训练，不发送 AI 动作，保持人工控制以保障安全与实时性
            if self.phase_manager.current_phase != 2:
                exps = self._sample_one_step_delay() if self.use_one_step_delay else self._sample_one_step_delay()

                for exp in exps:
                    if self.phase_manager.current_phase == 1:
                        if self.phase1_collect_only_intervention and exp.interventions <= 0.5:
                            continue
                        self.buffer.add_human(exp)
                        self.buffer.add_pvp(exp)
                        # 记录分段数据，便于按 P 保存
                        self.segment_human_buffer.append(exp)
                        self.segment_pvp_buffer.append(exp)
                        self.phase_manager.update_progress(episodes=1)

                        if self.phase_manager.should_transition_to_phase2():
                            self._save_demo_data()
                            old = self.phase_manager.current_phase
                            self.phase_manager.transition_to_phase2()
                            info = self.phase_manager.get_phase_info()
                            self.get_logger().info(f"=== PHASE TRANSITION: {old} -> {info['current_phase']} ({info['phase_name']}) ===")
                    else:
                        if exp.interventions > 0.5:
                            self.buffer.add_human(exp)
                            self.segment_human_buffer.append(exp)
                        self.buffer.add_pvp(exp)
                        self.segment_pvp_buffer.append(exp)

                # 打印采样明细（节流），与 car_rl 一致地看到模型输出 vs 实际输出
                if self.iteration % self.log_detail_interval == 0 and exps:
                    # 使用第一个 exp 的行为动作作为参考
                    ref_exp = exps[0]
                    self._log_step_detail(
                        intervention=ref_exp.interventions,
                        action_behavior=ref_exp.actions_behavior,
                        action_novice=ref_exp.actions_novice,
                    )

        stats = self.buffer.get_statistics()

        can_train = False
        if self.phase_manager.current_phase == 2:
            can_train = stats['human_size'] >= int(self.config['training']['buffer_warm_size'])
        elif self.phase_manager.current_phase == 3:
            can_train = stats['pvp_size'] >= int(self.config['training']['replay_batch_size'])

        if self.iteration % int(self.config['training']['update_interval']) == 0 and can_train:
            metrics = self.train_algorithm()
            if metrics:
                self.training_metrics.update(metrics)

        if self.iteration % int(self.config['training']['log_save_interval']) == 0:
            self._log_metrics()

        if self.iteration % int(self.config['training']['apprfunc_save_interval']) == 0:
            self._save_model()

        msg.process_time = time.time() - now_ts
        self.pubCarDACER.publish(msg)

        tjitools.ros_log(self.get_name(), 'Publish car_dacer_torch msg !!!')


def main():
    rclpy.init()
    rosNode = CarDACERTorch()
    rclpy.spin(rosNode)
    rclpy.shutdown()
