import datetime
import json
import os
import time
from collections import deque
from pathlib import Path

import can
import numpy as np
import rclpy
import torch
import yaml
from car_interfaces.msg import CarRLInterface, SurroundingInfoInterface
from pynput import keyboard
from rclpy.node import Node

from .realcar_utils import OBS_DIM, build_state, pack_can, process_action, reverse_process_action, safe_get
from .safety_manager import SafetyManager
from .segment_recorder import SegmentRecorder
from .torch_algorithm import TorchPVPTD3, TorchPVPTD3Config
from .torch_replay_buffer import Experience, TorchPVPBuffer

ros_node_name = "car_pvp_td3_torch"

try:
    import tjitools
except Exception:
    tjitools = None

class CarPVPTD3Torch(Node):
    def __init__(self):
        super().__init__(ros_node_name)

        self.config = self._load_config()

        self.subSurrounding = self.create_subscription(
            SurroundingInfoInterface, "surrounding_info_data", self.sub_callback_surrounding, 1
        )

        self.pubCar = self.create_publisher(CarRLInterface, "car_pvp_td3_data", 10)
        self.timerCar = self.create_timer(0.1, self.pub_callback_car_pvp_td3_torch)

        self.carControlBus = can.interface.Bus(
            channel=self.config['hardware']['can_interface'],
            bustype='socketcan',
        )

        self.iteration = 0
        self.rcvMsgSurroundingInfo = None
        self.takeover_recorder = deque(maxlen=2000)

        self.start_wall_time = time.time()
        self.last_timestamp = 0.0
        self.last_process_time = 0.0
        self.last_infer_time = 0.0

        self.last_intervention = 0.0
        self.last_state_error_yaw = 0.0
        self.last_state_error_distance = 0.0
        self.last_state_carspeed = 0.0

        self.last_action_human = np.zeros(2, dtype=np.float32)
        self.last_action_model = np.zeros(2, dtype=np.float32)
        self.last_action_sent = np.zeros(2, dtype=np.float32)
        self.last_sent_throttle = 0
        self.last_sent_brake = 0
        self.last_sent_steer = 0.0
        self.last_safety_override = 0

        self.prev_action_sent_for_smooth = None
        self.last_action_smooth_l1 = 0.0

        self.segment_recorder = SegmentRecorder(os.path.join(self.config['save_folder'], 'buffers', 'segments'))

        self.is_paused = False
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

        self._prev_state = None
        self._prev_action_novice = None
        self._prev_intervention = 0.0

        self._init_logging()
        self._init_safety()
        self._init_model()

        self.training_metrics = {}

        if tjitools is not None:
            tjitools.ros_log(self.get_name(), f"Start Node: {self.get_name()}")

    def _load_config(self) -> dict:
        config_path = self._resolve_config_path()
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        base_save = "/home/nvidia/Autodrive"
        save_folder = os.path.join(
            base_save,
            "results",
            "PVP_TD3_CAR_" + datetime.datetime.now().strftime("%y%m%d-%H%M%S"),
        )
        cfg["save_folder"] = save_folder
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(os.path.join(save_folder, "models"), exist_ok=True)
        os.makedirs(os.path.join(save_folder, "buffers"), exist_ok=True)

        with open(os.path.join(save_folder, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4)

        return cfg

    def _resolve_config_path(self) -> str:
        candidates = []
        try:
            from ament_index_python.packages import get_package_share_directory

            share_dir = Path(get_package_share_directory(ros_node_name))
            candidates.append(share_dir / "config" / "config.yaml")
        except Exception:
            pass

        candidates.extend(
            [
                Path(__file__).resolve().parent / "config.yaml",
                Path(os.getcwd()) / "src" / ros_node_name / ros_node_name / "config.yaml",
                Path(os.getcwd()) / ros_node_name / "config.yaml",
            ]
        )

        for cand in candidates:
            if cand.is_file():
                return str(cand)

        raise FileNotFoundError(f"config.yaml not found, tried: {[str(c) for c in candidates]}")

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

    def _init_safety(self):
        self.safety_manager = SafetyManager(self.config.get('safety', {}))

    def _init_model(self):
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

        eval_cfg = self.config.get('eval', {}) or {}
        self.deterministic = bool(eval_cfg.get('deterministic', True))

        env_cfg = self.config.get('env', {})
        self.obs_dim = int(env_cfg.get('state_dim', OBS_DIM))
        self.act_dim = int(env_cfg.get('action_dim', 2))

        algo_cfg = self.config.get('algo', {}) or {}
        net_cfg = self.config.get('network', {}) or {}

        cfg = TorchPVPTD3Config(
            gamma=float(algo_cfg.get('gamma', 0.99)),
            tau=float(algo_cfg.get('tau', 0.005)),
            lr=float(algo_cfg.get('learning_rate', 1e-4)),
            policy_delay=int(algo_cfg.get('policy_delay', 2)),
            target_policy_noise=float(algo_cfg.get('target_policy_noise', 0.2)),
            target_noise_clip=float(algo_cfg.get('target_noise_clip', 0.5)),
            q_value_bound=float(algo_cfg.get('q_value_bound', 1.0)),
            cql_coefficient=float(algo_cfg.get('cql_coefficient', 1.0)),
            reward_free=bool(algo_cfg.get('reward_free', True)),
            intervention_start_stop_td=bool(algo_cfg.get('intervention_start_stop_td', True)),
            use_balance_sample=bool(algo_cfg.get('use_balance_sample', True)),
            human_ratio=float(algo_cfg.get('human_ratio', 0.5)),
        )

        actor_hidden = list(net_cfg.get('actor_hidden_dims', [256, 256]))
        critic_hidden = list(net_cfg.get('critic_hidden_dims', [256, 256]))

        self.model = TorchPVPTD3(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            actor_hidden_dims=actor_hidden,
            critic_hidden_dims=critic_hidden,
            cfg=cfg,
            device=self.device,
        )

        self.buffer = TorchPVPBuffer(max_size=int(algo_cfg.get('buffer_size', 50000)), obs_dim=self.obs_dim, act_dim=self.act_dim)

        model_path = self._resolve_model_path()
        if model_path:
            self.model.load(model_path)
            self.get_logger().info(f"Loaded model: {model_path}")
        else:
            self.get_logger().info("Initialized new Torch PVP-TD3 model (no checkpoint found)")

    def _resolve_model_path(self):
        eval_cfg = self.config.get('eval', {}) or {}
        model_path = eval_cfg.get('model_path')
        if model_path and os.path.exists(model_path):
            return str(model_path)

        model_dir = eval_cfg.get('model_dir')
        if model_dir and os.path.isdir(model_dir):
            candidates = []
            for name in os.listdir(model_dir):
                if not (name.endswith('.pt') or name.endswith('.pth')):
                    continue
                full = os.path.join(model_dir, name)
                if os.path.isfile(full):
                    candidates.append(full)
            if candidates:
                return max(candidates, key=os.path.getmtime)

        return None

    def on_press(self, key):
        try:
            if hasattr(key, 'char') and (key.char == 'p' or key.char == 'P'):
                self.is_paused = not self.is_paused
                if self.is_paused:
                    self._save_segment()
                    self.get_logger().info("Paused (segment saved)")
                else:
                    self.get_logger().info("Resumed")
        except Exception:
            pass

    def _save_segment(self):
        try:
            self.segment_recorder.save()
        except Exception as e:
            self.get_logger().error(f"Segment save failed: {e}")
        finally:
            pass

    def sub_callback_surrounding(self, msgSurroundingInfo: SurroundingInfoInterface):
        self.rcvMsgSurroundingInfo = msgSurroundingInfo

    def get_state(self) -> np.ndarray:
        return build_state(self.rcvMsgSurroundingInfo, obs_dim=OBS_DIM)

    def process_action(self, action: np.ndarray) -> tuple:
        return process_action(action)

    def reverse_process_action(self, throttle_percentage: float, braking_percentage: float, steering_angle: float) -> np.ndarray:
        return reverse_process_action(throttle_percentage, braking_percentage, steering_angle)

    def send_action(self, action: np.ndarray):
        hw = self.config.get('hardware', {})
        if bool(hw.get('dry_run', False)):
            return

        throttle_percentage, braking_percentage, steering_angle = self.process_action(action)

        state = self.get_state()
        safety_override = 0
        if not self.safety_manager.check_action_safety(action, state):
            throttle_percentage = 0
            braking_percentage = 100
            steering_angle = 0.0
            safety_override = 1

        self.last_action_sent = np.asarray(action, dtype=np.float32).copy()
        self.last_sent_throttle = int(throttle_percentage)
        self.last_sent_brake = int(braking_percentage)
        self.last_sent_steer = float(steering_angle)
        self.last_safety_override = int(safety_override)

        can_msg, _ = pack_can(
            throttle_percentage,
            braking_percentage,
            steering_angle,
            int(self.config['hardware']['can_id']),
        )
        self.carControlBus.send(can_msg)

    def _compute_action(self, state: np.ndarray) -> tuple:
        t0 = time.time()
        action = self.model.predict(state, deterministic=self.deterministic)
        infer_time = float(time.time() - t0)
        action = action.detach().cpu().numpy().astype(np.float32).flatten()
        return np.clip(action, -1.0, 1.0), infer_time

    def _store_transition_from_prev(self, next_state: np.ndarray, intervention: float, stop_boundary: bool, action_behavior: np.ndarray):
        if self._prev_state is None or self._prev_action_novice is None:
            return

        obs = np.asarray(self._prev_state, dtype=np.float32)
        next_obs = np.asarray(next_state, dtype=np.float32)
        action_novice = np.asarray(self._prev_action_novice, dtype=np.float32)
        action_behavior = np.asarray(action_behavior, dtype=np.float32)

        exp = Experience.from_pvp(
            obs=obs,
            next_obs=next_obs,
            reward=0.0,
            done=False,
            a_behavior=action_behavior,
            a_novice=action_novice,
            a_human=action_behavior,
            intervention=float(intervention),
            stop_td=0.0 if stop_boundary else 1.0,
        )

        if stop_boundary or exp.interventions > 0.5:
            self.buffer.add_human(exp)
        else:
            self.buffer.add_pvp(exp)

        self.segment_recorder.add(
            {
                'obs': obs,
                'next_obs': next_obs,
                'action_novice': action_novice,
                'action_behavior': np.asarray(action_behavior, dtype=np.float32),
                'intervention': float(intervention),
                'takeover_start': bool(stop_boundary),
                'timestamp': float(self.last_timestamp),
            }
        )

    def _train_step(self):
        tr_cfg = self.config.get('training', {}) or {}
        if not bool(tr_cfg.get('enable_online_update', False)):
            return

        bs = int(tr_cfg.get('batch_size', 128))
        gradient_steps = int(tr_cfg.get('gradient_steps', 1))
        if gradient_steps <= 0:
            return

        try:
            metrics = self.model.train(buffer=self.buffer, batch_size=bs, gradient_steps=gradient_steps)
            if metrics:
                self.training_metrics.update(metrics)
        except Exception as e:
            self.get_logger().error(f"Train failed: {e}")

    def _save_checkpoint(self):
        try:
            model_path = os.path.join(self.config['save_folder'], 'models', f"pvp_td3_torch_{self.iteration:08d}.pt")
            self.model.save(model_path)
        except Exception as e:
            self.get_logger().error(f"Model save failed: {e}")

        try:
            buffer_path = os.path.join(self.config['save_folder'], 'buffers', f"pvp_buffer_{self.iteration:08d}.pkl")
            self.buffer.save(buffer_path)
        except Exception as e:
            self.get_logger().error(f"Buffer save failed: {e}")

    def _log_metrics(self):
        if not self.use_tensorboard:
            return

        takeover_rate = float(np.mean(np.array(self.takeover_recorder)) * 100.0) if self.takeover_recorder else 0.0
        safety_stats = self.safety_manager.get_safety_stats()
        buffer_stats = self.buffer.get_statistics()

        try:
            self.writer.add_scalar('takeover_rate', takeover_rate, self.iteration)
            self.writer.add_scalar('Timing/process_time_ms', float(self.last_process_time) * 1000.0, self.iteration)
            self.writer.add_scalar('Timing/infer_time_ms', float(self.last_infer_time) * 1000.0, self.iteration)
            self.writer.add_scalar('Time/elapsed_s', float(time.time() - self.start_wall_time), self.iteration)

            self.writer.add_scalar('Buffer/human_size', float(buffer_stats['human_size']), self.iteration)
            self.writer.add_scalar('Buffer/pvp_size', float(buffer_stats['pvp_size']), self.iteration)
            self.writer.add_scalar('Buffer/total_samples', float(buffer_stats['total_samples']), self.iteration)
            self.writer.add_scalar('Buffer/total_interventions', float(buffer_stats['total_interventions']), self.iteration)

            self.writer.add_scalar('Vehicle/carspeed', float(self.last_state_carspeed), self.iteration)
            self.writer.add_scalar('Vehicle/error_yaw', float(self.last_state_error_yaw), self.iteration)
            self.writer.add_scalar('Vehicle/error_distance', float(self.last_state_error_distance), self.iteration)
            self.writer.add_scalar('Vehicle/intervention', float(self.last_intervention), self.iteration)

            self.writer.add_scalar('Safety/emergency_brakes', float(safety_stats['emergency_brakes']), self.iteration)
            self.writer.add_scalar('Safety/steer_violations', float(safety_stats['steer_violations']), self.iteration)
            self.writer.add_scalar('Safety/safety_override', float(self.last_safety_override), self.iteration)

            self.writer.add_scalar('Action/human_x', float(self.last_action_human[0]), self.iteration)
            self.writer.add_scalar('Action/human_y', float(self.last_action_human[1]), self.iteration)
            self.writer.add_scalar('Action/model_x', float(self.last_action_model[0]), self.iteration)
            self.writer.add_scalar('Action/model_y', float(self.last_action_model[1]), self.iteration)
            self.writer.add_scalar('Action/sent_throttle', float(self.last_sent_throttle), self.iteration)
            self.writer.add_scalar('Action/sent_brake', float(self.last_sent_brake), self.iteration)
            self.writer.add_scalar('Action/sent_steer', float(self.last_sent_steer), self.iteration)

            self.writer.add_scalar('Comfort/action_smooth_l1', float(self.last_action_smooth_l1), self.iteration)
        except Exception:
            pass

    def pub_callback_car_pvp_td3_torch(self):
        msg = CarRLInterface()
        now_ts = time.time()
        msg.timestamp = now_ts
        self.last_timestamp = float(now_ts)

        if self.rcvMsgSurroundingInfo is None:
            return

        if self.is_paused:
            return

        self.iteration += 1

        process_start = time.time()

        gearpos = safe_get(self.rcvMsgSurroundingInfo, 'gearpos', None)
        if gearpos is not None:
            try:
                if int(gearpos) == 2:
                    return
            except Exception:
                pass

        state = self.get_state()

        intervention = 0.0 if int(self.rcvMsgSurroundingInfo.car_run_mode) == 1 else 1.0
        self.takeover_recorder.append(intervention)

        action_behavior = self.reverse_process_action(
            self.rcvMsgSurroundingInfo.throttle_percentage,
            self.rcvMsgSurroundingInfo.braking_percentage,
            self.rcvMsgSurroundingInfo.steerangle,
        )
        self.last_action_human = np.asarray(action_behavior, dtype=np.float32).copy()

        takeover_start = bool(intervention == 1.0 and self._prev_intervention == 0.0)
        takeover_end = bool(intervention == 0.0 and self._prev_intervention == 1.0)

        algo_cfg = self.config.get('algo', {}) or {}
        takeover_stop_td = bool(algo_cfg.get('takeover_stop_td', False))
        stop_boundary = bool(takeover_start or (takeover_stop_td and takeover_end))

        self._store_transition_from_prev(
            next_state=state,
            intervention=intervention,
            stop_boundary=stop_boundary,
            action_behavior=action_behavior,
        )

        action_novice, infer_time = self._compute_action(state)
        self.last_infer_time = float(infer_time)

        self.last_action_model = np.asarray(action_novice, dtype=np.float32).copy()
        self.send_action(action_novice)

        if self.prev_action_sent_for_smooth is None:
            self.last_action_smooth_l1 = 0.0
        else:
            self.last_action_smooth_l1 = float(np.mean(np.abs(self.last_action_sent - self.prev_action_sent_for_smooth)))
        self.prev_action_sent_for_smooth = self.last_action_sent.copy()

        self._prev_state = state.copy()
        self._prev_action_novice = action_novice.copy()
        self._prev_intervention = float(intervention)

        self.last_intervention = float(intervention)
        self.last_state_error_yaw = float(getattr(self.rcvMsgSurroundingInfo, 'error_yaw', 0.0))
        self.last_state_error_distance = float(getattr(self.rcvMsgSurroundingInfo, 'error_distance', 0.0))
        self.last_state_carspeed = float(getattr(self.rcvMsgSurroundingInfo, 'carspeed', 0.0))

        tr_cfg = self.config.get('training', {}) or {}
        if self.iteration % int(tr_cfg.get('update_interval', 5)) == 0:
            self._train_step()

        if self.iteration % int(tr_cfg.get('log_save_interval', 100)) == 0:
            self._log_metrics()

        if self.iteration % int(tr_cfg.get('save_interval', 1000)) == 0:
            self._save_checkpoint()

        self.last_process_time = float(time.time() - process_start)
        msg.process_time = float(self.last_process_time)
        self.pubCar.publish(msg)

        if tjitools is not None:
            tjitools.ros_log(self.get_name(), 'Publish car_pvp_td3_torch msg !!!')


def main():
    rclpy.init()
    rosNode = CarPVPTD3Torch()
    rclpy.spin(rosNode)
    rclpy.shutdown()
