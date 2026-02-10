import csv
import datetime
import json
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import can
import numpy as np
import rclpy
import torch
import yaml
from rclpy.node import Node

from car_interfaces.msg import CarRLInterface, SurroundingInfoInterface

from .safety_manager import SafetyManager
from .torch_algorithm import PVPDACERTorch, TorchDACERConfig
from .torch_networks import DACERTorchAgent, DACERActionConfig


OBS_DIM = 344
PI = 3.1415926535


class CarDACERTorchEval(Node):
    def __init__(self, *, noisy: bool):
        super().__init__('car_dacer_torch_eval')

        self.noisy = bool(noisy)
        self.config = self._load_config()

        self.subSurrounding = self.create_subscription(
            SurroundingInfoInterface, 'surrounding_info_data', self.sub_callback_surrounding, 1
        )
        self.pubCarDACER = self.create_publisher(
            CarRLInterface, 'car_dacer_data', 10
        )
        self.timer = self.create_timer(0.1, self._on_timer)

        self.rcvMsgSurroundingInfo: Optional[SurroundingInfoInterface] = None
        self.iteration = 0
        self.start_wall_time = time.time()
        self.takeover_recorder = []
        self.takeover_max_len = 2000

        self._init_safety()
        self._init_device()
        self._init_algo()
        self._init_can()
        self._init_log_file()
        self._init_tensorboard()

        self.get_logger().info(self._startup_banner())

    def _init_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(
                log_dir=self.config['save_folder'],
                flush_secs=self.config.get('logging', {}).get('tensorboard_flush_secs', 20),
            )
            self.use_tensorboard = True
        except Exception:
            self.writer = None
            self.use_tensorboard = False

    def _startup_banner(self) -> str:
        mode = 'NOISY' if self.noisy else 'DETERMINISTIC'
        model_path = self._model_path
        return (
            f"=== DACER TORCH EVAL START ===\n"
            f"  mode: {mode}\n"
            f"  model: {model_path}\n"
            f"  log: {self._csv_path}\n"
            f"  can_interface: {self.config.get('hardware', {}).get('can_interface')}\n"
            f"  can_id: {self.config.get('hardware', {}).get('can_id')}\n"
            f"==============================="
        )

    def _load_config(self) -> dict:
        ros_node_name = 'car_dacer_torch'
        config_path = os.getcwd() + f"/src/{ros_node_name}/{ros_node_name}/" + 'config.yaml'
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        base_save = '/home/nvidia/Autodrive'
        save_folder = os.path.join(
            base_save + '/results/',
            'DACER_TORCH_EVAL_' + datetime.datetime.now().strftime('%y%m%d-%H%M%S'),
        )
        cfg['save_folder'] = save_folder
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(os.path.join(save_folder, 'eval_logs'), exist_ok=True)

        with open(os.path.join(save_folder, 'config_eval.json'), 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4)

        return cfg

    def _init_safety(self):
        self.safety_manager = SafetyManager(self.config.get('safety', {}))

    def _init_device(self):
        hw = self.config.get('hardware', {})
        use_gpu = bool(hw.get('use_gpu', True))
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def _resolve_model_path(self) -> str:
        eval_cfg = self.config.get('eval', {}) or {}
        model_path = eval_cfg.get('model_path')
        if model_path and os.path.exists(model_path):
            return str(model_path)

        model_dir = eval_cfg.get('model_dir') or '/home/nvidia/Autodrive/results/DACER_TORCH_CAR/models/'
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"model_dir not found: {model_dir}")

        candidates = []
        for name in os.listdir(model_dir):
            if not name.endswith('.pt'):
                continue
            if not name.startswith('dacer_torch_'):
                continue
            full = os.path.join(model_dir, name)
            if os.path.isfile(full):
                candidates.append(full)

        if not candidates:
            raise FileNotFoundError(f"no model found under: {model_dir}")

        return max(candidates, key=os.path.getmtime)

    def _init_algo(self):
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
        self._model_path = self._resolve_model_path()
        self.algorithm.load(self._model_path)
        self.algorithm.agent.eval()

        eval_cfg = self.config.get('eval', {}) or {}
        self.eval_deterministic = bool(eval_cfg.get('deterministic', True))
        self.eval_noise_std = float(eval_cfg.get('noise_std', 0.05))

    def _init_can(self):
        hw = self.config.get('hardware', {})
        self.dry_run = bool(hw.get('dry_run', False))
        self.carControlBus = can.interface.Bus(
            channel=hw.get('can_interface', 'can1'),
            bustype='socketcan',
        )

    def _init_log_file(self):
        ts = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
        self._csv_path = os.path.join(self.config['save_folder'], 'eval_logs', f"eval_{ts}.csv")
        self._csv_f = open(self._csv_path, 'w', newline='', encoding='utf-8')
        self._csv_writer = csv.DictWriter(self._csv_f, fieldnames=self._csv_header())
        self._csv_writer.writeheader()
        self._csv_f.flush()

    def _csv_header(self):
        return [
            'ts',
            'iter',
            'process_time',
            'infer_time',
            'gearpos',
            'car_run_mode',
            'error_yaw',
            'error_distance',
            'carspeed',
            'turn_signals',
            'human_throttle_percentage',
            'human_braking_percentage',
            'human_steerangle',
            'state_vec',
            'action_raw_0',
            'action_raw_1',
            'action_sent_0',
            'action_sent_1',
            'sent_throttle_percentage',
            'sent_braking_percentage',
            'sent_steering_angle',
            'safety_override',
            'can_bytes',
            'surroundinginfo',
            'path_rfu',
            'model_path',
            'mode',
        ]

    def sub_callback_surrounding(self, msg: SurroundingInfoInterface):
        self.rcvMsgSurroundingInfo = msg

    def get_state(self) -> np.ndarray:
        if self.rcvMsgSurroundingInfo is None:
            return np.zeros(OBS_DIM, dtype=np.float32)

        msg = self.rcvMsgSurroundingInfo
        state = []

        errorYaw = float(getattr(msg, 'error_yaw', 0.0)) / PI
        errorYaw = np.clip(errorYaw, -1.0, 1.0)
        state.append(errorYaw)

        errorDistance = float(getattr(msg, 'error_distance', 0.0)) / 5.0
        errorDistance = np.clip(errorDistance, -1.0, 1.0)
        state.append(errorDistance)

        carspeed = float(getattr(msg, 'carspeed', 0.0)) / 15.0
        carspeed = np.clip(carspeed, 0.0, 1.0)
        state.append(carspeed)

        turn_state = float(getattr(msg, 'turn_signals', -1.0))
        turn_state = (turn_state + 1.0) / 2.0
        state.append(turn_state)

        radar_data = list(getattr(msg, 'surroundinginfo', []))
        expected_radar_len = 240
        if len(radar_data) < expected_radar_len:
            radar_data.extend([1.0] * (expected_radar_len - len(radar_data)))
        elif len(radar_data) > expected_radar_len:
            radar_data = radar_data[:expected_radar_len]
        state.extend(radar_data)

        path = list(getattr(msg, 'path_rfu', []))
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

    def process_action(self, action: np.ndarray) -> Tuple[int, int, float]:
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

    def _pack_can(self, throttle_percentage: int, braking_percentage: int, steering_angle: float) -> Tuple[can.Message, list]:
        gearpos = 0x03
        enableSignal = 1
        ultrasonicSwitch = 0
        dippedHeadlight = 0
        contourLamp = 0

        brakeEnable = 1 if int(braking_percentage) != 0 else 0
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

        msg = can.Message(
            arbitration_id=self.config.get('hardware', {}).get('can_id', 0x210),
            data=canData,
            extended_id=False,
        )
        return msg, canData

    @torch.no_grad()
    def _compute_action(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        t0 = time.time()
        obs = torch.from_numpy(np.expand_dims(state, axis=0).astype(np.float32)).to(self.device)
        action = self.algorithm.get_action(obs, deterministic=self.eval_deterministic, add_noise=False)
        infer_time = float(time.time() - t0)
        action = action.detach().cpu().numpy().flatten().astype(np.float32)
        action = np.clip(action, -1.0, 1.0)

        action_sent = action.copy()
        if self.noisy:
            noise = np.random.randn(*action_sent.shape).astype(np.float32) * float(self.eval_noise_std)
            action_sent = np.clip(action_sent + noise, -1.0, 1.0)

        return action, action_sent, infer_time

    def _send_action(self, action_sent: np.ndarray, state: np.ndarray) -> Tuple[int, int, float, bool, list]:
        throttle_percentage, braking_percentage, steering_angle = self.process_action(action_sent)
        safety_override = False

        if not self.safety_manager.check_action_safety(action_sent, state):
            throttle_percentage = 0
            braking_percentage = 100
            steering_angle = 0.0
            safety_override = True

        can_msg, can_bytes = self._pack_can(throttle_percentage, braking_percentage, steering_angle)

        if not self.dry_run:
            self.carControlBus.send(can_msg)

        return throttle_percentage, braking_percentage, steering_angle, safety_override, can_bytes

    def _safe_get(self, msg: Any, name: str, default: Any = None) -> Any:
        try:
            return getattr(msg, name)
        except Exception:
            return default

    def _on_timer(self):
        now_ts = time.time()
        msg_pub = CarRLInterface()
        msg_pub.timestamp = now_ts

        if self.rcvMsgSurroundingInfo is None:
            return

        self.iteration += 1
        state = self.get_state()

        gearpos = self._safe_get(self.rcvMsgSurroundingInfo, 'gearpos', None)
        if gearpos is not None:
            try:
                if int(gearpos) == 2:
                    return
            except Exception:
                pass

        action_raw, action_sent, infer_time = self._compute_action(state)
        sent_th, sent_br, sent_steer, safety_override, can_bytes = self._send_action(action_sent, state)

        process_time = time.time() - now_ts
        msg_pub.process_time = process_time
        self.pubCarDACER.publish(msg_pub)

        row = {
            'ts': now_ts,
            'iter': self.iteration,
            'process_time': process_time,
            'infer_time': float(infer_time),
            'gearpos': self._safe_get(self.rcvMsgSurroundingInfo, 'gearpos', None),
            'car_run_mode': self._safe_get(self.rcvMsgSurroundingInfo, 'car_run_mode', None),
            'error_yaw': self._safe_get(self.rcvMsgSurroundingInfo, 'error_yaw', None),
            'error_distance': self._safe_get(self.rcvMsgSurroundingInfo, 'error_distance', None),
            'carspeed': self._safe_get(self.rcvMsgSurroundingInfo, 'carspeed', None),
            'turn_signals': self._safe_get(self.rcvMsgSurroundingInfo, 'turn_signals', None),
            'human_throttle_percentage': self._safe_get(self.rcvMsgSurroundingInfo, 'throttle_percentage', None),
            'human_braking_percentage': self._safe_get(self.rcvMsgSurroundingInfo, 'braking_percentage', None),
            'human_steerangle': self._safe_get(self.rcvMsgSurroundingInfo, 'steerangle', None),
            'state_vec': json.dumps(state.tolist(), ensure_ascii=False),
            'action_raw_0': float(action_raw[0]),
            'action_raw_1': float(action_raw[1]),
            'action_sent_0': float(action_sent[0]),
            'action_sent_1': float(action_sent[1]),
            'sent_throttle_percentage': int(sent_th),
            'sent_braking_percentage': int(sent_br),
            'sent_steering_angle': float(sent_steer),
            'safety_override': int(1 if safety_override else 0),
            'can_bytes': json.dumps(can_bytes, ensure_ascii=False),
            'surroundinginfo': json.dumps(list(self._safe_get(self.rcvMsgSurroundingInfo, 'surroundinginfo', [])), ensure_ascii=False),
            'path_rfu': json.dumps(list(self._safe_get(self.rcvMsgSurroundingInfo, 'path_rfu', [])), ensure_ascii=False),
            'model_path': self._model_path,
            'mode': 'noisy' if self.noisy else 'deterministic',
        }
        self._csv_writer.writerow(row)
        self._csv_f.flush()

        # 评估指标：TensorBoard 记录（节流，避免刷屏/写盘过快）
        if self.use_tensorboard and (self.iteration % 10 == 0):
            car_run_mode = self._safe_get(self.rcvMsgSurroundingInfo, 'car_run_mode', 1)
            try:
                intervention = 0.0 if int(car_run_mode) == 1 else 1.0
            except Exception:
                intervention = 0.0

            self.takeover_recorder.append(float(intervention))
            if len(self.takeover_recorder) > int(self.takeover_max_len):
                self.takeover_recorder = self.takeover_recorder[-int(self.takeover_max_len):]
            takeover_rate = float(np.mean(np.asarray(self.takeover_recorder)) * 100.0) if self.takeover_recorder else 0.0

            self.writer.add_scalar('Timing/process_time_ms', float(process_time) * 1000.0, self.iteration)
            self.writer.add_scalar('Timing/infer_time_ms', float(infer_time) * 1000.0, self.iteration)
            self.writer.add_scalar('Time/elapsed_s', float(time.time() - self.start_wall_time), self.iteration)
            self.writer.add_scalar('Vehicle/carspeed', float(self._safe_get(self.rcvMsgSurroundingInfo, 'carspeed', 0.0)), self.iteration)
            self.writer.add_scalar('Vehicle/error_yaw', float(self._safe_get(self.rcvMsgSurroundingInfo, 'error_yaw', 0.0)), self.iteration)
            self.writer.add_scalar('Vehicle/error_distance', float(self._safe_get(self.rcvMsgSurroundingInfo, 'error_distance', 0.0)), self.iteration)
            self.writer.add_scalar('Vehicle/takeover_rate', float(takeover_rate), self.iteration)
            self.writer.add_scalar('takeover_rate', float(takeover_rate), self.iteration)

            self.writer.add_scalar('Action/raw_x', float(action_raw[0]), self.iteration)
            self.writer.add_scalar('Action/raw_y', float(action_raw[1]), self.iteration)
            self.writer.add_scalar('Action/sent_x', float(action_sent[0]), self.iteration)
            self.writer.add_scalar('Action/sent_y', float(action_sent[1]), self.iteration)
            self.writer.add_scalar('Action/sent_throttle', float(sent_th), self.iteration)
            self.writer.add_scalar('Action/sent_brake', float(sent_br), self.iteration)
            self.writer.add_scalar('Action/sent_steer', float(sent_steer), self.iteration)
            self.writer.add_scalar('Safety/safety_override', float(1.0 if safety_override else 0.0), self.iteration)

    def destroy_node(self):
        try:
            self._csv_f.flush()
            self._csv_f.close()
        except Exception:
            pass
        super().destroy_node()


def _main(noisy: bool):
    rclpy.init()
    node = CarDACERTorchEval(noisy=noisy)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main_det():
    _main(noisy=False)


def main_noisy():
    _main(noisy=True)
