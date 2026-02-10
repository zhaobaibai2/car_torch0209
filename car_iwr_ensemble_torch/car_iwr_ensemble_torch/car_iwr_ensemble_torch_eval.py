import csv
import datetime
import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import can
import numpy as np
import rclpy
import torch
import yaml
from car_interfaces.msg import CarRLInterface, SurroundingInfoInterface
from rclpy.node import Node

from .iwr_model import Ensemble
from .realcar_utils import OBS_DIM, build_state, pack_can, process_action as rc_process_action, safe_get
from .safety_manager import SafetyManager

ros_node_name = "car_iwr_ensemble_torch"

class CarIWREnsembleTorchEval(Node):
    def __init__(self, *, noisy: bool):
        super().__init__("car_iwr_ensemble_torch_eval")

        self.noisy = bool(noisy)
        self.config = self._load_config()

        self.subSurrounding = self.create_subscription(
            SurroundingInfoInterface, "surrounding_info_data", self.sub_callback_surrounding, 1
        )
        self.pubCar = self.create_publisher(CarRLInterface, "car_iwr_ensemble_data", 10)
        self.timer = self.create_timer(0.1, self._on_timer)

        self.rcvMsgSurroundingInfo: Optional[SurroundingInfoInterface] = None
        self.iteration = 0
        self.start_wall_time = time.time()
        self.takeover_recorder = []
        self.takeover_max_len = 2000

        self.prev_action_sent_for_smooth = None

        self._init_safety()
        self._init_device()
        self._init_model()
        self._init_can()
        self._init_log_file()
        self._init_tensorboard()

        self.get_logger().info(self._startup_banner())

    def _init_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(
                log_dir=self.config["save_folder"],
                flush_secs=self.config.get("logging", {}).get("tensorboard_flush_secs", 20),
            )
            self.use_tensorboard = True
        except Exception:
            self.writer = None
            self.use_tensorboard = False

    def _startup_banner(self) -> str:
        mode = "NOISY" if self.noisy else "DETERMINISTIC"
        return (
            "=== IWR ENSEMBLE TORCH EVAL START ===\n"
            f"  mode: {mode}\n"
            f"  model: {self._model_path}\n"
            f"  log: {self._csv_path}\n"
            f"  can_interface: {self.config.get('hardware', {}).get('can_interface')}\n"
            f"  can_id: {self.config.get('hardware', {}).get('can_id')}\n"
            "==============================="
        )

    def _load_config(self) -> dict:
        config_path = self._resolve_config_path()
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        base_save = "/home/nvidia/Autodrive"
        save_folder = os.path.join(
            base_save,
            "results",
            "IWR_EVAL_" + datetime.datetime.now().strftime("%y%m%d-%H%M%S"),
        )
        cfg["save_folder"] = save_folder
        os.makedirs(save_folder, exist_ok=True)
        os.makedirs(os.path.join(save_folder, "eval_logs"), exist_ok=True)

        with open(os.path.join(save_folder, "config_eval.json"), "w", encoding="utf-8") as f:
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

    def _init_safety(self):
        self.safety_manager = SafetyManager(self.config.get("safety", {}))

    def _init_device(self):
        hw = self.config.get("hardware", {})
        use_gpu = bool(hw.get("use_gpu", True))
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def _resolve_model_path(self) -> str:
        eval_cfg = self.config.get("eval", {}) or {}
        model_path = eval_cfg.get("model_path")
        if model_path and os.path.exists(model_path):
            return str(model_path)

        model_dir = eval_cfg.get("model_dir")
        if not model_dir or not os.path.isdir(model_dir):
            raise FileNotFoundError(f"model_dir not found: {model_dir}")

        candidates = []
        for name in os.listdir(model_dir):
            if not (name.endswith(".pth") or name.endswith(".pt")):
                continue
            full = os.path.join(model_dir, name)
            if os.path.isfile(full):
                candidates.append(full)

        if not candidates:
            raise FileNotFoundError(f"no model found under: {model_dir}")

        return max(candidates, key=os.path.getmtime)

    def _init_model(self):
        env_cfg = self.config.get("env", {})
        obs_dim = int(env_cfg.get("state_dim", OBS_DIM))
        act_dim = int(env_cfg.get("action_dim", 2))

        model_cfg = self.config.get("model", {}) or {}
        hidden_sizes = tuple(int(x) for x in model_cfg.get("hidden_sizes", [256, 256]))
        num_nets = int(model_cfg.get("num_nets", 5))

        self.model = Ensemble(
            observation_shape=(obs_dim,),
            action_shape=(act_dim,),
            device=self.device,
            hidden_sizes=hidden_sizes,
            num_nets=num_nets,
        )

        self._model_path = self._resolve_model_path()
        self.model.load(self._model_path)

        eval_cfg = self.config.get("eval", {}) or {}
        self.eval_deterministic = bool(eval_cfg.get("deterministic", True))
        self.eval_noise_std = float(eval_cfg.get("noise_std", 0.05))

    def _init_can(self):
        hw = self.config.get("hardware", {})
        self.dry_run = bool(hw.get("dry_run", False))
        self.carControlBus = can.interface.Bus(
            channel=hw.get("can_interface", "can1"),
            bustype="socketcan",
        )

    def _init_log_file(self):
        ts = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        self._csv_path = os.path.join(self.config["save_folder"], "eval_logs", f"eval_{ts}.csv")
        self._csv_f = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_f, fieldnames=self._csv_header())
        self._csv_writer.writeheader()
        self._csv_f.flush()

    def _csv_header(self):
        return [
            "ts",
            "iter",
            "process_time",
            "gearpos",
            "car_run_mode",
            "error_yaw",
            "error_distance",
            "carspeed",
            "turn_signals",
            "human_throttle_percentage",
            "human_braking_percentage",
            "human_steerangle",
            "state_vec",
            "action_raw_0",
            "action_raw_1",
            "action_sent_0",
            "action_sent_1",
            "sent_throttle_percentage",
            "sent_braking_percentage",
            "sent_steering_angle",
            "safety_override",
            "action_smooth_l1",
            "can_bytes",
            "surroundinginfo",
            "path_rfu",
            "model_path",
            "mode",
        ]

    def sub_callback_surrounding(self, msg: SurroundingInfoInterface):
        self.rcvMsgSurroundingInfo = msg

    def get_state(self) -> np.ndarray:
        return build_state(self.rcvMsgSurroundingInfo, obs_dim=OBS_DIM)

    def process_action(self, action: np.ndarray) -> Tuple[int, int, float]:
        return rc_process_action(action)

    def _compute_action(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        t0 = time.time()
        action = self.model.act(state, i=-1)
        infer_time = float(time.time() - t0)

        action = np.asarray(action, dtype=np.float32).flatten()
        action = np.clip(action, -1.0, 1.0)

        action_sent = action.copy()
        if self.noisy:
            noise = np.random.randn(*action_sent.shape).astype(np.float32) * float(self.eval_noise_std)
            action_sent = np.clip(action_sent + noise, -1.0, 1.0)

        return action, action_sent, infer_time

    def _send_action(self, action_sent: np.ndarray, state: np.ndarray) -> Tuple[int, int, float, bool, list, np.ndarray]:
        throttle_percentage, braking_percentage, steering_angle = self.process_action(action_sent)
        safety_override = False

        if not self.safety_manager.check_action_safety(action_sent, state):
            throttle_percentage = 0
            braking_percentage = 100
            steering_angle = 0.0
            safety_override = True

        can_msg, can_bytes = pack_can(
            throttle_percentage,
            braking_percentage,
            steering_angle,
            int(self.config.get("hardware", {}).get("can_id", 0x210)),
        )

        if not self.dry_run:
            self.carControlBus.send(can_msg)

        action_sent_used = np.asarray(action_sent, dtype=np.float32).copy()
        return throttle_percentage, braking_percentage, steering_angle, safety_override, can_bytes, action_sent_used

    def _on_timer(self):
        now_ts = time.time()
        msg_pub = CarRLInterface()
        msg_pub.timestamp = now_ts

        if self.rcvMsgSurroundingInfo is None:
            return

        self.iteration += 1
        state = self.get_state()

        gearpos = safe_get(self.rcvMsgSurroundingInfo, "gearpos", None)
        if gearpos is not None:
            try:
                if int(gearpos) == 2:
                    return
            except Exception:
                pass

        action_raw, action_sent, infer_time = self._compute_action(state)
        sent_th, sent_br, sent_steer, safety_override, can_bytes, action_sent_used = self._send_action(action_sent, state)

        if self.prev_action_sent_for_smooth is None:
            action_smooth_l1 = 0.0
        else:
            action_smooth_l1 = float(np.mean(np.abs(action_sent_used - self.prev_action_sent_for_smooth)))
        self.prev_action_sent_for_smooth = action_sent_used.copy()

        process_time = time.time() - now_ts
        msg_pub.process_time = process_time
        self.pubCar.publish(msg_pub)

        row = {
            "ts": now_ts,
            "iter": self.iteration,
            "process_time": process_time,
            "gearpos": safe_get(self.rcvMsgSurroundingInfo, "gearpos", None),
            "car_run_mode": safe_get(self.rcvMsgSurroundingInfo, "car_run_mode", None),
            "error_yaw": safe_get(self.rcvMsgSurroundingInfo, "error_yaw", None),
            "error_distance": safe_get(self.rcvMsgSurroundingInfo, "error_distance", None),
            "carspeed": safe_get(self.rcvMsgSurroundingInfo, "carspeed", None),
            "turn_signals": safe_get(self.rcvMsgSurroundingInfo, "turn_signals", None),
            "human_throttle_percentage": safe_get(self.rcvMsgSurroundingInfo, "throttle_percentage", None),
            "human_braking_percentage": safe_get(self.rcvMsgSurroundingInfo, "braking_percentage", None),
            "human_steerangle": safe_get(self.rcvMsgSurroundingInfo, "steerangle", None),
            "state_vec": json.dumps(state.tolist(), ensure_ascii=False),
            "action_raw_0": float(action_raw[0]),
            "action_raw_1": float(action_raw[1]),
            "action_sent_0": float(action_sent_used[0]),
            "action_sent_1": float(action_sent_used[1]),
            "sent_throttle_percentage": int(sent_th),
            "sent_braking_percentage": int(sent_br),
            "sent_steering_angle": float(sent_steer),
            "safety_override": int(1 if safety_override else 0),
            "action_smooth_l1": float(action_smooth_l1),
            "can_bytes": json.dumps(can_bytes, ensure_ascii=False),
            "surroundinginfo": json.dumps(list(safe_get(self.rcvMsgSurroundingInfo, "surroundinginfo", [])), ensure_ascii=False),
            "path_rfu": json.dumps(list(safe_get(self.rcvMsgSurroundingInfo, "path_rfu", [])), ensure_ascii=False),
            "model_path": self._model_path,
            "mode": "noisy" if self.noisy else "deterministic",
        }
        self._csv_writer.writerow(row)
        self._csv_f.flush()

        if self.use_tensorboard and (self.iteration % 10 == 0):
            car_run_mode = safe_get(self.rcvMsgSurroundingInfo, "car_run_mode", 1)
            try:
                intervention = 0.0 if int(car_run_mode) == 1 else 1.0
            except Exception:
                intervention = 0.0

            self.takeover_recorder.append(float(intervention))
            if len(self.takeover_recorder) > int(self.takeover_max_len):
                self.takeover_recorder = self.takeover_recorder[-int(self.takeover_max_len) :]
            takeover_rate = float(np.mean(np.asarray(self.takeover_recorder)) * 100.0) if self.takeover_recorder else 0.0

            self.writer.add_scalar("Timing/process_time_ms", float(process_time) * 1000.0, self.iteration)
            self.writer.add_scalar("Timing/infer_time_ms", float(infer_time) * 1000.0, self.iteration)
            self.writer.add_scalar("Time/elapsed_s", float(time.time() - self.start_wall_time), self.iteration)
            self.writer.add_scalar("Vehicle/carspeed", float(safe_get(self.rcvMsgSurroundingInfo, "carspeed", 0.0)), self.iteration)
            self.writer.add_scalar("Vehicle/error_yaw", float(safe_get(self.rcvMsgSurroundingInfo, "error_yaw", 0.0)), self.iteration)
            self.writer.add_scalar("Vehicle/error_distance", float(safe_get(self.rcvMsgSurroundingInfo, "error_distance", 0.0)), self.iteration)
            self.writer.add_scalar("Vehicle/takeover_rate", float(takeover_rate), self.iteration)

            self.writer.add_scalar("Action/raw_x", float(action_raw[0]), self.iteration)
            self.writer.add_scalar("Action/raw_y", float(action_raw[1]), self.iteration)
            self.writer.add_scalar("Action/sent_x", float(action_sent_used[0]), self.iteration)
            self.writer.add_scalar("Action/sent_y", float(action_sent_used[1]), self.iteration)
            self.writer.add_scalar("Action/sent_throttle", float(sent_th), self.iteration)
            self.writer.add_scalar("Action/sent_brake", float(sent_br), self.iteration)
            self.writer.add_scalar("Action/sent_steer", float(sent_steer), self.iteration)
            self.writer.add_scalar("Safety/safety_override", float(1.0 if safety_override else 0.0), self.iteration)

            self.writer.add_scalar("Comfort/action_smooth_l1", float(action_smooth_l1), self.iteration)

    def destroy_node(self):
        try:
            self._csv_f.flush()
            self._csv_f.close()
        except Exception:
            pass
        super().destroy_node()


def _main(noisy: bool):
    rclpy.init()
    node = CarIWREnsembleTorchEval(noisy=noisy)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main_det():
    _main(noisy=False)


def main_noisy():
    _main(noisy=True)
