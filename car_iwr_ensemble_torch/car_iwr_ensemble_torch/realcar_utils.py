import json
from typing import Any, Tuple

import numpy as np
import can


OBS_DIM = 344
PI = 3.1415926535


def safe_get(msg: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(msg, name)
    except Exception:
        return default


def build_state(msg: Any, obs_dim: int = OBS_DIM) -> np.ndarray:
    if msg is None:
        return np.zeros(obs_dim, dtype=np.float32)

    state = []

    error_yaw = float(safe_get(msg, "error_yaw", 0.0)) / PI
    error_yaw = np.clip(error_yaw, -1.0, 1.0)
    state.append(error_yaw)

    error_distance = float(safe_get(msg, "error_distance", 0.0)) / 5.0
    error_distance = np.clip(error_distance, -1.0, 1.0)
    state.append(error_distance)

    carspeed = float(safe_get(msg, "carspeed", 0.0)) / 15.0
    carspeed = np.clip(carspeed, 0.0, 1.0)
    state.append(carspeed)

    turn_state = float(safe_get(msg, "turn_signals", -1.0))
    turn_state = (turn_state + 1.0) / 2.0
    state.append(turn_state)

    radar_data = list(safe_get(msg, "surroundinginfo", []) or [])
    expected_radar_len = 240
    if len(radar_data) < expected_radar_len:
        radar_data.extend([1.0] * (expected_radar_len - len(radar_data)))
    elif len(radar_data) > expected_radar_len:
        radar_data = radar_data[:expected_radar_len]
    state.extend(radar_data)

    path = list(safe_get(msg, "path_rfu", []) or [])
    expected_path_len = 100
    if len(path) < expected_path_len:
        path.extend([0.0] * (expected_path_len - len(path)))
    elif len(path) > expected_path_len:
        path = path[:expected_path_len]
    path = np.array(path, dtype=np.float32) / 30.0
    path = np.clip(path, -1.0, 1.0)
    state.extend(list(path))

    final_state = np.array(state, dtype=np.float32)
    if len(final_state) != obs_dim:
        fixed = np.zeros(obs_dim, dtype=np.float32)
        n = min(len(final_state), obs_dim)
        fixed[:n] = final_state[:n]
        final_state = fixed

    return final_state


def process_action(action: np.ndarray) -> Tuple[int, int, float]:
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


def reverse_process_action(throttle_percentage: float, braking_percentage: float, steering_angle: float) -> np.ndarray:
    if braking_percentage != 0:
        x = -float(braking_percentage / 100)
    else:
        x = float(throttle_percentage / 100)
    x = np.clip(x, -1.0, 1.0)

    y = float(steering_angle) / 200.0
    y = np.clip(y, -1.0, 1.0)

    return np.array([x, y], dtype=np.float32)


def pack_can(throttle_percentage: int, braking_percentage: int, steering_angle: float, can_id: int) -> Tuple[can.Message, list]:
    gearpos = 0x03
    enableSignal = 1
    ultrasonicSwitch = 0
    dippedHeadlight = 0
    contourLamp = 0

    brakeEnable = 1 if int(braking_percentage) != 0 else 0
    alarmLamp = 0
    turnSignalControl = 0
    appInsert = 0

    byte7 = appInsert << 7 | turnSignalControl << 1 | alarmLamp
    byte6 = int(braking_percentage) << 1 | brakeEnable
    byte5 = (int(steering_angle) & 0xFF00) >> 8
    byte4 = int(steering_angle) & 0x00FF
    byte3 = 0
    byte2 = ((int(throttle_percentage) * 10) & 0xFF00) >> 8
    byte1 = (int(throttle_percentage) * 10) & 0x00FF
    byte0 = gearpos << 6 | enableSignal << 5 | ultrasonicSwitch << 4 | dippedHeadlight << 1 | contourLamp

    can_data = [byte0, byte1, byte2, byte3, byte4, byte5, byte6, byte7]

    msg = can.Message(
        arbitration_id=can_id,
        data=can_data,
        extended_id=False,
    )
    return msg, can_data


def dumps_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return "[]"
