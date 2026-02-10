import os

import rclpy

from .car_dacer_torch import CarDACERTorch


class CarDACERTorchSkip(CarDACERTorch):
    def _init_algo_and_buffer(self):
        super()._init_algo_and_buffer()
        self._try_skip_to_phase3()

    def _resolve_model_path(self):
        eval_cfg = self.config.get('eval', {}) or {}
        model_path = eval_cfg.get('model_path')
        if model_path and os.path.exists(model_path):
            return str(model_path)

        model_dir = eval_cfg.get('model_dir')
        if model_dir and os.path.isdir(model_dir):
            candidates = []
            for name in os.listdir(model_dir):
                if not name.endswith('.pt'):
                    continue
                if not name.startswith('dacer_torch_'):
                    continue
                full = os.path.join(model_dir, name)
                if os.path.isfile(full):
                    candidates.append(full)
            if candidates:
                return max(candidates, key=os.path.getmtime)

        return None

    def _try_skip_to_phase3(self):
        model_path = self._resolve_model_path()
        if not model_path:
            self.get_logger().info('[SkipPhase] 未配置或未找到可加载的模型(model_path/model_dir)，保持原阶段逻辑。')
            return

        try:
            self.algorithm.load(model_path)

            # 直接跳到 Phase3：在线 PVP
            self.phase_manager.current_phase = 3
            self.phase_manager.phase1_episodes = self.phase_manager.phase1_threshold
            self.phase_manager.phase2_updates = self.phase_manager.phase2_threshold

            self.get_logger().info(f"[SkipPhase] 已加载预训练模型: {model_path}")
            self.get_logger().info('[SkipPhase] 已强制进入 Phase3 (PVP Learning)')
        except Exception as e:
            self.get_logger().error(f"[SkipPhase] 模型加载失败: {e} | 保持原阶段逻辑")


def main():
    rclpy.init()
    rosNode = CarDACERTorchSkip()
    rclpy.spin(rosNode)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
