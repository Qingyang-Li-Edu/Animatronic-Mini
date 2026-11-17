# 验证结果

这些 JSON 是当前源码与模型配置的可复核结果，不包含训练权重或受限 VRM 文件。

- `bluefish_physics_final.json`：7 个 MuJoCo 物理场景。
- `greet_stress.json`：固定基座招手策略连续回放和参数扰动。
- `greet_balance_seed37.json`：自由站立招手、多推力和 120 秒连续测试。
- `action_switching.json`：连续动作切换测试。
- `retarget_calibration.json`：VRM 骨骼与物理骨架映射校准。
- `training_summary.json`：阶段汇总和已知限制。

训练权重、TensorBoard 日志和 `.venv-train` 只保留在本机，按 `.gitignore` 排除。
VRM 文件受原作者条款限制，不提交到仓库。
