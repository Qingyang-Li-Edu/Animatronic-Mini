# Mac 本地训练与物理验证

## 可复现环境

训练使用项目内的 `.venv-train`（Homebrew Python 3.12），与 ROS Docker 和
`chibi-sim` conda 分开。conda 的 OpenBLAS/OpenMP 与 PyTorch 自带 OpenMP
冲突，隔离环境导入及张量运算已通过，不设置 `KMP_DUPLICATE_LIB_OK`。

```bash
/opt/homebrew/bin/python3.12 -m venv .venv-train
.venv-train/bin/python -m pip install -r training/requirements-macos.lock.txt
```

当前选用 CPU、两个 spawn 仿真进程，每进程一个 Torch 线程。当前进程的
`torch.backends.mps.is_available()` 返回 False，因此没有声称使用 GPU。

## 物理模型

```bash
.venv-train/bin/python src/chibi_sim/scripts/prepare_training_models.py
.venv-train/bin/python src/chibi_sim/scripts/validate_training_physics.py
```

训练模型是不可见物理骨架 `bluefish_physics_free.xml`；外观统一使用本机下载的 `BlueFatFish.vrm` 套壳。VRM 不进入碰撞计算。

验证报告：`validation/reports/bluefish_physics_final.json`。覆盖静立、1/2/4 ms 步长、
质量 0.8/1.2 倍与摩擦 0.5/1.0、10 cm 落地和 65 度倾倒碰撞。
静立检查要求姿态偏差 <10 度、最终状态速度范数 <0.1、重心投影在足底
支撑多边形内留出 5 mm、接触法向力与重力差异 <10%，且无仿真警告。
倾倒场景只验证头/躯干触地与有限状态，不要求自行站起。

这些阈值用于假设参数下的仿真检查。质量、惯量、摩擦和电机限制尚未经
实物测量，静立通过不能证明已经学会抗扰平衡或行走。

## 固定基座动作训练

```bash
.venv-train/bin/python training/train_pose.py --dry-run
.venv-train/bin/python training/train_pose.py --motion greet \
  --timesteps 262144 --workers 2 --device cpu \
  --out training/checkpoints/ppo-safe
.venv-train/bin/python training/evaluate.py --motion greet \
  --policy training/checkpoints/ppo-safe/greet.zip \
  --out validation/reports/greet_ppo_safe.json
```

默认策略为 PPO 残差控制：原动作轨迹作为参考，策略在每个执行器上学习
最多 ±0.12 rad 的位置修正，随后受位置限位和配置力矩 65% 的硬上限约束。
这是在原轨迹上训练误差补偿，不能称为从零学会自主动作。
奖励包括位置误差、速度、动作幅度、力矩占比和动作变化惩罚。

单独评估静止控制、直接参考轨迹与 PPO，报告活动关节和全体关节 RMSE，
以及相对于源模型配置上限的最大力矩比。先前的 NumPy 指令回归脚本
`train_pose_linear.py` 可用于检查数据通路；目标本身已在输入中，不能把它
当作独立自主策略的证据。

训练产物保存位置：

- `training/checkpoints/ppo-safe/greet.zip`：PPO 权重。
- 同名 `.json`：种子、执行步数、设备、依赖和模型/动作文件 SHA-256。
- `training/logs/ppo-greet-safe.log`：控制台日志。
- `validation/reports/`：物理与策略评估结果。

2026-11-17 的招手评估：

| 指标 | 直接参考轨迹 | 残差 PPO |
| --- | ---: | ---: |
| 活动关节 RMSE（rad） | 0.018366 | 0.005377 |
| 全关节 RMSE（rad） | 0.009919 | 0.004041 |
| 标称峰值力矩/源配置上限 | 0.2163 | 0.5833 |

误差改善伴随更高力矩需求。`greet_stress.json` 记录不重置物理状态的 100 轮
连续回放，以及种子 100–109 的十组独立身体质量/惯量 0.8–1.2 倍和初始
状态扰动。最差活动关节 RMSE 为 0.005620 rad，峰值力矩比 0.65；预设
门槛为 RMSE <0.05 rad、力矩比 <0.7，全部通过。

## 自由基座站立与推力实验

```bash
.venv-train/bin/python training/train_balance.py --timesteps 524288 \
  --seed 7 --push-force .8 --out training/checkpoints/balance-seed7
.venv-train/bin/python training/evaluate_balance.py \
  --policy training/checkpoints/balance-seed7/stand.zip \
  --out validation/reports/balance.json
```

`balance_env.py` 使用自由根部模型，70 维观测与 10 个腿部关节动作，100 Hz
控制。观测包括投影重力、根部速度、关节状态、上一步动作、双足接触和高度，
不提供未来推力信息。PPO 在中立姿态附近输出 ±0.12 rad 修正，保留原位置
伺服器，力矩硬上限为原配置的 65%。头部和手臂保持中立。

每回合 6 秒，在 1/2.5/4 秒向躯干施加持续 0.12 秒的水平力。质量和惯量
按身体独立随机缩放 0.8–1.2 倍，摩擦 0.5–1.0，初始倾角 ±2 度，并扰动
关节状态。倾角 >35 度、高度 <0.115 m、非足部触地或仿真警告会提前终止。

评估用未参与训练的种子 1000–1029，与中立伺服指令成对比较。每回合需
完成 6 秒、最终倾角 <5 度、根部速度范数 <0.1、位移 <0.05 m、力矩比 <0.7。
速度范数包含自由根部线速度和角速度，仅用作本项目的联合指标。
0.8 N 场景要求至少 90% 成功率，同时报告倾角均方根检验学习是否有益。

基线在 0.8 N 下本身可以站稳，因此通过该门槛不能独立证明 PPO 有改善。
更强推力用于探索当前模型与控制器的适用边界，结果不外推至实物。
训练和评估结果以实际报告为准；未通过的实验保留供诊断。

首轮种子 7（训练推力 0.8 N，524,288 步）在 30 个留出种子中的结果：

| 测试推力 | 中立指令成功数 | PPO 成功数 | 中立倾角 RMS | PPO 倾角 RMS |
| --- | ---: | ---: | ---: | ---: |
| 0 N | 30/30 | 30/30 | 0.144° | 0.150° |
| 0.8 N | 30/30 | 30/30 | 0.283° | 0.162° |
| 3 N | 20/30 | 23/30 | 3.836° | 2.454° |

0.8 N 时平均倾角 RMS 降低约 42.8%；无推力时略有回退。3 N 仍有 7 组
失败，失败回合会提前结束，强推力下应优先比较成功率，不能只看平均倾角。
这轮只证明指定模型和测试分布下的改善，不能声称任意扰动均能恢复。

同配置独立训练种子 27 也完成 524,288 步。0.8 N 留出测试同样 30/30 通过，
平均倾角 RMS 为 0.165°（比中立指令低约 41.7%）；3 N 下同样为 23/30。
两轮复现了弱扰动下的改善，尚不足以覆盖更多训练种子或真实硬件。

种子 17 的强推力训练同样执行 524,288 步，但训练推力增加到 3 N。
留出评估中，3 N 成功数达到 24/30（80%），仍未达到 90% 的强推力门槛；
无推力/0.8 N 的平均倾角 RMS 反而增加到 0.440°/0.463°。该权重作为
实验保存于 `balance-push3-seed17/stand.zip`，不替换首轮弱扰动策略。
训练目录对应的 JSON 报告分别记录任务通过状态与弱扰动下的学习改善，
应结合具体推力阅读，不能把 0.8 N 通过泛化为 3 N 通过。
汇总报告为 `validation/reports/training_summary.json`；三轮完整评估分别为
`balance_seed7.json`、`balance_seed27.json`、`balance_push3_seed17.json`。

## 查看训练与真实物理回放

```bash
.venv-train/bin/tensorboard --logdir training/checkpoints --host 127.0.0.1 --port 6006
```

打开 <http://127.0.0.1:6006/>，查看 `rollout/ep_rew_mean` 和
`rollout/ep_len_mean`；不同任务的奖励定义不同，不能直接跨任务比较数值。

默认外观入口使用下载的 VRM 套壳，实时显示 PPO 物理状态：

```bash
.venv-train/bin/python scripts/serve_bluefish_viewer.py --source training
```

打开 <http://127.0.0.1:8766/>，选择平衡、强扰动或招手。页面展示实际仿真的根部位移、旋转和关节状态，断开时保持最后姿态并提示状态。原版 VRM 的脸、材质和蒙皮保留。

`training/replay_balance.py` 默认打开该外观页面。原生 MuJoCo 窗口只作为碰撞调试，可显式传入 `--physics-debug`，不会再显示自建人物外观。

自建外观生成器与 35 个 STL 网格已移除。`bluefish_physics*.xml` 仅保留物理骨架，`bluefish_shell_migration.json` 记录换壳前后的物理等价验证；原评估报告和权重保留，旧外观录像不再作为当前预览入口。

## 尚未完成

尚无行走策略、跌倒起身策略或实物校准。只有在多训练种子和更广扰动中
证实自由基座控制有效后，才适合推进抬脚/行走奖励。

## 自由站立招手

`greet_balance_env.py` 将已训练的招手上半身策略与自由基座腿部平衡策略组合，
再以 262,144 步 PPO 微调腿部控制。回合包含 0.5 秒稳定段、招手周期和安静尾段，
评估同时记录倾角、根部漂移、招手活动关节误差、脚底移动和接触点滑移。

结果报告：`validation/reports/greet_balance_seed37.json`。

- 0.8 N：30/30 通过，倾角 RMS 0.160°，手臂 RMSE 0.00830 rad。
- 3 N：26/30 通过，成功率 86.7%，尚未达到 90% 门槛。
- 连续 120 秒、不断开 MuJoCo 状态：通过；最终根部速度 0.0217、漂移 9.6 mm。
- 接触点滑移积分约 0.22 m/120 秒，属于当前代理脚底和未校准摩擦参数的测量结果。

网页新增 `站立招手` 与 `招手 · 3 N`。只有通过报告中的策略才会被桥接服务加载，
避免把未验证权重误当作当前策略。

动作切换报告为 `validation/reports/action_switching.json`：在不重置 MuJoCo 状态的 32 秒回放中依次执行稳定、招手、静止、再次招手和结束静止，最大倾角 0.64°、根部漂移 4.1 mm、活动招手关节 RMSE 0.0187 rad，通过切换检查。静止阶段手臂误差会自然变大，因此报告区分活动阶段和静止阶段。

3 N 专项微调权重 `greet-balance-push3-seed47/stand_greet.zip` 未通过回归，0/0.8/3 N 成功率为 56.7%/50.0%/53.3%，已保留用于诊断，网页继续加载通过 0.8 N 回归的 seed37 权重。

套壳校准脚本为 `scripts/calibrate_bluefish_shell.py`，验证 40 组随机姿态的
坐标旋转误差小于 1e-15，脚底显示映射偏差约 1.0/0.4 mm。该校准只针对 VRM
显示与仿真骨架之间的坐标关系，不代表实际机械尺寸已经测量。
