# BlueFatFish 套壳架构

项目外观统一使用本机下载的 `third_party/bluefatfish/BlueFatFish.vrm`。
该文件保留原作者模型、材质、表情和头发物理，不会复制进 Docker 镜像，也不会提交到公开仓库。

MuJoCo 使用 `src/chibi_sim/mjcf/bluefish_physics.xml` 和
`bluefish_physics_free.xml` 作为隐藏物理骨架。骨架只包含关节、质量、执行器和
碰撞代理，不包含自建外观网格。训练和物理验证使用自由根部文件，网页运行时将
仿真状态映射到 VRM 人体骨骼。

## 本地查看

```bash
.venv-train/bin/python scripts/serve_bluefish_viewer.py --source training
```

打开 <http://127.0.0.1:8766/>。页面提供平衡、强扰动、站立招手和固定基座动作。
VRM 与物理骨架之间的映射校准记录在
`validation/reports/retarget_calibration.json`。

## 重新生成训练模型

```bash
.venv-train/bin/python src/chibi_sim/scripts/prepare_training_models.py
.venv-train/bin/python src/chibi_sim/scripts/validate_training_physics.py
```

物理参数仍是仿真假设，不能用于机械制造或电机选型。当前已完成自由站立招手，
尚未完成行走、跌倒起身和实物参数校准。
