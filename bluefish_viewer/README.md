# 蓝色小肥鱼 VRM 外观预览

模型由 `../third_party/bluefatfish/BlueFatFish.vrm` 提供。仅限本机非商业预览；服务器绑定回环地址，不能从局域网访问，也不会把模型复制进 Docker 镜像。

```bash
cd Animatronic-Mini
python3 scripts/serve_bluefish_viewer.py
```

打开 <http://localhost:8766/>。浏览器需要能访问 jsDelivr 以加载固定版本的 Three.js 和 `@pixiv/three-vrm`；模型文件从本机读取。服务默认加载训练 venv 中的 MuJoCo PPO 回放，将实际关节状态、自由根部姿态、推力和表情通过本机 HTTP 轮询映射到 VRM。需要 Docker ROS 时使用 `--source ros`。

平衡、强扰动、站立招手和动作按钮驱动本地训练回放；使用 `--source ros` 时才转发到 `/chibi/motion`。开心与中性按钮只切换 VRM 表情，头发摆动由 VRM SpringBone 计算。

本地 PPO 仿真断开超过 2 秒时，页面明确提示未连接并保持最后姿态。`BlueFatFish.vrm` 是外观源，隐藏物理骨架是碰撞与训练源。

2026-11-17 验证：VRM 坐标旋转 40 组随机姿态通过，足底显示映射偏差约 1.0/0.4 mm；原 MuJoCo 六动作测试通过（1,417 个状态样本）。站立招手策略 0.8 N 下 30/30 通过，120 秒连续回放通过；3 N 下 26/30。
