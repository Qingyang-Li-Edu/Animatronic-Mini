# 蓝色大肥鱼：现成模型检索与接入

核查日期：2026-11-17。

## 选中的原作者发布

- 名称：【MZEM】蓝色小肥鱼 Deepseek 鲸鱼娘
- 作者：木子恶魔
- 原始页面：https://www.aplaybox.com/details/model/ayXkRzVflmnd
- 作者说明制作全流程由本人完成，页面标注 VRM。
- 下载包已核对：VRM 1.x、53 个标准人体骨骼条目、19 个表情、1 套蒙皮、6 个材质、8 张内嵌贴图、SpringBone 头发物理。无外部资源。
- 这是可直接显示的现成角色模型，比例和服装与参考毛绒图有差异。

## 作者页面的配布规则

作者页面禁止商用与二次配布，允许修改和用于视频以外的用途，并要求完整借物表。下载包还附带更细的个人使用条款：允许个人非商业 Avatar 使用；禁止重新分享文件、商用、拆取贴图/骨骼/网格；VRM 元数据同时标注 `allowRedistribution: false`、`avatarPermission: onlyAuthor`、`modification: prohibited`。元数据与页面规则对修改/Avatar 使用的表述存在冲突，本项目保守处理为：本机私用、保持 VRM 原文件不变、只做运行时姿势预览、页面署名，不打包进 Docker 镜像或公开分享。

未发现 Creative Commons 授权或 Blender 源文件；不把模型当作 MIT、CC 等开放许可资源。

## 文件与本地预览

下载包与模型保存在 `third_party/bluefatfish/`，目录被 `.gitignore` 和 `.dockerignore` 排除。原始授权文件保留为 `model-terms.txt`。

在本工作区根目录运行：

```bash
conda run -n chibi-sim python scripts/inspect_vrm.py third_party/bluefatfish/BlueFatFish.vrm
```

该脚本只读解析 GLB/VRM 容器头与 VRM 元数据；不会执行或转换模型。

## 套用方式

.venv-train/bin/python scripts/serve_bluefish_viewer.py --source training，打开 `http://127.0.0.1:8766/`。页面直接读取原始 VRM，保留蒙皮、纹理、表情和头发物理。默认服务运行本地 MuJoCo PPO 回放，将 24 个实时关节、自由根部姿态和推力映射到 VRM；需要 Docker ROS 时使用 `--source ros`。

仿真中断时显示未连接提示并保持最后姿态。当前是相同状态驱动的外观 Avatar，VRM 骨架比例与仿真连杆不完全相同，不能作为精确的接触或碰撞可视化。

2026-11-17：六种原 MuJoCo 动作测试通过；本地 HTTP 六动作和真实关节范围反馈测试全部通过。读取后的原始 VRM SHA-256 为 `c14879ff5c42501043896aae03ce971b2c11eee2e76106d47ee98741aa842ff5`，文件未被修改。

VRM 仍只是外观 Avatar，不是机械 CAD、碰撞体或可制造机器人结构。不要拆出其网格或纹理并用于制作另一模型，也不要公开分发 VRM 或包含 VRM 的镜像。
