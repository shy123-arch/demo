# PICO 遥操 lab-sim：无支撑 G1 全身控制与 demo 录制

本仓库将 MolmoSpaces 的 PICO/GMR 动作流及 29D ONNX tracking policy 接入 lab-sim 实验室。
G1 使用自由基座、无底座支撑，通过全身关节力矩跟踪动作；左右 trigger/grip 控制 Dex3。
每次启动立即记录 lab-sim 的物理状态，结束后可生成四视角 MP4。

## 仓库内容

```text
lab-sim/                       实验室场景、G1/Dex3、控制适配器、录制和渲染
molmospaces-teleop/             pose bridge、tracking runtime、ONNX及外部数据、动作与配置
XRoboToolkit-Unity-Client/      上传项目中的 PICO Unity 客户端及 SDK
third_party/GMR/               GMR 源码与 G1 重定向资产
third_party/XRoboToolkit-PC-Service/        PC Service 源码和 SDK
third_party/XRoboToolkit-PC-Service-Pybind/ callback-enabled Python binding 源码
scripts/                       安装和两个启动入口
```

依赖来源和本地修改见 [THIRD_PARTY.md](THIRD_PARTY.md)。核心模型、场景和策略都在仓库内；
首次使用还需安装 Python/系统依赖和头显客户端。无需原电脑上的绝对路径或 MolmoSpaces 场景数据集。
以下启动说明针对 Linux x86_64，推荐 Ubuntu 22.04/24.04、Python 3.10–3.12。

## 1. 安装电脑端依赖

```bash
git clone https://github.com/shy123-arch/demo.git
cd demo
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev build-essential cmake libosmesa6 libgl1 libglfw3
bash scripts/setup_pico.sh
```

安装脚本分别建立 `lab-sim/.venv` 和 `.venv-bridge`，安装 GMR，并编译带
`register_frame_callback/clear_frame_callback/has_frame_callback` 的 XRoboToolkit binding。
编译与安装需要网络下载 Python 包；全身姿态重定向的默认 IK solver 是 `daqp`。
若你已有之前跑通的 GMR/SDK 环境，可用 `BRIDGE_PYTHON=/你的环境/bin/python` 启动 bridge。

## 2. 连接与校准 PICO

1. 安装并启动 [XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service/releases)。
   PC Service 是独立应用，安装 Python binding 不会自动启动它；源码也位于本仓库 `third_party/`。
2. 使用你原来已经跑通的 PICO 客户端；需要重建时按
   [Unity 客户端说明](XRoboToolkit-Unity-Client/README_cn.md) 打开项目、选择 Android 并部署到头显。
3. 头显连接运行 PC Service 的电脑 IP；确保局域网可达，PC Service 的 TCP 63901 可访问。
4. 开启 **Full Body Tracking**，连接并校准双腿的 PICO Swift trackers，确认 PC Service 能看到
   24 个身体关节姿态。只有头显和双手位置、没有有效全身追踪数据时，不能提供本方案要求的腿部行走参考。

默认将 PC Service、pose bridge 和 lab-sim 运行在同一台可显示 MuJoCo 窗口的电脑上。
云服务器终端本身不能直接显示头显/桌面窗口，需要桌面或远程桌面环境。

## 3. 启动遥操并录制

两个终端均在 clone 后的 `demo` 根目录执行。

终端 1：启动 lab-sim 接收器、物理控制和录制：

```bash
bash scripts/run_lab_pico.sh --viewer
```

终端 2：启动 PICO → GMR bridge，将身高改为操作者实际值（米）：

```bash
ACTUAL_HUMAN_HEIGHT=1.70 bash scripts/run_pico_bridge.sh
```

确认 bridge 收到身体数据，再在初始自然站姿按右 A。保持仿真窗口实时运行。

| 控件 | lab-sim 行为 |
|---|---|
| 右 A | 开始/恢复全身动作跟踪，执行启动对齐 |
| 左 X | 停止接收新动作并进入策略 hold；自由基座仍参与物理仿真 |
| 左右 trigger/grip | 对应 Dex3 手的闭合控制 |
| 右 B | 结束当前 demo，保存轨迹与元数据 |
| Ctrl-C / 关闭仿真窗口 | 保存并结束当前 demo |

录制从仿真启动开始，包含等待与暂停阶段。再次录制重新运行终端 1；bridge 可以继续运行。
如果看到上游旧日志“B skips next scene”，在本入口以本表为准：B 结束 lab-sim demo。
断流触发上游 hold；hold 并不锁定机器人基座，也不保证任意姿态不会摔倒。
骨盆低于 0.45 m 或状态异常时自动停止并保存。请在实验室空余区域先完成站立、抬手和小步动作。

默认输出到 `lab-sim/outputs/pico_demo_日期_时间/`，终端会打印完整位置。也可指定：

```bash
bash scripts/run_lab_pico.sh --viewer --output-dir outputs/my_demo_001
```

每段使用新的目录名，避免覆盖已有文件。

## 4. 从录制轨迹生成多视角视频

```bash
cd lab-sim
MUJOCO_GL=osmesa .venv/bin/python run.py pico-render --output-dir outputs/my_demo_001
```

默认生成 `task.mp4`：第三人称、左眼、右眼、右腕四宫格，25 FPS。渲染读取已记录的
qpos/qvel，不重新运行控制器，所以不会改变已录制动作。实时控制为 50 Hz 策略、500 Hz 物理。
如希望结束录制后自动渲染，启动时加 `--render-after`。

- `trajectory.npz`：时间戳、完整 qpos/qvel、身体目标与力矩、参考姿态、手指目标、按键、stream 状态和物体状态。
- `metadata.json`：场景、策略路径、无支撑状态、帧数、停止原因、最低骨盆高度和位移等。
- `task.mp4`：离线生成的四视角视频。也可用 `--camera reference` 等选择单一相机。

当前采用 lab-sim 原生录制方案，没有转换为 HDF5。

## 常见启动问题

- `No module named yaml/zmq`：重新安装 `lab-sim/requirements.txt`。
- SDK 缺少 callback API：运行 `scripts/setup_pico.sh` 编译本仓库 binding；官方原版 Python binding 不含这些接口。
- UDP 28704 被占用：关闭旧的 MolmoSpaces/lab-sim 接收进程。
- 一直等待数据：检查 PC Service、头显 IP、Full Body Tracking 和 trackers 校准；在 bridge 有数据后按右 A。
- 没有图形桌面：去掉 `--viewer` 可录制，结束后用 `MUJOCO_GL=osmesa` 离线渲染。
- 两台电脑：bridge 端设 `UDP_STREAM_HOST=仿真电脑IP`；主控制数据在 UDP 28704 中。
  ZMQ 28703 是额外按键通道，跨机使用需同时调整 `molmospaces-teleop/configs/motion_tracking_live.yaml` 的 `vr_ctrl_addr`。

## 已验证范围

本机已运行聚焦适配/录制测试；8 秒离线 walk 参考录制得到 400 帧，
`assisted=false`、`equality_active=false`，骨盆最低约 0.712 m，平面位移约 0.130 m，状态有限且未摔倒。
PICO 接收端 UDP/ZMQ 初始化、NPZ 保存与 MP4 渲染亦已检查。
随仓库的 PC SDK 和 callback binding 已在本机编译并完成 Python 导入/注册检查，GMR 的 PICO 重定向 worker 也已成功初始化。
这些检查没有使用真实头显输入；实际头显遥操的动作质量、连续走动和场景交互仍需现场确认。
GMR 和 callback binding 的来源与原电脑安装的区别已在 THIRD_PARTY.md 明示。
