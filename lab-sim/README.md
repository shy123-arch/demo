# G1 实验室 · MuJoCo

按照所给实验室照片搭建的可运行场景，包含左侧整面窗户、白色实验台与柜体、天花板灯格、显微镜、显示器、离心机、试剂瓶和实验椅。房间约 **6.8 × 8.6 m**，中央留出通道，G1 操作台高度 **0.80 m**。这是根据单张图片做的比例近似建模。

![MuJoCo 实际渲染](outputs/reference.png)

已集成宇树 **G1 29 自由度身体 + 14 自由度 Dex3 双手**，保留原始关节名称、惯量、限位与力矩范围。实验室和机器人资产均位于本目录，可独立使用，不需要复制其他项目的 Python 包。

| 任务 | 场景与接口 | 演示方式 |
|---|---|---|
| 桌面取放 | 自由运动样品瓶、样品盒，带底板与边缘碰撞的托盘 | 右臂 IK、PD 控制、靠近物体后启用辅助抓取约束 |
| 实验流程 | 滑轨抽屉、把手、带弹簧的仪器按钮、位置传感器 | 手臂接近、辅助把手力、开合抽屉、触发按钮 |
| 巡检 | 家具碰撞、机器人半径膨胀的 A* 路线、4 个巡检点、头部 RGB / 深度 | 辅助路线演示，以及 GR00T ONNX 驱动的无支撑物理巡检 |
| Psi0 | 与本机 `/mnt/workspace/Wilson/Psi0` 的 SIMPLE HTTP `/act` 协议对接 | 图像与状态请求、动作块接收、双手/双臂/躯干/导航命令映射 |
| ScaleBFM | 官方 M 模型的五点身体姿态输入 | 骨盆、双腕、双踝目标驱动全部 29 个身体关节；无底座支撑 |

取放和实验流程脚本是**辅助演示**：底座有支撑，取放可使用抓取 weld，抽屉与按钮可施加辅助力。物体受重力与碰撞影响。它们用于验证场景和接口，不能作为无辅助抓取或全身控制策略的成功率。用 `LabEnv(assisted=False)` 可关闭支撑；该模式需要全身控制器负责平衡。物理巡检和 ScaleBFM 身体控制入口均已配置相应策略。

## 启动

本工作区已经创建 `.venv` 并安装 MuJoCo、渲染等依赖。

```bash
cd /mnt/workspace/Wilson/lab-sim
.venv/bin/python run.py view
```

需要图形桌面。MuJoCo 窗口中鼠标可旋转、平移、缩放；`R` 重置，`1` 取放，`2` 抽屉与仪器，`3` 辅助巡检。也可以直接指定任务打开窗口：

```bash
.venv/bin/python run.py demo --task pick_place --viewer
```

服务器无桌面时，用软件 OpenGL 渲染：

```bash
MUJOCO_GL=osmesa .venv/bin/python run.py render
MUJOCO_GL=osmesa .venv/bin/python run.py demo --task all --video
```

输出在 `outputs/`，包括三个 MP4、每项任务的 JSON 结果和最终图片。相机名为 `reference`、`overview`、`workstation`、`top`、`head`、`left_wrist`、`right_wrist`。Linux 需有 `libOSMesa`；有正常 GPU 驱动时也可设置 `MUJOCO_GL=egl`。

无底座支撑的物理行走与完整巡检：

```bash
MUJOCO_GL=osmesa .venv/bin/python run.py walk --seconds 12 --video
MUJOCO_GL=osmesa .venv/bin/python run.py walk --seconds 65 --patrol
```

GR00T 策略以 50 Hz 产生关节目标，PD 在每个 500 Hz 物理步重新计算力矩。已完成一次 **60.54 秒、4 个巡检点**的无支撑闭环测试，骨盆最低高度 **0.740 m**，无摔倒、数值状态有限；记录见 `outputs/patrol_physics_report.json`。这是当前初始状态和固定布局下的验证，未做随机扰动鲁棒性评估。

## PICO 无支撑全身遥操作

**完整项目的安装、PICO 校准及双终端启动请优先阅读 [仓库根目录说明](../README.md)。**
默认读取同仓库的 `../molmospaces-teleop`，也兼容下文原始上传目录；可用
`--molmo-root` 或 `MOLMO_TELEOP_ROOT` 指定已有安装。

该入口直接复用相邻 `molospace` 项目中已经跑通的 PICO/GMR、UDP/ZMQ
传输和 29D ONNX motion-tracking policy。机器人底座没有支撑，29 个身体关节
以 50 Hz 接收策略目标、500 Hz 计算 PD 力矩；PICO trigger/grip 分别控制左右
Dex3 手。每次运行保存 `trajectory.npz` 和 `metadata.json`，视频从物理轨迹
离线生成，不占用实时控制循环。

先在终端 1 启动现有 PICO pose bridge：

```bash
cd /mnt/workspace/Wilson/ck/demo/molospace/molmospaces-teleop/molmospaces-teleop
scripts/run_pose_bridge.sh
```

再在终端 2 启动 lab-sim：

```bash
cd /mnt/workspace/Wilson/ck/demo/lab-sim
.venv/bin/python run.py pico-teleop --viewer
```

控制方式：

- 右手 A：开始或恢复 PICO 全身控制。
- 左手 X：停止接收新动作并安全保持。
- 右手 B：结束当前 demo，刷新并关闭录制文件。
- 左右 trigger/grip：连续控制对应 Dex3 手指闭合。
- 桌面窗口关闭或终端 `Ctrl-C`：同样安全保存当前 demo。

命令结束时会打印本次输出目录。离线生成四视角视频：

```bash
MUJOCO_GL=osmesa .venv/bin/python run.py pico-render \
  --output-dir outputs/pico_demo_YYYYMMDD_HHMMSS
```

默认四视角为第三人称、左/右双目相机和右腕相机，输出
`task.mp4`。如果提示 UDP 28704 已占用，请先关闭旧的 MolmoSpaces 或
lab-sim 遥操作进程。只测试控制器而不连接 PICO：

```bash
MUJOCO_GL=osmesa .venv/bin/python run.py pico-teleop \
  --offline-motion walk --seconds 2 --output-dir /tmp/lab-sim-pico-smoke
```

新机器安装：

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## ScaleBFM 五点身体控制

已集成 [ScaleBFM](https://github.com/zengweishuai/ScaleBFM) 官方
`humanoid_transformer_m/model_22200.pt` 原始权重。五点模式的编号是 **4**：
骨盆、左腕、右腕、左踝、右踝。五点是身体姿态目标，策略输出全部 29 个身体关节的控制目标；Dex3 的 14 个手指关节独立保持姿态。

直接运行 20 秒身体控制序列（站立 → 下蹲 → 右手前伸 → 双手前伸 → 骨盆侧移 → 站立）：

```bash
cd /mnt/workspace/Wilson/lab-sim
MUJOCO_GL=osmesa .venv/bin/python run.py scalebfm --video
```

有图形桌面时可用 `.venv/bin/python run.py scalebfm --viewer`。
命令目标由程序直接给出，不需要 Xsens、VR 设备或 Psi0 服务。
运行使用真实 MuJoCo 状态反馈，关闭底座支撑和所有抓取约束；策略以 50 Hz 推理，PD 力矩在每个 500 Hz 物理步重新计算。
该入口只运行本地仿真。

输出位于 `outputs/scalebfm_five_point/`：

- `five_point.mp4`：实际物理执行的视频；彩色点表示身体位置目标。
- `commands.json`：五点世界坐标姿态关键帧，可编辑后作为新的控制输入。
- `report.json`：完成状态、位置/姿态误差、骨盆高度、推理耗时。
- `trajectory.npz`：每个控制时刻的目标、实际五点姿态，以及完整 qpos/qvel。
- `pose_*.png`、`laboratory.png`：动作截图与实验室全景。

重新运行自定义目标：

```bash
MUJOCO_GL=osmesa .venv/bin/python run.py scalebfm \
  --targets outputs/scalebfm_five_point/commands.json \
  --output-dir outputs/scalebfm_custom --video
```

每个关键帧包含 `time`（秒）与 `poses`。五个键依次为 `pelvis`、
`left_wrist_yaw_link`、`right_wrist_yaw_link`、`left_ankle_roll_link`、
`right_ankle_roll_link`；每点给出 `position: [x,y,z]`（世界坐标，米）
和 `quaternion_wxyz: [w,x,y,z]`（单位四元数）。这些是关节链上的 body 原点，
不等同于手掌抓取 site。时间从 0 开始严格递增；位置使用平滑插值，姿态使用球面插值。
适配器按当前骨盆坐标系转换六个未来帧（0–0.10 秒），由官方 mode 4 掩码选出五点。

默认使用 CPU 原始 PyTorch 推理，可通过 `--device cuda:7` 指定 CUDA 设备。
无需加载针对 RTX 4090 编译的 TensorRT 引擎。
新环境额外安装 `.venv/bin/python -m pip install -r requirements-scalebfm.txt`。
权重、元数据、校验值及代码来源见 `assets/g1/policy/scalebfm_m/PROVENANCE.md`。

基础身体姿态演示采用固定站位：双脚保持目标姿态，身体下蹲和双臂运动。
已完成一次 **20 秒、1000 个策略步**的闭环执行，全部 equality 约束关闭，
骨盆最低高度 0.689 m，无摔倒。五点位置 RMSE 分别为：骨盆 **3.28 cm**，
左腕 **2.01 cm**，右腕 **1.80 cm**，左踝 **0.81 cm**，右踝 **0.76 cm**。
CPU 推理中位耗时 **10.7 ms**；20 秒视频已完整解码检查。
这些数据对应 `outputs/scalebfm_five_point/report.json` 中的这一次运行。
现有 4 项测试与新增 3 项 ScaleBFM 检查均通过，包括全局平移/航向变换一致性、
无支撑站立、姿态插值边界与无效命令拒绝。

### ScaleBFM 实际接触任务

本小节的旧版任务使用仿真根部位姿和预先编写的世界坐标目标，属于特权状态控制。
下方新增的相机实验使用独立的传感器接口，结果分别保存。

已用相同官方权重执行三个独立任务，全部 29 个身体关节由五点策略控制，
Dex3 手指保持张开。没有启用底座支撑、抓取 weld、物体助力 actuator 或外加力。
成功判定同时要求：完成动作序列、保持平衡、真实手部接触，以及对象达到目标状态。

| 任务 | 判据与实测结果 | 动作时长 |
|---|---|---|
| 按仪器按钮 | 食指接触，按钮行程 **14.04 mm**（触发阈值 8 mm），撤手后弹回 | 16 s |
| 平推样品盒 | 沿台面前移 **57.5 mm**，目标 60 mm；最终位置距目标 **5.7 mm**，保持平放 | 20 s |
| 关闭抽屉 | 抽屉初始打开 **180 mm**，双手食指推至完全关闭并撤手 | 14 s |

三项最低骨盆高度均约 **0.754 m**，未摔倒。按钮与推盒试验分别出现一次
台面擦碰区间，峰值接触力约 6.97 N / 4.12 N；接触记录保留在报告中。
推盒初次试验在撤手时使盒子倾斜，后续采用先水平撤手、再抬高的路径得到上述结果。
这些是固定布局下经过调整的演示，不是随机化评测成功率。

每项试验在第一个物理步之前将机器人初始化到对应工位；关抽屉任务同时初始化
抽屉开度为 0.18 m。执行中只下发五点姿态与手指关节目标。
本组结果验证原地物理交互，不包含自主走到工位、开抽屉或无辅助抓取。

```bash
MUJOCO_GL=osmesa LP_NUM_THREADS=2 .venv/bin/python run.py scalebfm-tasks --task all --video
```

单独运行可选择 `--task button`、`--task push_box`、`--task close_drawer`。
用 `--render-only` 可以直接从已保存的物理状态渲染视频，不重新执行策略。
五点关键帧位于 `assets/scalebfm_tasks/`；输出位于
`outputs/scalebfm_tasks/validated/`，包含汇总 `summary.json` 和各任务目录：

- `plan.json`：初始站位、抽屉初始开度、五点关键帧及手指目标。
- `report.json`：任务判据结果、对象变化、500 Hz 物理步上的接触力/冲量记录与助力审计。
- `trajectory.npz`：50 Hz 的完整 qpos/qvel、命令目标与对象状态。
- `task.mp4`、`interaction.png`、`final.png`：从该物理轨迹渲染的过程与结果，视频标注对象的实际位移。

### 相机环境感知 + 编码器 / IMU 的 ScaleBFM 实验

已实际运行桌前按按钮、推盒子的无辅助物理交互。环境输入仅为安装在躯干上的
左右 **640×480 RGB** 图像，基线 65 mm；视觉通过颜色、轮廓和双目视差定位
蓝色圆按钮、青色盒盖及白色盒侧面。身体输入为 29 关节编码器位置/速度与
骨盆原生陀螺仪；IMU 姿态由陀螺仪在 500 Hz 积分得到，从直立状态启动。
没有把仿真物体坐标、深度/分割缓冲、真实根部平移、接触或成功标志输入控制器。
相机标定、机器人自身运动学和指尖尺寸属于已知硬件参数。

`scalebfm_proprio.py` 通过编码器 FK、IMU 和双脚固定假设估计局部身体位置，
官方 ScaleBFM M 的模式 4 在 50 Hz 输出全部 29 个身体关节目标，PD 在 500 Hz 执行。
`vision_servo.py` 在动作边界读取 RGB，接触阶段约每秒观察一次；推盒时根据新图像
修正指尖与盒侧面的相对误差。按钮采用视觉对准后的小步有界前推，并检查可见位移与回弹。
这不是逐帧连续视觉控制，也未训练通用视觉模型。

最终控制版本的实测结果如下。数值来自**回合结束后的独立评测**，不回流到控制器。

| 试验 | 视觉估计 | 独立物理结果 |
|---|---|---|
| 原位按钮 | 行程 11.1 mm，观察到回弹 | 行程 **11.46 mm**，触发并释放；13.7 s |
| 按钮横移 25 mm，偏移不提供给控制器 | 初始目标随图像横移 25.2 mm；行程 8.1 mm | 行程 **9.20 mm**，触发并释放 |
| 原位推盒 | 最终前移 46.3 mm | 台面位移 **50.49 mm**，保持平放；17.9 s |
| 盒子横移 20 mm，偏移不提供给控制器 | 最终前移 39.1 mm，未确认成功 | 台面位移只有 **27.35 mm**，未达到 40 mm 判据 |
| 相机遮黑 | 未检测到目标，保持站立 | 未伸手，按钮无行程 |

原位推盒的首版仅移动 22.5 mm，随后加入图像反馈修正得到上述 50.49 mm。
初版视觉完成门槛为 35 mm，导致横移盒子试验被误判成功；审查后统一为 40 mm，
复跑正确标记该次失败。复跑的动作与物理轨迹和此前完全一致，说明改动仅影响最终结果标志。
这些开发试验与有限扰动对照不能解释为随机化成功率。尤其盒子的双目轮廓定位存在
厘米级抖动和偏差，当前 `visual_success` 是启发式判断，不能代替真实任务判据。

全部试验未启用支撑、weld、物体辅助 actuator 或外加力，最低骨盆高度约 0.754 m。
原位按钮存在中指与台面的擦碰（峰值约 9 N）；所有非目标接触保留在报告中。
初始工位在首个物理步之前设置；不包含走到工位、抓取或抽屉的视觉控制。
双脚共同滑移无法被当前估计器观测；尚未验证真实相机误差、传感器噪声、延迟或实机迁移。

```bash
MUJOCO_GL=osmesa LP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m lab_sim.vision_tasks --task button --output-dir outputs/vision_tasks/reproduce_button
MUJOCO_GL=osmesa LP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m lab_sim.vision_tasks --task push_box --output-dir outputs/vision_tasks/reproduce_box
MUJOCO_GL=osmesa LP_NUM_THREADS=2 \
  .venv/bin/python -m lab_sim.vision_render --output-dir outputs/vision_tasks/reproduce_button
```

录像默认从 `trajectory.npz` 的时间戳推断原始记录帧率。当前轨迹每 20 ms 记录一次，
因此输出为 **50 FPS**；外部机器人观察视角与左右相机视角使用同一时刻的状态逐帧同步渲染。

对照参数：`--blind`、`--button-shift 0 0.025 0`、`--box-shift 0 -0.02`。
三个本体控制测试通过：坐标平移/航向独立性、独立机器人 FK 对照、编码器下蹲与历史动作。
图像模块完成 15 项像素输入检查，包括平移、变暗、目标缺失及歧义拒绝。

完整结果位于 `outputs/vision_tasks/summary.json`。主录像为
`button_final/task.mp4`、`box_final/task.mp4`，扰动失败录像为 `box_shifted_checked/task.mp4`。
各目录中的 `visual_decisions.json`、原始左右 RGB、`controller_trace.npz` 属于控制侧日志；
`evaluator.json`、`trajectory.npz` 属于独立评测。调用 `VisionRobotIO.evaluate()` 后
该回合拒绝继续执行 `step()`。录像的三个视角均从已记录的物理状态离线重建，不重新推进物理。
左右相机画面标注为离线重建，并显示最近一次实际控制观测的时间；这些逐帧重建画面没有输入控制器。
实际控制仍在动作边界读取 RGB，接触阶段约每秒一次，原始稀疏观测保留在各任务目录中。
外部观察视角和评测数字也从未输入控制器。

### 新增无辅助抓取、搬放和移动试验（2026-09-09）

身体继续使用官方 ScaleBFM M 五点模式，Dex3 手指使用普通限矩 PD。
抓取目标来自双目 RGB 的蓝瓶/白色标签检测，尺寸由图像和视差估计；
手指闭合姿态由机器人几何和独立手部夹具校准得到。夹具校准使用受支撑手腕，
不计入下表的全身成功结果。全身试验没有底座支撑、抓取 weld、物体助力或外加力。

| 任务 | 独立物理结果 | 判定 |
|---|---|---|
| 抓瓶抬起 | 瓶底离台最高 97.97 mm，连续无非手支撑夹持 2.078 s | 通过；抬起时视觉受手部遮挡，控制器未独立确认 |
| 抓取、横向搬放 | 空中搬移 90.97 mm，最终位移 81.61 mm，释放后倾角 0.11° | 物理及视觉通过 |
| 瓶子隐藏横移 10 mm 后搬放 | 空中搬移 91.25 mm，最终位移 81.86 mm | 物理及视觉通过 |
| 机器人前进 | 30 cm 目标，实际 25.08 cm，无碰撞并停稳；双脚均抬离地面 | 通过本轮距离容差 ±7.5 cm 及步态判据 |
| 机器人侧移 | 实际 37.20 cm，右脚未达到离地判据，定位误差明显 | 未通过 |

黑图抓取对照停在 `no_visual_target_hold`，没有伸手或闭合抓取。
抓取使用预设工位，视觉仍在动作段之间更新。抬手后的视角倾斜可能造成漏检，
接近阶段会短暂使用此前的 RGB 目标；闭合后存在真实手部遮挡。
搬放流程在空中转移后重新看见瓶子，才依据图像确认抬起和最终位置。

移动试验是**本体感知驱动的相对步态命令**，RGB 仅作观察记录，不是视觉自主导航，
也未把走到工位与抓取串成一个任务。平移由原生加速度计/陀螺仪积分，并在命令双支撑阶段
用编码器 FK 估计的速度纠偏；高度仍依赖支撑脚假设。没有读取自由底座世界位置或接触真值来控制。
这是短时仿真验证，尚未验证实机 IMU 偏置、脚底滑动或长距离累计误差。
早期大步幅试验曾前进过量并碰到凳子，完整失败记录保留，未计为成功。

抓取评分在 500 Hz 物理步累计瓶底离台、手指接触和非手支撑；
移动的脚底离地轨迹在事后按 50 Hz 记录审计。真值仅用于事后评分、录像和离线开发调参。
成功标准不是泛化成功率：抓取离台 ≥30 mm、连续无非手支撑夹持 ≥0.5 s；
搬放还要求持握及最终平移均 ≥80 mm，释放并直立落台；移动要求目标容差内、无家具碰撞，
每只脚前移超过目标一半且脚底高于地面 5 mm 的累计时间 ≥80 ms。

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m lab_sim.grasp_tasks --task pick_place --output-dir outputs/grasp_tasks/reproduce
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m lab_sim.scalebfm_mobility --output-dir outputs/scalebfm_mobility/reproduce \
  --length .15 --pairs 2 --sway .07 --lift .10 --swing-seconds .8 --inertial-xy
MUJOCO_GL=egl .venv/bin/python -m lab_sim.vision_render --output-dir outputs/grasp_tasks/reproduce
```

主结果位于 `outputs/grasp_tasks/{lift_v3,place_final,place_shifted}` 和
`outputs/scalebfm_mobility/{forward_final,side_final}`。三视角录像统一为 50 FPS。
汇总、代码快照、完整视频校验与压缩包位于 `outputs/grasp_mobility_20260909/`。

## 接入 Psi0

适配依据是 `Psi0/src/psi/deploy/psi0_serve_simple.py` 和 `SIMPLE/src/simple/baselines/psi0.py`，使用 **SIMPLE HTTP 协议**，不连接实机 SDK。

先验证相机、序列化和关节映射，无需加载模型：

```bash
MUJOCO_GL=osmesa .venv/bin/python run.py psi0 --dry-run
```

生成 `outputs/psi0_head.png` 和可直接审查的 `outputs/psi0_request.json`。该检查不运行 Psi0 模型推理。

准备好对应的 SIMPLE 格式 checkpoint 后，在 Psi0 环境启动原仓库服务（替换 run-dir 与 step）：

```bash
cd /mnt/workspace/Wilson/Psi0
PYTHONPATH=src .venv-psi/bin/python -m psi.deploy.psi0_serve_simple \
  --host 127.0.0.1 --port 22085 --policy psi0 \
  --run-dir /path/to/run --ckpt-step 40000 --device cuda:0
```

再运行实验室客户端：

```bash
cd /mnt/workspace/Wilson/lab-sim
MUJOCO_GL=osmesa .venv/bin/python run.py psi0 \
  --url http://127.0.0.1:22085/act \
  --instruction "Pick up the blue sample bottle and place it in the tray." \
  --chunks 10 --video
```

| 协议字段 | 约定 |
|---|---|
| `image.rgb_head_stereo_left` | RGB `uint8`，`480 × 640 × 3`，G1 头部相机 |
| `state.states` | `float32 (1,32)`：双手 14 + 双臂 14 + 上次命令 torso roll/pitch/yaw/height 4 |
| `action[:,0:14]` | 双手绝对关节角；左手 thumb/middle/index，右手 thumb/index/middle |
| `action[:,14:28]` | 左右臂各 7 个绝对关节角 |
| `action[:,28:32]` | torso roll/pitch/yaw/height |
| `action[:,32:36]` | vx、vy、turning flag、target yaw |

按名称映射关节，不依赖 MuJoCo 树的排列顺序。客户端不做归一化；服务端使用 checkpoint 对应的统计量。动作频率为 30 Hz，物理时间步为 0.002 s。客户端默认保留底座辅助，不自动启用物体抓取 weld；接入全身控制可向 `Psi0Adapter` 传入实现 `step(env, command, ticks)` 的控制器。`turning_flag` 保留在命令字典中，辅助底座模式按 target yaw 旋转，不复现训练时的 turning flag 语义。

使用已集成的 GR00T 下层控制器时，在 Psi0 命令后加 `--controller gr00t`。这个模式完全关闭底座支撑，使用目标航向调节器生成 ONNX 所需的 yaw-rate 命令；torso rpyh、速度有明确限幅，具体见 `lab_sim/walking.py`。不同 checkpoint 使用的下层控制约定可能不同，应先确认其训练配置。

**尚未使用真实 Psi0 checkpoint 完成闭环任务验证**：目前提供的是可运行的协议与控制适配。实际实验室任务表现取决于所选 checkpoint 和任务数据，通用/其他任务权重不保证完成新实验流程。

## 场景与控制代码

- `scenes/lab_g1.xml`：完整场景，可由 MuJoCo 直接加载。
- `scenes/lab_empty.xml`：实验室和交互物体，不含机器人。
- `scripts/build_room.py`：房间、家具、仪器生成器。
- `lab_sim/props.py`：物体、抽屉、按钮、传感器。
- `scripts/build_scene.py`：合成完整 MJCF、配置碰撞和相机。
- `lab_sim/env.py`：reset / step、命名关节、RGB / 深度、IK、PD 和辅助抓取。
- `lab_sim/tasks.py`、`navigation.py`：三类示例与路线规划。
- `lab_sim/psi0.py`：Psi0 协议和动作映射。
- `lab_sim/locomotion.py`：GR00T G1 Balance / Walk ONNX 适配器；权重与许可位于 `assets/g1/policy/`。
- `lab_sim/walking.py`：高频 PD、物理巡检与 Psi0 下层控制桥接。
- `lab_sim/scalebfm.py`：官方 ScaleBFM M 权重加载、五点世界坐标目标、历史观测与全身 PD。
- `lab_sim/scalebfm_demo.py`：姿态关键帧命令、执行记录与视频。
- `lab_sim/scalebfm_tasks.py`：按钮、推盒、关抽屉的实际接触验证及物理轨迹回放。

修改生成器后重新构建：

```bash
.venv/bin/python scripts/build_room.py
.venv/bin/python scripts/build_scene.py
.venv/bin/python -m unittest discover -s tests -v
```

最小控制接口：

```python
from lab_sim.env import LabEnv

env = LabEnv(assisted=False)
obs = env.reset()
# 外部策略每次返回 43 个机器人执行器力矩，顺序见 obs['actuator_names']。
# obs = env.step(action=torques, nstep=10)
rgb = env.render("head", 640, 480)
depth_m = env.render("head", 640, 480, depth=True)
env.close()
```

装饰仪器使用简化碰撞体；样品瓶中的液体是视觉几何，没有流体仿真。原始场景中的实验椅固定在地面；独立的 `lab_g1_chair.xml` 场景增加了下述可移动椅子。实验室结构以基础几何搭建，保留可调尺寸与较低仿真开销。

## RGB 引导的边走边推椅子（2026-09-09）

`outputs/chair_tasks/walk_v7/` 在官方 ScaleBFM M 五点控制下通过单次仿真评测：
椅子前进 **32.60 cm**，机器人骨盆前进 **22.66 cm**；左右脚分别前进
**21.21 / 22.59 cm**，最大整脚底离地 **11.54 / 14.18 mm**。
左右脚离地、手部接触与椅子前移的同时发生时长分别为 **0.116 / 0.120 s**。
结束时双手已松开，机器人与椅子稳定；没有非目标碰撞、底座支撑、焊接、椅子驱动器或外加力。

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m lab_sim.chair_tasks --output-dir outputs/chair_tasks/reproduce
MUJOCO_GL=egl .venv/bin/python -m lab_sim.vision_render \
  --output-dir outputs/chair_tasks/reproduce
```

默认参数对应成功试验：机器人预设站位 `(-0.08, -2.0)`，每轮步长命令 0.15 m、
两轮交替摆脚，骨盆侧移命令 0.12 m、摆脚抬高命令 0.20 m、脚尖上抬目标 15°、
顶部保持 0.3 s。命令幅度与实际执行有明显差距，不能把 20 cm 命令当成实际抬脚高度。
控制器以 RGB 检测椅背正面上沿，在两脚支撑且侧向居中时更新视觉，编码器与 IMU 用于机器人状态估计。
视觉按动作段更新；视频三个视角为同一记录状态的 **50 FPS 离线回放**，并非 50 Hz 视觉闭环。

椅子质量 10 kg，底座完全自由，五个被动球形脚轮带非零轴承阻力和接触摩擦。
这是万向脚轮近似，未模拟真实脚轮叉架转向和拖距。详细机械标定见
`outputs/chair_tasks/scene_mechanics/MECHANICS.md`；标定中的外力探针只用于离线机械检查。
`lab_sim/chair_walking_evaluation.py` 在控制结束后才读取轨迹及接触真值，
联合检查实际迈步、手部推动和椅子运动的时间重叠。

这是近距离预设起点的低抬脚短距离演示，未验证自主走到椅子、复杂障碍、真实硬件或随机初始位置的可靠性。
`walk_v1` 至 `walk_v6` 的失败记录保留。黑相机试验 `blind_hold` 未启动接触或行走；
椅背识别的遮挡、歧义与亮度反例记录在 `outputs/chair_vision/validation.json`。

机器人资产来源及许可见 `assets/g1/NOTICE.md`。参考照片仅作为用户提供的建模参考。
