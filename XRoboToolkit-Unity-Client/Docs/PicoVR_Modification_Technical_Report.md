# PicoVR Unity Client 技术报告

日期：2026-06-14

## 1. 项目概述

`XRoboToolkit-Unity-Client` 是一个基于 Unity 的 PICO/XR 客户端工程，用于在 PICO 设备上采集和发送头部、手柄、身体追踪等 XR 数据，并接收远端视频流进行显示。工程面向机器人遥操作场景，PICO 端作为操作者的交互终端，负责视觉反馈、追踪数据上报和部分运行状态提示。

本仓库主要包含：

- PICO/XR 追踪数据采集
- 远程相机视频接收和显示
- PICO 相机/视频相关配置
- Android 原生解码插件接入
- UI 控制面板、日志显示和连接管理
- PICO Unity Integration SDK 相关依赖

工程使用 Unity 2022.3 系列，主要运行目标是 Android/PICO。视频解码依赖 Android 原生层 `MediaDecoder`，Unity C# 层负责窗口、配置、网络控制、UDP relay 和 UI 状态展示。

## 2. 原始仓库能力

原始工程主要支持两类视频源：

```text
PICO4U
ZEDMINI
```

`PICO4U` 更偏向 PICO 设备自身或 Pico4U 相关视频流，`ZEDMINI` 面向双目/宽幅视频源。原始配置里没有专门针对当前实验使用的外部主摄像头，即大疆 Action / USB webcam / HoloTeleop camera_streamer 的 `WEBCAM` 视频源。

原始远程视频窗口包含：

- RawImage 视频显示
- 关闭按钮 `X`
- TCP 接收和原生解码流程
- 基础相机请求参数

原始工程没有针对当前 HoloTeleop 场景提供：

- 外部 `WEBCAM` 默认视频源
- HoloTeleop UDP `HTVF` 分片视频协议适配
- 视频刷新/重连按钮
- 机器人温度遥测显示
- 自动暂停/fail-safe 状态红点
- PICO 端显示当前接收 FPS

## 3. 当前系统链路

当前实验链路可以概括为：

```text
HoloTeleop / camera_streamer
        |
        |  H264 video over TCP/UDP
        v
PICO Unity Client
        |
        |  MediaDecoder
        v
PICO 视频窗口显示
```

同时机器人侧会通过 UDP 遥测发送温度和事件：

```text
HoloTeleop deploy.py
        |
        |  UDP JSON, default port 13601
        v
PICO Unity Client telemetry UI
```

PICO 端也会持续向 HoloTeleop 发送 XR 数据和控制器按钮。HoloTeleop 将 PICO 控制器按钮映射为：

```text
left_key_one  = left X
right_key_one = right A
```

其中：

```text
left X  = pause live teleop streaming
right A = start/resume live teleop streaming
```

## 4. 本次修改目标

本次修改的目标是让 PICO 端更贴合当前机器人遥操作实验流程：

- 让外部主摄像头画面作为默认视频源
- 降低视频码率、分辨率和帧率，减少延迟和解码压力
- 支持 HoloTeleop UDP 视频传输协议
- 提供视频刷新按钮，减少现场重连步骤
- 显示机器人最高温度和对应关节
- 在暂停或 fail-safe 时显示明确红点提示
- 保持 UI 不遮挡视频主体

## 5. 新增 WEBCAM 视频源

修改文件：

```text
Assets/StreamingAssets/video_source.yml
Assets/Scripts/Conf/VideoSourceConfigManager.cs
```

原工程没有专门的 `WEBCAM` 视频源。本次新增并适配 `WEBCAM`，用于接收 HoloTeleop / camera_streamer 发来的外部摄像头画面，例如大疆 Action 或 USB webcam。

当前 `WEBCAM` 配置为：

```yaml
- name: "WEBCAM"
  camera: "WEBCAM"
  description: "Generic webcam / D435i RGB source"
  properties:
    - name: "monoDisplay"
      value: true
    - name: "autoRectSize"
      value: true
    - name: "RawImageRectSize"
      value: "202.5x360"
    - name: "CamWidth"
      value: 540
    - name: "CamHeight"
      value: 960
    - name: "CamFPS"
      value: 25
    - name: "CamBitrate"
      value: 1500000
    - name: "VideoTransport"
      value: "AUTO"
```

配置含义：

- `camera: "WEBCAM"`：告诉发送端请求外部摄像头
- `monoDisplay: true`：按单目画面显示，不做 stereo split
- `autoRectSize: true`：根据视频比例计算显示框
- `540x960`：竖屏传输分辨率
- `25fps`：降低帧率以减少压力
- `1.5Mbps`：降低码率以减少网络和解码压力
- `RawImageRectSize: "202.5x360"`：保持小竖屏显示尺寸
- `AUTO`：允许 TCP/UDP 兼容

默认视频源也改为 `WEBCAM`：

```csharp
private const string DefaultVideoSourceName = "WEBCAM";
```

如果配置中找不到 `WEBCAM`，才回退到 `PICO4U`。

## 6. 视频参数优化

原先测试时使用过较高分辨率和码率，例如：

```text
720x1280 @ 30fps, 4Mbps
```

当前最终请求为：

```text
540x960 @ 25fps, 1.5Mbps
```

这样做的原因：

- 竖屏比例与当前主画面需求一致
- 分辨率等比缩小，降低编码和解码负担
- 显示尺寸保持 `202.5x360`，操作者看到的画面大小不变
- 码率降低后可以减少无线链路压力
- 25fps 对遥操作观察仍可用，同时降低帧处理频率

注意：如果发送端摄像头不支持 `540x960`，需要在 HoloTeleop 侧或 camera_streamer 侧选择接近的竖屏分辨率。

## 7. UDP 视频协议支持

修改文件：

```text
Assets/Scripts/Camera/RemoteCameraWindow.cs
```

PICO 端增加了对 HoloTeleop UDP 视频协议的支持。当前支持两类 UDP 帧：

1. HoloTeleop `HTVF` 分片协议
2. 旧格式：`4 字节大端 frame size + H264 frame`

`HTVF` 分片头结构：

```text
0..3    magic: "HTVF"
4       version
6..7    header_size
8..11   frame_id
12..13  frag_id
14..15  frag_count
16..19  frame_size
20..21  payload_size
24..    payload
```

接收策略：

- 只重组最新 `frame_id`
- 旧帧直接丢弃
- 超时帧丢弃
- 重组完成后转发给 Android 原生 `MediaDecoder`
- UDP-only 模式下使用本地 relay 端口连接解码器

这部分修改的核心目标是减少旧帧积压，避免网络波动后出现明显视频延迟。

## 8. 视频刷新/重连按钮

修改文件：

```text
Assets/Scripts/Camera/RemoteCameraWindow.cs
Assets/Scripts/UI/UICameraCtrl.cs
Assets/Resources/RemoteCamera.prefab
```

远程视频窗口原来只有关闭按钮 `X`。现场如果视频卡住，需要先关闭窗口再重新连接，操作步骤较多。

本次新增 `R` 刷新按钮：

```text
R = refresh / reconnect
X = close
```

当前按钮位置：

```text
R: x=150
X: x=210
```

两者不会重叠，并且 `R` 位于 `X` 左侧。

刷新流程：

1. 点击视频窗口右上角 `R`
2. 调用 `RemoteCameraWindow.OnRefreshBtn()`
3. 进入 `UICameraCtrl.ReconnectCameraStream()`
4. 停止当前解码、UDP relay 和本地接收
5. 向发送端发送关闭相机请求
6. 使用上一次 IP 和当前配置重新请求视频流

该功能减少了现场调试时的断流恢复成本。

## 9. 当前接收 FPS 显示

修改文件：

```text
Assets/Scripts/Camera/RemoteCameraWindow.cs
Assets/Scripts/UI/UICameraCtrl.cs
```

新增 `RemoteCameraWindow.CurrentVideoFps`。

该值表示：

- PICO 端实际解码并更新纹理的 FPS
- 不是发送端配置的 FPS
- 可以反映 PICO 端实际接收和解码状态

这对判断链路是否丢帧、解码是否跟不上比较有用。

## 10. 机器人温度遥测显示

修改文件：

```text
Assets/Scripts/UI/PicoTemperatureTelemetryDisplay.cs
Assets/Resources/RemoteCamera.prefab
```

PICO 端监听：

```text
UDP 13601
```

HoloTeleop 发送 `robot_thermal` JSON，例如：

```json
{
  "type": "robot_thermal",
  "version": 1,
  "temperature": {
    "top": [
      {
        "name": "left_knee_joint",
        "max": 82,
        "pair": [82, 80]
      }
    ]
  }
}
```

显示设计：

- 视频左侧显示温度数字
- 视频右侧显示对应关节短名称
- 最多显示 5 个最高温关节
- 不占用视频画面主体

颜色规则：

```text
< 80 C     绿色
80-100 C  黄色
> 100 C   红色
```

当最高温超过 `100 C` 时，PICO 端会触发短手柄震动。震动带冷却时间，避免持续打扰操作。

## 11. 暂停红点提示

修改文件：

```text
Assets/Scripts/UI/PicoPauseIndicatorDisplay.cs
Assets/Scripts/UI/PicoTemperatureTelemetryDisplay.cs
Assets/Resources/RemoteCamera.prefab
```

新增视频左上角红点，用于提示当前遥操作已暂停或进入 fail-safe pause。

红点位置：

```text
x = 视频左边缘 + 8
y = 视频上边缘 - 8
```

红点大小：

```text
14x14
```

红点位于视频框内左上角，不占用温度显示区域。

手动暂停信号与 HoloTeleop 最新代码对齐：

```text
left_key_one  = left X  = pause
right_key_one = right A = start/resume
```

PICO 本地按钮逻辑：

- 左手 X 按下：显示红点
- 右手 A 按下：隐藏红点

自动暂停 / fail-safe 对齐：

HoloTeleop 最新代码会通过 PICO telemetry UDP 发送：

```json
{
  "type": "robot_event",
  "event": "vr_fail_safe_pause",
  "action": "hold",
  "reason": "..."
}
```

PICO 端收到 `robot_event / vr_fail_safe_pause` 后显示红点。

实现上，`PicoTemperatureTelemetryDisplay` 会把 `robot_event` 转发给其他 UI 组件：

```csharp
public static event Action<string, string> RobotEventReceived;
```

`PicoPauseIndicatorDisplay` 订阅该事件，并处理 `vr_fail_safe_pause`。

## 12. 应用名称和默认 UI 状态

项目应用显示名已改为：

```text
PicoVR
```

包名保持测试包名，不覆盖官方版本。

默认 UI 状态调整：

- Head 默认勾选
- Controller 默认勾选
- Model 默认 Full Body
- 默认视频源偏向 `WEBCAM`

这些默认值减少了每次现场启动后的重复操作。

## 13. 已尝试但未保留的方案

曾尝试在 PICO 端通过解析 H264 SPS 自动识别横竖屏分辨率，并动态重建解码纹理。

最终没有保留该方案，原因是：

- TCP 直连原生解码器时，Unity C# 层无法稳定看到每帧 H264 数据
- 动态重建 `MediaDecoder` 对稳定性有风险
- 现场更需要固定、可控、低风险的视频参数

当前最终方案是：

```text
固定请求 540x960
固定显示 202.5x360
不做自动横竖屏切换
```

## 14. 安装和验证

安装前确认 PICO 已连接：

```bash
adb devices
```

安装 APK：

```bash
adb install -r "/Users/velix/Downloads/XRoboToolkit_Webcam.apk"
```

如果签名冲突：

```bash
adb uninstall com.xrobotoolkit.client.webcamtest
adb install "/Users/velix/Downloads/XRoboToolkit_Webcam.apk"
```

建议验证项目：

- 默认请求 `WEBCAM`
- 请求参数为 `540x960 @ 25fps, 1.5Mbps`
- 视频显示为小竖屏且不拉伸
- `R` 按钮可以刷新重连视频
- `X` 按钮可以关闭视频窗口
- 左手 X 后视频左上角出现红点
- 右手 A 后红点消失
- HoloTeleop 自动 fail-safe pause 后红点出现
- 温度最高 5 个关节显示在视频两侧
- 温度颜色和震动逻辑正常

## 15. 主要相关文件

```text
Assets/StreamingAssets/video_source.yml
Assets/Scripts/Conf/VideoSourceConfigManager.cs
Assets/Scripts/Camera/RemoteCameraWindow.cs
Assets/Scripts/UI/UICameraCtrl.cs
Assets/Scripts/UI/PicoTemperatureTelemetryDisplay.cs
Assets/Scripts/UI/PicoPauseIndicatorDisplay.cs
Assets/Resources/RemoteCamera.prefab
```

## 16. 后续建议

建议后续优化方向：

1. 在 PICO UI 中开放视频参数配置，例如分辨率、FPS、码率和 TCP/UDP。
2. 在 HoloTeleop 中增加明确的 `vr_start` / `vr_resume` telemetry event，使 PICO 端红点完全基于机器人侧状态闭环。
3. 将 UDP 重组和低延迟解码下沉到 Android 原生层，减少 C# 到本地 TCP relay 的额外拷贝。
4. 增加一键恢复默认显示尺寸，避免现场调试时显示框被误调。
5. 后续如果要支持多摄像头动态切换，建议由发送端显式发送分辨率和相机状态，而不是仅依赖接收端解析视频帧。
