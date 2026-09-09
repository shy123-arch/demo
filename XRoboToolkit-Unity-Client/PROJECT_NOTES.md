# PicoVR Unity Client Notes

This repository is a Unity 2022.3.16f1c1 project for the PicoVR webcam/remote vision client.

## Current App

- Product name: `PicoVR`
- Test package name: `com.xrobotoolkit.client.webcamtest`
- Main scene: `Assets/Main.unity`
- Video source config: `Assets/StreamingAssets/video_source.yml`
- PICO temperature telemetry UDP port: `13601`

## Build

1. Open the project with Unity `2022.3.16f1c1`.
2. Open `File -> Build Settings`.
3. Select Android and include `Assets/Main.unity`.
4. Configure signing in `Project Settings -> Player -> Publishing Settings`.
5. Build APK.

## Install

```bash
adb install -r "/path/to/PicoVR.apk"
```

If Android reports a signature conflict:

```bash
adb uninstall com.xrobotoolkit.client.webcamtest
adb install "/path/to/PicoVR.apk"
```

## Do Not Commit

The following are local/generated files and should stay out of Git:

- `Library/`
- `Temp/`
- `Logs/`
- `UserSettings/`
- `local-android-sdk/`
- `*.apk`
- `*.keystore`
- `.DS_Store`
