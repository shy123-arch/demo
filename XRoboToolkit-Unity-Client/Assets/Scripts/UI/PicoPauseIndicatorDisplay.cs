using System;
using LitJson;
using Robot;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR;

public class PicoPauseIndicatorDisplay : MonoBehaviour
{
    private const float DotSize = 14f;
    private const float DotMargin = 8f;
    private const XRNode HoloTeleopPauseHand = XRNode.LeftHand;   // HoloTeleop left_key_one = left X
    private const XRNode HoloTeleopResumeHand = XRNode.RightHand; // HoloTeleop right_key_one = right A

    private RemoteCameraWindow _remoteCameraWindow;
    private RectTransform _dotRect;
    private Image _dotImage;
    private bool _paused;
    private bool _prevPauseButton;
    private bool _prevResumeButton;
    private static Sprite _dotSprite;

    private void Awake()
    {
        _remoteCameraWindow = GetComponent<RemoteCameraWindow>();
        EnsureDot();
        SetPaused(false);
    }

    private void OnEnable()
    {
        TcpHandler.ReceiveFunctionEvent += OnFunctionMessage;
        PicoTemperatureTelemetryDisplay.RobotEventReceived += OnRobotEventTelemetry;
        EnsureDot();
        SetPaused(_paused);
    }

    private void OnDisable()
    {
        TcpHandler.ReceiveFunctionEvent -= OnFunctionMessage;
        PicoTemperatureTelemetryDisplay.RobotEventReceived -= OnRobotEventTelemetry;
    }

    private void Update()
    {
        UpdateFromHoloTeleopControlButtons();
    }

    private void LateUpdate()
    {
        PositionDot();
    }

    public void SetPaused(bool paused)
    {
        _paused = paused;
        EnsureDot();
        if (_dotRect != null)
            _dotRect.gameObject.SetActive(_paused);
    }

    private void UpdateFromHoloTeleopControlButtons()
    {
        var pauseButton = ReadPrimaryButton(HoloTeleopPauseHand);
        var resumeButton = ReadPrimaryButton(HoloTeleopResumeHand);

        if (pauseButton && !_prevPauseButton)
            SetPaused(true);

        if (resumeButton && !_prevResumeButton)
            SetPaused(false);

        _prevPauseButton = pauseButton;
        _prevResumeButton = resumeButton;
    }

    private static bool ReadPrimaryButton(XRNode node)
    {
        var device = InputDevices.GetDeviceAtXRNode(node);
        return device.isValid
               && device.TryGetFeatureValue(CommonUsages.primaryButton, out var pressed)
               && pressed;
    }

    private void OnFunctionMessage(string functionName, string value)
    {
        if (!IsPauseFunction(functionName))
            return;

        if (TryReadPaused(value, out var paused))
            SetPaused(paused);
    }

    private void OnRobotEventTelemetry(string eventName, string payload)
    {
        if (string.Equals(eventName, "vr_fail_safe_pause", StringComparison.OrdinalIgnoreCase))
        {
            SetPaused(true);
            return;
        }

        if (string.Equals(eventName, "vr_start", StringComparison.OrdinalIgnoreCase)
            || string.Equals(eventName, "vr_resume", StringComparison.OrdinalIgnoreCase)
            || string.Equals(eventName, "vr_active", StringComparison.OrdinalIgnoreCase))
        {
            SetPaused(false);
        }
    }

    private static bool IsPauseFunction(string functionName)
    {
        if (string.IsNullOrEmpty(functionName))
            return false;

        return functionName.Equals("TeleopPaused", StringComparison.OrdinalIgnoreCase)
               || functionName.Equals("RobotPaused", StringComparison.OrdinalIgnoreCase)
               || functionName.Equals("PauseState", StringComparison.OrdinalIgnoreCase)
               || functionName.Equals("VRState", StringComparison.OrdinalIgnoreCase)
               || functionName.Equals("TeleopState", StringComparison.OrdinalIgnoreCase)
               || functionName.Equals("RobotState", StringComparison.OrdinalIgnoreCase);
    }

    private static bool TryReadPaused(string value, out bool paused)
    {
        paused = false;
        if (string.IsNullOrEmpty(value))
            return false;

        var normalized = value.Trim().Trim('"').ToLowerInvariant();
        if (bool.TryParse(normalized, out paused))
            return true;

        if (int.TryParse(normalized, out var intValue))
        {
            paused = intValue != 0;
            return true;
        }

        if (IsPausedWord(normalized))
        {
            paused = true;
            return true;
        }

        if (IsActiveWord(normalized))
        {
            paused = false;
            return true;
        }

        try
        {
            var json = JsonMapper.ToObject(value.Replace("\\", ""));
            if (TryReadBoolKey(json, out paused, "paused", "pause", "isPaused", "vrPaused", "teleopPaused"))
                return true;

            if (TryReadBoolKey(json, out var active, "active", "enabled", "vrActive", "vr_enabled", "teleopActive"))
            {
                paused = !active;
                return true;
            }

            if (TryReadStringKey(json, out var state, "state", "status", "mode"))
            {
                state = state.ToLowerInvariant();
                if (IsPausedWord(state))
                {
                    paused = true;
                    return true;
                }

                if (IsActiveWord(state))
                {
                    paused = false;
                    return true;
                }
            }
        }
        catch
        {
            return false;
        }

        return false;
    }

    private static bool TryReadBoolKey(JsonData json, out bool value, params string[] keys)
    {
        value = false;
        if (json == null || !json.IsObject)
            return false;

        foreach (var key in keys)
        {
            if (!json.ContainsKey(key))
                continue;

            if (bool.TryParse(json[key].ToString(), out value))
                return true;

            if (int.TryParse(json[key].ToString(), out var intValue))
            {
                value = intValue != 0;
                return true;
            }
        }

        return false;
    }

    private static bool TryReadStringKey(JsonData json, out string value, params string[] keys)
    {
        value = null;
        if (json == null || !json.IsObject)
            return false;

        foreach (var key in keys)
        {
            if (!json.ContainsKey(key))
                continue;

            value = json[key].ToString();
            return true;
        }

        return false;
    }

    private static bool IsPausedWord(string value)
    {
        return value.Contains("pause")
               || value.Contains("stop")
               || value.Contains("hold")
               || value.Contains("disable");
    }

    private static bool IsActiveWord(string value)
    {
        return value.Contains("run")
               || value.Contains("start")
               || value.Contains("active")
               || value.Contains("enable");
    }

    private void EnsureDot()
    {
        if (_dotRect != null)
            return;

        var image = GetRemoteCameraImage();
        if (image == null)
            return;

        var parent = image.rectTransform.parent as RectTransform;
        if (parent == null)
            return;

        var dot = new GameObject("PicoPauseIndicator", typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        dot.transform.SetParent(parent, false);
        _dotRect = dot.GetComponent<RectTransform>();
        _dotRect.sizeDelta = new Vector2(DotSize, DotSize);
        _dotRect.pivot = new Vector2(0.5f, 0.5f);

        _dotImage = dot.GetComponent<Image>();
        _dotImage.raycastTarget = false;
        _dotImage.sprite = GetDotSprite();
        _dotImage.color = new Color32(255, 30, 30, 255);
        dot.SetActive(false);
        PositionDot();
    }

    private void PositionDot()
    {
        if (_dotRect == null)
            return;

        var image = GetRemoteCameraImage();
        if (image == null)
            return;

        var imageRect = image.rectTransform;
        _dotRect.anchorMin = imageRect.anchorMin;
        _dotRect.anchorMax = imageRect.anchorMax;

        var videoWidth = imageRect.sizeDelta.x > 0f ? imageRect.sizeDelta.x : imageRect.rect.width;
        var videoHeight = imageRect.sizeDelta.y > 0f ? imageRect.sizeDelta.y : imageRect.rect.height;
        var leftEdgeX = imageRect.anchoredPosition.x - videoWidth * imageRect.pivot.x;
        var topEdgeY = imageRect.anchoredPosition.y + videoHeight * (1f - imageRect.pivot.y);

        _dotRect.anchoredPosition = new Vector2(leftEdgeX + DotMargin, topEdgeY - DotMargin);
    }

    private RawImage GetRemoteCameraImage()
    {
        if (_remoteCameraWindow == null)
            _remoteCameraWindow = GetComponent<RemoteCameraWindow>();

        return _remoteCameraWindow != null ? _remoteCameraWindow.RemoteCameraImage : null;
    }

    private static Sprite GetDotSprite()
    {
        if (_dotSprite != null)
            return _dotSprite;

        const int size = 32;
        var texture = new Texture2D(size, size, TextureFormat.RGBA32, false);
        texture.wrapMode = TextureWrapMode.Clamp;
        var center = (size - 1) * 0.5f;
        var radius = center - 1f;
        for (var y = 0; y < size; y++)
        {
            for (var x = 0; x < size; x++)
            {
                var dx = x - center;
                var dy = y - center;
                var distance = Mathf.Sqrt(dx * dx + dy * dy);
                var alpha = Mathf.Clamp01(radius + 1f - distance);
                texture.SetPixel(x, y, new Color(1f, 1f, 1f, alpha));
            }
        }

        texture.Apply();
        _dotSprite = Sprite.Create(texture, new Rect(0f, 0f, size, size), new Vector2(0.5f, 0.5f));
        return _dotSprite;
    }
}
