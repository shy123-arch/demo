using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using LitJson;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR;

public class PicoTemperatureTelemetryDisplay : MonoBehaviour
{
    private const int DefaultTelemetryPort = 13601;
    private const int TopTemperatureCount = 5;
    private const float TelemetryStaleSeconds = 2.5f;
    private const float TemperaturePanelWidth = 64f;
    private const float LabelPanelWidth = 128f;
    private const float PanelHeight = 138f;
    private const float PanelMargin = 6f;
    private const float HapticWarningInterval = 1.5f;
    private const float HapticWarningAmplitude = 0.65f;
    private const float HapticWarningDuration = 0.12f;

    public static event Action<string, string> RobotEventReceived;

    private readonly object _payloadLock = new object();
    private readonly List<MotorTemperature> _temperatures = new List<MotorTemperature>();

    private RemoteCameraWindow _remoteCameraWindow;
    private RectTransform _panelRect;
    private Image _panelBackground;
    private Text _text;
    private RectTransform _labelPanelRect;
    private Image _labelPanelBackground;
    private Text _labelText;
    private UdpClient _udpClient;
    private Thread _receiveThread;
    private volatile bool _running;
    private string _pendingPayload;
    private float _lastTelemetryTime = -1f;
    private float _lastHapticWarningTime = -10f;
    private int _maxTemperature;

    private void Awake()
    {
        _remoteCameraWindow = GetComponent<RemoteCameraWindow>();
        EnsurePanel();
        SetWaitingText();
    }

    private void OnEnable()
    {
        EnsurePanel();
        StartReceiver();
    }

    private void OnDisable()
    {
        StopReceiver();
    }

    private void OnDestroy()
    {
        StopReceiver();
    }

    private void Update()
    {
        ConsumeLatestPayload();
        RefreshStaleState();
    }

    private void LateUpdate()
    {
        PositionPanel();
    }

    private void StartReceiver()
    {
        if (_running)
            return;

        _running = true;
        _receiveThread = new Thread(ReceiveLoop) { IsBackground = true };
        _receiveThread.Start();
    }

    private void StopReceiver()
    {
        _running = false;

        if (_udpClient != null)
        {
            _udpClient.Close();
            _udpClient = null;
        }

        _receiveThread = null;
    }

    private void ReceiveLoop()
    {
        try
        {
            _udpClient = new UdpClient(DefaultTelemetryPort);
            _udpClient.Client.ReceiveTimeout = 250;
            var endpoint = new IPEndPoint(IPAddress.Any, 0);

            while (_running)
            {
                try
                {
                    var data = _udpClient.Receive(ref endpoint);
                    var payload = Encoding.UTF8.GetString(data);
                    lock (_payloadLock)
                    {
                        _pendingPayload = payload;
                    }
                }
                catch (SocketException)
                {
                }
                catch (ObjectDisposedException)
                {
                    break;
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning("PICO temperature telemetry stopped: " + ex.Message);
        }
    }

    private void ConsumeLatestPayload()
    {
        string payload = null;
        lock (_payloadLock)
        {
            if (!string.IsNullOrEmpty(_pendingPayload))
            {
                payload = _pendingPayload;
                _pendingPayload = null;
            }
        }

        if (string.IsNullOrEmpty(payload))
            return;

        PublishRobotEventIfPresent(payload);

        if (TryParseTelemetry(payload))
        {
            _lastTelemetryTime = Time.realtimeSinceStartup;
            UpdateDisplay();
        }
    }

    private static void PublishRobotEventIfPresent(string payload)
    {
        try
        {
            var root = JsonMapper.ToObject(payload);
            if (!HasKey(root, "type") || root["type"].ToString() != "robot_event")
                return;

            if (!HasKey(root, "event"))
                return;

            RobotEventReceived?.Invoke(root["event"].ToString(), payload);
        }
        catch (Exception ex)
        {
            Debug.LogWarning("Invalid PICO robot event telemetry: " + ex.Message);
        }
    }

    private bool TryParseTelemetry(string payload)
    {
        try
        {
            var root = JsonMapper.ToObject(payload);
            if (!HasKey(root, "type") || root["type"].ToString() != "robot_thermal")
                return false;

            if (!HasKey(root, "temperature"))
                return false;

            var temperature = root["temperature"];
            _temperatures.Clear();
            _maxTemperature = HasKey(temperature, "max") ? ReadInt(temperature["max"], 0) : 0;

            if (HasKey(temperature, "top") && temperature["top"].IsArray)
            {
                var top = temperature["top"];
                for (var i = 0; i < top.Count && i < TopTemperatureCount; i++)
                {
                    var item = top[i];
                    var jointName = HasKey(item, "name") ? ShortJointName(item["name"].ToString()) : "J" + i;
                    var value = HasKey(item, "max") ? ReadInt(item["max"], 0) : ReadMaxFromPair(item);
                    _temperatures.Add(new MotorTemperature(jointName, value));
                    if (value > _maxTemperature)
                        _maxTemperature = value;
                }
            }
            else if (HasKey(temperature, "joints") && temperature["joints"].IsArray)
            {
                var joints = temperature["joints"];
                for (var i = 0; i < joints.Count; i++)
                {
                    var joint = joints[i];
                    var jointName = HasKey(joint, "name") ? ShortJointName(joint["name"].ToString()) : "J" + i;
                    if (!HasKey(joint, "pair") || !joint["pair"].IsArray)
                        continue;

                    var pair = joint["pair"];
                    for (var sensor = 0; sensor < pair.Count; sensor++)
                    {
                        var value = ReadInt(pair[sensor], 0);
                        _temperatures.Add(new MotorTemperature(jointName, value));
                        if (value > _maxTemperature)
                            _maxTemperature = value;
                    }
                }
            }

            if (_temperatures.Count == 0 && _maxTemperature > 0)
            {
                var jointName = HasKey(temperature, "joint") ? ShortJointName(temperature["joint"].ToString()) : "Max";
                _temperatures.Add(new MotorTemperature(jointName, _maxTemperature));
            }

            _temperatures.Sort((a, b) => b.temperature.CompareTo(a.temperature));
            return _temperatures.Count > 0;
        }
        catch (Exception ex)
        {
            Debug.LogWarning("Invalid PICO temperature telemetry: " + ex.Message);
            return false;
        }
    }

    private void UpdateDisplay()
    {
        EnsurePanel();
        if (_text == null || _labelText == null)
            return;

        var text = new StringBuilder();
        var labelText = new StringBuilder();
        var count = Mathf.Min(TopTemperatureCount, _temperatures.Count);
        for (var i = 0; i < count; i++)
        {
            var item = _temperatures[i];
            var color = GetTemperatureColorHex(item.temperature);
            text.Append("<color=");
            text.Append(color);
            text.Append(">");
            text.Append(item.temperature);
            text.Append("C</color>");

            labelText.Append("<color=");
            labelText.Append(color);
            labelText.Append(">");
            labelText.Append(item.name);
            labelText.Append("</color>");
            if (i < count - 1)
            {
                text.AppendLine();
                labelText.AppendLine();
            }
        }

        _text.text = text.ToString();
        _labelText.text = labelText.ToString();
        _text.color = Color.white;
        _labelText.color = Color.white;
        _panelBackground.color = GetBackgroundColor(_maxTemperature);
        _labelPanelBackground.color = GetBackgroundColor(_maxTemperature);
        _panelRect.gameObject.SetActive(true);
        _labelPanelRect.gameObject.SetActive(true);

        if (_maxTemperature > 100)
            PulseHapticsIfNeeded();
    }

    private void RefreshStaleState()
    {
        if (_lastTelemetryTime < 0f)
            return;

        if (Time.realtimeSinceStartup - _lastTelemetryTime > TelemetryStaleSeconds)
        {
            SetWaitingText();
            _lastTelemetryTime = -1f;
        }
    }

    private void SetWaitingText()
    {
        EnsurePanel();
        if (_text == null || _labelText == null)
            return;

        _text.text = "--\n--\n--\n--\n--";
        _labelText.text = "";
        _text.color = new Color32(180, 190, 200, 255);
        _labelText.color = new Color32(180, 190, 200, 255);
        _panelBackground.color = Color.clear;
        _labelPanelBackground.color = Color.clear;
        _panelRect.gameObject.SetActive(true);
        _labelPanelRect.gameObject.SetActive(true);
    }

    private void EnsurePanel()
    {
        if (_panelRect != null && _labelPanelRect != null)
            return;

        var image = GetRemoteCameraImage();
        if (image == null)
            return;

        var parent = image.rectTransform.parent as RectTransform;
        if (parent == null)
            return;

        CreateTextPanel(parent, "PicoTemperatureValues", TemperaturePanelWidth, TextAnchor.MiddleRight,
            out _panelRect, out _panelBackground, out _text);
        _panelRect.pivot = new Vector2(1f, 0.5f);

        CreateTextPanel(parent, "PicoTemperatureLabels", LabelPanelWidth, TextAnchor.MiddleLeft,
            out _labelPanelRect, out _labelPanelBackground, out _labelText);
        _labelPanelRect.pivot = new Vector2(0f, 0.5f);

        PositionPanel();
    }

    private static void CreateTextPanel(RectTransform parent, string name, float width, TextAnchor alignment,
        out RectTransform panelRect, out Image background, out Text text)
    {
        var panel = new GameObject(name, typeof(RectTransform), typeof(CanvasRenderer), typeof(Image));
        panel.transform.SetParent(parent, false);
        panelRect = panel.GetComponent<RectTransform>();
        panelRect.sizeDelta = new Vector2(width, PanelHeight);

        background = panel.GetComponent<Image>();
        background.raycastTarget = false;
        background.color = Color.clear;

        var textObj = new GameObject("Text", typeof(RectTransform), typeof(CanvasRenderer), typeof(Text));
        textObj.transform.SetParent(panel.transform, false);
        var textRect = textObj.GetComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.offsetMin = Vector2.zero;
        textRect.offsetMax = Vector2.zero;

        text = textObj.GetComponent<Text>();
        text.raycastTarget = false;
        text.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        text.fontSize = 18;
        text.alignment = alignment;
        text.horizontalOverflow = HorizontalWrapMode.Overflow;
        text.verticalOverflow = VerticalWrapMode.Truncate;
        text.supportRichText = true;
        text.color = Color.white;
    }

    private void PositionPanel()
    {
        if (_panelRect == null || _labelPanelRect == null)
            return;

        var image = GetRemoteCameraImage();
        if (image == null)
            return;

        var imageRect = image.rectTransform;
        _panelRect.anchorMin = imageRect.anchorMin;
        _panelRect.anchorMax = imageRect.anchorMax;
        _labelPanelRect.anchorMin = imageRect.anchorMin;
        _labelPanelRect.anchorMax = imageRect.anchorMax;

        var videoWidth = imageRect.sizeDelta.x;
        if (videoWidth <= 0f)
            videoWidth = imageRect.rect.width;

        var leftEdgeX = imageRect.anchoredPosition.x - videoWidth * imageRect.pivot.x;
        var rightEdgeX = leftEdgeX + videoWidth;
        _panelRect.anchoredPosition = new Vector2(leftEdgeX - PanelMargin, imageRect.anchoredPosition.y);
        _labelPanelRect.anchoredPosition = new Vector2(rightEdgeX + PanelMargin, imageRect.anchoredPosition.y);
    }

    private RawImage GetRemoteCameraImage()
    {
        if (_remoteCameraWindow == null)
            _remoteCameraWindow = GetComponent<RemoteCameraWindow>();

        return _remoteCameraWindow != null ? _remoteCameraWindow.RemoteCameraImage : null;
    }

    private static bool HasKey(JsonData data, string key)
    {
        return data != null && data.IsObject && data.ContainsKey(key);
    }

    private static int ReadInt(JsonData data, int fallback)
    {
        if (data == null)
            return fallback;

        if (int.TryParse(data.ToString(), out var value))
            return value;

        return fallback;
    }

    private static int ReadMaxFromPair(JsonData item)
    {
        if (!HasKey(item, "pair") || !item["pair"].IsArray)
            return 0;

        var pair = item["pair"];
        var max = 0;
        for (var i = 0; i < pair.Count; i++)
        {
            var value = ReadInt(pair[i], 0);
            if (value > max)
                max = value;
        }

        return max;
    }

    private static string ShortJointName(string jointName)
    {
        if (string.IsNullOrEmpty(jointName))
            return "J";

        var name = jointName.ToLowerInvariant()
            .Replace("_joint", "")
            .Replace("joint", "")
            .Replace("_pitch", "")
            .Replace("_roll", "")
            .Replace("_yaw", "")
            .Replace("_", " ")
            .Trim();

        var side = "";
        if (name.StartsWith("left "))
        {
            side = "L ";
            name = name.Substring(5);
        }
        else if (name.StartsWith("right "))
        {
            side = "R ";
            name = name.Substring(6);
        }

        if (name.Contains("knee"))
            return side + "knee";

        if (name.Contains("waist") || name.Contains("torso") || name.Contains("trunk"))
            return "waist";

        if (name.Contains("hip"))
            return side + "hip";

        if (name.Contains("ankle"))
            return side + "ankle";

        if (name.Contains("shoulder"))
            return side + "shld";

        if (name.Contains("elbow"))
            return side + "elbow";

        if (name.Contains("wrist"))
            return side + "wrist";

        var parts = name.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length == 0)
            return side.Trim();

        var shortName = parts[0].Length > 5 ? parts[0].Substring(0, 5) : parts[0];
        return side + shortName;
    }

    private static string GetTemperatureColorHex(int temperature)
    {
        if (temperature > 100)
            return "#ff5f50";

        if (temperature >= 80)
            return "#ffdc46";

        return "#8dff9a";
    }

    private static Color GetBackgroundColor(int temperature)
    {
        return Color.clear;
    }

    private void PulseHapticsIfNeeded()
    {
        if (Time.realtimeSinceStartup - _lastHapticWarningTime < HapticWarningInterval)
            return;

        _lastHapticWarningTime = Time.realtimeSinceStartup;
        SendHapticImpulse(InputDeviceCharacteristics.Left);
        SendHapticImpulse(InputDeviceCharacteristics.Right);
    }

    private static void SendHapticImpulse(InputDeviceCharacteristics hand)
    {
        var devices = new List<InputDevice>();
        InputDevices.GetDevicesWithCharacteristics(
            InputDeviceCharacteristics.Controller | InputDeviceCharacteristics.HeldInHand | hand,
            devices);

        for (var i = 0; i < devices.Count; i++)
        {
            var device = devices[i];
            if (device.isValid && device.TryGetHapticCapabilities(out var capabilities) && capabilities.supportsImpulse)
                device.SendHapticImpulse(0, HapticWarningAmplitude, HapticWarningDuration);
        }
    }

    private struct MotorTemperature
    {
        public readonly string name;
        public readonly int temperature;

        public MotorTemperature(string name, int temperature)
        {
            this.name = name;
            this.temperature = temperature;
        }
    }
}
