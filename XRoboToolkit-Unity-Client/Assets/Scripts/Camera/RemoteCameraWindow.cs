using System;
using System.Collections;
using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using LitJson;
using Network;
using Robot;
using UnityEngine.UI;


/// <summary>
/// Display window of PC camera
/// Responsible for receiving, decoding, and displaying data
/// </summary>
public class RemoteCameraWindow : MonoBehaviour
{
    private const int MaxDecoderTextureUpdatesPerFrame = 4;
    private const int HoloTeleopUdpHeaderSize = 24;
    private const int HoloTeleopUdpFragmentPayloadSize = 1176; // HoloTeleop default: CAMERA_UDP_MTU 1200 - 24 byte header.
    private const int HoloTeleopUdpFrameTimeoutMs = 80;
    private const byte HoloTeleopUdpVersion = 1;

    public RawImage RemoteCameraImage;
    private TcpListener _tcpListener;
    private TcpClient _client;
    private NetworkStream _stream;
    private Texture2D _texture;
    public Texture2D Texture => _texture;
    private byte[] _imageBuffer;
    private CancellationTokenSource _receiveImageTs = null;
    private Task _imageReceiveTask;

    private int _resolutionWidth = 2160;
    private int _resolutionHeight = 2160 / 2 * 4 / 3;
    private int _videoFps = 60;
    private int _bitrate = 40 * 1024 * 1024;
    private string _transport = "TCP";
    private int _receivedFrameCount;
    private float _lastFpsSampleTime;
    private CancellationTokenSource _udpRelayCts;
    private Task _udpRelayTask;
    private UdpClient _udpRelayClient;
    private TcpClient _decoderRelayClient;
    private NetworkStream _decoderRelayStream;

    public static float CurrentVideoFps { get; private set; }

    public CustomButton listenBtn;

    private void Awake()
    {
        transform.position = Camera.main.transform.position;
        transform.rotation = Camera.main.transform.rotation;
    }

    public void StartListen(int width, int height, int fps, int bitrate, int port, string transport = "TCP")
    {
        _resolutionWidth = width;
        _resolutionHeight = height;
        _videoFps = fps;
        _bitrate = bitrate;
        _transport = NormalizeTransport(transport);
        _receivedFrameCount = 0;
        _lastFpsSampleTime = Time.realtimeSinceStartup;
        CurrentVideoFps = 0f;

        StartCoroutine(OnStartListen(port));
    }

    private void OnDisable()
    {
        StopListening();
        Debug.Log("RemoteCameraWindow OnDisable");
    }

    public void OnCloseBtn()
    {
        // Reset listen button
        listenBtn.SetOn(false);
        // send close event to server
        NetworkCommander.Instance.CloseCamera();
        gameObject.SetActive(false);
    }

    public void OnRefreshBtn()
    {
        var cameraCtrl = FindObjectOfType<UICameraCtrl>();
        if (cameraCtrl != null)
        {
            cameraCtrl.ReconnectCameraStream();
            return;
        }

        StopListening();
        StartListen(_resolutionWidth, _resolutionHeight, _videoFps, _bitrate, 12345, _transport);
    }

    public void StopListening()
    {
        CurrentVideoFps = 0f;
        StopUdpRelay();
        MediaDecoder.release();
        TcpHandler.SendFunctionValue("StopReceivePcCamera", "");
    }

    public IEnumerator OnStartListen(int port)
    {
        Debug.Log("StartListen port:" + port);

        _texture = new Texture2D(_resolutionWidth, _resolutionHeight, TextureFormat.RGB24, false, false);
        RemoteCameraImage.texture = _texture;
        yield return null;

        StopUdpRelay();
        var decoderPort = UseUdpOnlyTransport ? GetUdpRelayDecoderPort(port) : port;
        MediaDecoder.initialize((int)_texture.GetNativeTexturePtr(), _resolutionWidth, _resolutionHeight);
        MediaDecoder.startServer(decoderPort, false);
        if (UseUdpRelay)
            StartUdpRelay(port, decoderPort);
        yield return null;

        JsonData cameraParam = new JsonData();
        cameraParam["ip"] = Utils.GetLocalIPv4();
        cameraParam["port"] = port;
        cameraParam["width"] = _resolutionWidth;
        cameraParam["height"] = _resolutionHeight;
        cameraParam["fps"] = _videoFps;
        cameraParam["bitrate"] = _bitrate;
        cameraParam["transport"] = _transport;
        cameraParam["lowLatency"] = true;
        TcpHandler.SendFunctionValue("StartReceivePcCamera", cameraParam.ToJson());
    }

    private void LateUpdate()
    {
        //Keep the window facing the camera at all times
        if (Camera.main != null)
        {
            transform.position = Camera.main.transform.position;
            transform.rotation = Camera.main.transform.rotation;
        }
    }

    private void Update()
    {
        if (_texture != null && Application.platform == RuntimePlatform.Android)
        {
            var updateCount = 0;
            while (updateCount < MaxDecoderTextureUpdatesPerFrame && IsDecoderUpdateFrame())
            {
                UpdateDecoderTexture();
                UpdateVideoFps();
                updateCount++;
            }

            if (updateCount > 0)
                GL.InvalidateState();
        }
    }

    private bool UseUdpOnlyTransport => _transport.Equals("UDP", StringComparison.OrdinalIgnoreCase);
    private bool UseUdpRelay => UseUdpOnlyTransport || _transport.Equals("AUTO", StringComparison.OrdinalIgnoreCase);

    private bool IsDecoderUpdateFrame()
    {
        return MediaDecoder.isUpdateFrame();
    }

    private void UpdateDecoderTexture()
    {
        MediaDecoder.updateTexture();
    }

    private static string NormalizeTransport(string transport)
    {
        if (string.Equals(transport, "UDP", StringComparison.OrdinalIgnoreCase))
            return "UDP";

        if (string.Equals(transport, "AUTO", StringComparison.OrdinalIgnoreCase))
            return "AUTO";

        return "TCP";
    }

    private void UpdateVideoFps()
    {
        _receivedFrameCount++;
        var now = Time.realtimeSinceStartup;
        var elapsed = now - _lastFpsSampleTime;
        if (elapsed < 1f)
            return;

        CurrentVideoFps = _receivedFrameCount / elapsed;
        _receivedFrameCount = 0;
        _lastFpsSampleTime = now;
    }

    private static int GetUdpRelayDecoderPort(int udpPort)
    {
        return udpPort >= 65534 ? udpPort - 1 : udpPort + 1;
    }

    private void StartUdpRelay(int udpPort, int decoderPort)
    {
        StopUdpRelay();
        _udpRelayCts = new CancellationTokenSource();
        _udpRelayTask = Task.Run(() => UdpRelayLoop(udpPort, decoderPort, _udpRelayCts.Token));
    }

    private void StopUdpRelay()
    {
        if (_udpRelayCts != null)
        {
            _udpRelayCts.Cancel();
            _udpRelayCts.Dispose();
            _udpRelayCts = null;
        }

        CloseUdpRelaySockets();
        _udpRelayTask = null;
    }

    private void CloseUdpRelaySockets()
    {
        if (_udpRelayClient != null)
        {
            _udpRelayClient.Close();
            _udpRelayClient = null;
        }

        if (_decoderRelayStream != null)
        {
            _decoderRelayStream.Close();
            _decoderRelayStream = null;
        }

        if (_decoderRelayClient != null)
        {
            _decoderRelayClient.Close();
            _decoderRelayClient = null;
        }
    }

    private void UdpRelayLoop(int udpPort, int decoderPort, CancellationToken token)
    {
        HoloTeleopUdpFrameAssembly assembly = null;
        var latestFrameId = 0u;
        var hasLatestFrame = false;
        var droppedFrameId = 0u;
        var hasDroppedFrame = false;
        var assemblyStartTime = 0;

        try
        {
            _udpRelayClient = new UdpClient(udpPort);
            _udpRelayClient.Client.ReceiveTimeout = 100;
            var remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);

            while (!token.IsCancellationRequested)
            {
                byte[] packet;
                try
                {
                    packet = _udpRelayClient.Receive(ref remoteEndPoint);
                }
                catch (SocketException)
                {
                    continue;
                }
                catch (ObjectDisposedException)
                {
                    break;
                }

                if (assembly != null && IsFrameAssemblyTimedOut(assemblyStartTime))
                {
                    droppedFrameId = assembly.FrameId;
                    hasDroppedFrame = true;
                    assembly = null;
                }

                if (TryReadLegacyUdpFrame(packet, out var legacyFrame))
                {
                    WriteFrameToDecoder(decoderPort, legacyFrame, token);
                    assembly = null;
                    continue;
                }

                if (!TryReadHoloTeleopUdpFragment(packet, out var fragment))
                    continue;

                if (hasLatestFrame && !IsFrameIdNewerOrSame(fragment.FrameId, latestFrameId))
                    continue;

                if (hasDroppedFrame && fragment.FrameId == droppedFrameId)
                    continue;

                if (assembly == null || assembly.FrameId != fragment.FrameId)
                {
                    if (!hasLatestFrame || fragment.FrameId != latestFrameId)
                        hasDroppedFrame = false;

                    latestFrameId = fragment.FrameId;
                    hasLatestFrame = true;
                    assembly = new HoloTeleopUdpFrameAssembly(fragment.FrameId, fragment.FrameSize, fragment.FragmentCount);
                    assemblyStartTime = Environment.TickCount;
                }

                if (!assembly.TryAdd(fragment))
                    continue;

                if (assembly.IsComplete)
                {
                    WriteFrameToDecoder(decoderPort, assembly.Frame, token);
                    droppedFrameId = assembly.FrameId;
                    hasDroppedFrame = true;
                    assembly = null;
                }
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning("UDP video relay stopped: " + ex.Message);
        }
        finally
        {
            CloseUdpRelaySockets();
        }
    }

    private static TcpClient ConnectToLocalDecoder(int port, CancellationToken token)
    {
        for (var i = 0; i < 50 && !token.IsCancellationRequested; i++)
        {
            try
            {
                var client = new TcpClient();
                client.NoDelay = true;
                client.Connect(IPAddress.Loopback, port);
                return client;
            }
            catch (SocketException)
            {
                Thread.Sleep(20);
            }
        }

        Debug.LogWarning("UDP video relay could not connect to local decoder on port " + port);
        return null;
    }

    private static bool TryReadLegacyUdpFrame(byte[] packet, out byte[] frame)
    {
        frame = null;
        if (packet == null || packet.Length < 5 || IsHoloTeleopUdpPacket(packet))
            return false;

        var frameSize = ReadInt32BE(packet, 0);
        if (frameSize <= 0 || frameSize != packet.Length - 4)
            return false;

        frame = new byte[frameSize];
        Buffer.BlockCopy(packet, 4, frame, 0, frameSize);
        return true;
    }

    private static bool TryReadHoloTeleopUdpFragment(byte[] packet, out HoloTeleopUdpFragment fragment)
    {
        fragment = default(HoloTeleopUdpFragment);
        if (packet == null || packet.Length < HoloTeleopUdpHeaderSize || !IsHoloTeleopUdpPacket(packet))
            return false;

        var version = packet[4];
        var headerSize = ReadUInt16BE(packet, 6);
        if (version != HoloTeleopUdpVersion || headerSize != HoloTeleopUdpHeaderSize)
            return false;

        var payloadSize = ReadUInt16BE(packet, 20);
        if (payloadSize <= 0 || HoloTeleopUdpHeaderSize + payloadSize > packet.Length)
            return false;

        fragment = new HoloTeleopUdpFragment
        {
            FrameId = ReadUInt32BE(packet, 8),
            FragmentId = ReadUInt16BE(packet, 12),
            FragmentCount = ReadUInt16BE(packet, 14),
            FrameSize = ReadInt32BE(packet, 16),
            PayloadSize = payloadSize,
            Packet = packet
        };

        return fragment.FragmentCount > 0
               && fragment.FragmentId < fragment.FragmentCount
               && fragment.FrameSize > 0;
    }

    private static bool IsHoloTeleopUdpPacket(byte[] packet)
    {
        return packet.Length >= 4
               && packet[0] == (byte)'H'
               && packet[1] == (byte)'T'
               && packet[2] == (byte)'V'
               && packet[3] == (byte)'F';
    }

    private void WriteFrameToDecoder(int decoderPort, byte[] frame, CancellationToken token)
    {
        if (frame == null || frame.Length == 0)
            return;

        if (_decoderRelayStream == null)
        {
            _decoderRelayClient = ConnectToLocalDecoder(decoderPort, token);
            if (_decoderRelayClient == null)
                return;

            _decoderRelayStream = _decoderRelayClient.GetStream();
        }

        var header = new byte[4];
        WriteUInt32BE(header, 0, frame.Length);
        _decoderRelayStream.Write(header, 0, header.Length);
        _decoderRelayStream.Write(frame, 0, frame.Length);
        _decoderRelayStream.Flush();
    }

    private static int ReadUInt16BE(byte[] data, int offset)
    {
        return (data[offset] << 8) | data[offset + 1];
    }

    private static int ReadInt32BE(byte[] data, int offset)
    {
        return (data[offset] << 24)
               | (data[offset + 1] << 16)
               | (data[offset + 2] << 8)
               | data[offset + 3];
    }

    private static uint ReadUInt32BE(byte[] data, int offset)
    {
        return ((uint)data[offset] << 24)
               | ((uint)data[offset + 1] << 16)
               | ((uint)data[offset + 2] << 8)
               | data[offset + 3];
    }

    private static void WriteUInt32BE(byte[] data, int offset, int value)
    {
        data[offset] = (byte)((value >> 24) & 0xff);
        data[offset + 1] = (byte)((value >> 16) & 0xff);
        data[offset + 2] = (byte)((value >> 8) & 0xff);
        data[offset + 3] = (byte)(value & 0xff);
    }

    private struct HoloTeleopUdpFragment
    {
        public uint FrameId;
        public int FragmentId;
        public int FragmentCount;
        public int FrameSize;
        public int PayloadSize;
        public byte[] Packet;
    }

    private sealed class HoloTeleopUdpFrameAssembly
    {
        private readonly bool[] _receivedFragments;
        private int _receivedFragmentCount;

        public HoloTeleopUdpFrameAssembly(uint frameId, int frameSize, int fragmentCount)
        {
            FrameId = frameId;
            Frame = new byte[frameSize];
            _receivedFragments = new bool[fragmentCount];
        }

        public uint FrameId { get; }
        public byte[] Frame { get; }
        public bool IsComplete => _receivedFragmentCount == _receivedFragments.Length;

        public bool TryAdd(HoloTeleopUdpFragment fragment)
        {
            if (fragment.FrameSize != Frame.Length
                || fragment.FragmentCount != _receivedFragments.Length
                || fragment.FragmentId < 0
                || fragment.FragmentId >= _receivedFragments.Length
                || _receivedFragments[fragment.FragmentId])
                return false;

            var offset = fragment.FragmentId * HoloTeleopUdpFragmentPayloadSize;
            if (offset < 0 || offset + fragment.PayloadSize > Frame.Length)
                return false;

            Buffer.BlockCopy(fragment.Packet, HoloTeleopUdpHeaderSize, Frame, offset, fragment.PayloadSize);
            _receivedFragments[fragment.FragmentId] = true;
            _receivedFragmentCount++;
            return true;
        }
    }

    private static bool IsFrameIdNewerOrSame(uint frameId, uint latestFrameId)
    {
        return frameId == latestFrameId || (frameId - latestFrameId) < 0x80000000u;
    }

    private static bool IsFrameAssemblyTimedOut(int assemblyStartTime)
    {
        unchecked
        {
            return Environment.TickCount - assemblyStartTime > HoloTeleopUdpFrameTimeoutMs;
        }
    }
}
