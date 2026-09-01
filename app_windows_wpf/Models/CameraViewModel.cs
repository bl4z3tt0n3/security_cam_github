using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using LocalSecurityMonitor.Wpf.Services;

namespace LocalSecurityMonitor.Wpf.Models;

public sealed class CameraViewModel : INotifyPropertyChanged, IDisposable
{
    private static readonly IReadOnlyDictionary<string, Color> StatusColors =
        new Dictionary<string, Color>(StringComparer.OrdinalIgnoreCase)
        {
            ["LIVE"] = Color.FromRgb(22, 125, 74),
            ["CONNECTING"] = Color.FromRgb(180, 120, 22),
            ["RECONNECTING"] = Color.FromRgb(180, 120, 22),
            ["OFFLINE"] = Color.FromRgb(173, 64, 58),
            ["ERROR"] = Color.FromRgb(173, 64, 58),
            ["DISABLED"] = Color.FromRgb(91, 101, 115),
            ["NOT_CONFIGURED"] = Color.FromRgb(91, 101, 115),
        };

    private readonly SharedFrameReader _frameReader = new();

    private ImageSource? _displayImage;
    private byte[]? _rawFrame;
    private int _rawWidth;
    private int _rawHeight;
    private int _rawStride;
    private string _status = "NON CONFIGURATA";
    private string _statusCode = "NOT_CONFIGURED";
    private string _message = string.Empty;
    private string _metaText = string.Empty;
    private Brush _statusBrush = CreateBrush(StatusColors["NOT_CONFIGURED"]);
    private Visibility _emptyVisibility = Visibility.Visible;
    private int _rotationDegrees;
    private bool _mirrored;

    public CameraViewModel(CameraInfo info)
    {
        SlotIndex = info.SlotIndex;
        CameraId = info.CameraId;
        Name = info.Name;
        Enabled = info.Enabled;
        Configured = info.Configured;
        StreamUrl = info.StreamUrl;
        Transport = info.Transport;
        Editor = info.Editor;
    }

    public event PropertyChangedEventHandler? PropertyChanged;

    public int SlotIndex { get; }
    public string CameraId { get; }
    public string Name { get; private set; }
    public bool Enabled { get; private set; }
    public bool Configured { get; private set; }
    public string? StreamUrl { get; private set; }
    public string Transport { get; private set; }
    public EditorData Editor { get; private set; }

    public string SlotLabel => $"CAM {SlotIndex}";
    public string Status => _status;
    public string Message => _message;
    public Brush StatusBrush => _statusBrush;
    public ImageSource? DisplayImage => _displayImage;
    public Visibility EmptyVisibility => _emptyVisibility;
    public string EmptyMessage => string.IsNullOrWhiteSpace(_message) ? "Nessun frame" : _message;
    public string MetaText => _metaText;
    public bool HasImage => _displayImage is not null;
    public int RotationDegrees => _rotationDegrees;
    public bool IsMirrored => _mirrored;
    public int RawWidth => _rawWidth;
    public int RawHeight => _rawHeight;

    public void ApplySnapshot(SnapshotData snapshot)
    {
        Name = snapshot.Name;
        Enabled = snapshot.Enabled;
        Configured = snapshot.Configured;
        _statusCode = snapshot.Status;
        _status = StatusLabel(snapshot.Status);
        _message = snapshot.Message;
        _statusBrush = CreateBrush(StatusColors.TryGetValue(snapshot.Status, out var color)
            ? color
            : StatusColors["NOT_CONFIGURED"]);
        var details = new List<string>();
        if (snapshot.StreamWidth is not null && snapshot.StreamHeight is not null)
        {
            details.Add($"{snapshot.StreamWidth}×{snapshot.StreamHeight}");
        }
        if (!string.IsNullOrWhiteSpace(snapshot.Codec))
        {
            details.Add(snapshot.Codec!);
        }
        if (!string.IsNullOrWhiteSpace(snapshot.HardwareAcceleration)
            && !snapshot.HardwareAcceleration.Equals("unknown", StringComparison.OrdinalIgnoreCase))
        {
            details.Add($"HW {snapshot.HardwareAcceleration}");
        }
        if (snapshot.LastFrameAgeSeconds is not null)
        {
            details.Add($"ultimo frame {snapshot.LastFrameAgeSeconds.Value:0.0}s");
        }
        if (snapshot.DroppedFrames > 0)
        {
            details.Add($"drop {snapshot.DroppedFrames}");
        }
        _metaText = string.Join("  •  ", details);

        if (!string.IsNullOrWhiteSpace(snapshot.FrameSharedMemoryName))
        {
            if (TryReadSharedFrame(snapshot))
            {
                RebuildImage();
            }
        }

        OnPropertyChanged(nameof(Name));
        OnPropertyChanged(nameof(Enabled));
        OnPropertyChanged(nameof(Configured));
        OnPropertyChanged(nameof(Status));
        OnPropertyChanged(nameof(Message));
        OnPropertyChanged(nameof(StatusBrush));
        OnPropertyChanged(nameof(EmptyMessage));
        OnPropertyChanged(nameof(MetaText));
        OnPropertyChanged(nameof(DisplayImage));
        OnPropertyChanged(nameof(EmptyVisibility));
    }

    public void UpdateEditor(CameraInfo info)
    {
        Name = info.Name;
        Enabled = info.Enabled;
        Configured = info.Configured;
        StreamUrl = info.StreamUrl;
        Transport = info.Transport;
        Editor = info.Editor;
        OnPropertyChanged(nameof(Name));
        OnPropertyChanged(nameof(Enabled));
        OnPropertyChanged(nameof(Configured));
    }

    public void SetDisplayTransform(int rotationDegrees, bool mirrored)
    {
        _rotationDegrees = ((rotationDegrees % 360) + 360) % 360;
        _mirrored = mirrored;
        RebuildImage();
        OnPropertyChanged(nameof(DisplayImage));
        OnPropertyChanged(nameof(RotationDegrees));
        OnPropertyChanged(nameof(IsMirrored));
    }

    public void RotateCounterClockwise()
    {
        SetDisplayTransform(_rotationDegrees + 90, _mirrored);
    }

    public void SetMirrored(bool mirrored)
    {
        SetDisplayTransform(_rotationDegrees, mirrored);
    }

    private bool TryReadSharedFrame(SnapshotData snapshot)
    {
        if (!_frameReader.TryRead(snapshot, out var frame) || frame is null)
        {
            return false;
        }
        _rawFrame = frame.Pixels;
        _rawWidth = frame.Width;
        _rawHeight = frame.Height;
        _rawStride = frame.Stride;
        return true;
    }

    private void RebuildImage()
    {
        if (_rawFrame is null || _rawFrame.Length == 0 || _rawStride <= 0)
        {
            _displayImage = null;
            _emptyVisibility = Visibility.Visible;
            return;
        }

        try
        {
            BitmapSource source = BitmapSource.Create(
                _rawWidth,
                _rawHeight,
                96,
                96,
                PixelFormats.Bgr24,
                null,
                _rawFrame,
                _rawStride);
            source.Freeze();
            if (_rotationDegrees != 0 || _mirrored)
            {
                var group = new TransformGroup();
                if (_rotationDegrees != 0)
                {
                    group.Children.Add(new RotateTransform(-_rotationDegrees));
                }
                if (_mirrored)
                {
                    group.Children.Add(new ScaleTransform(-1, 1));
                }
                var transformed = new TransformedBitmap(source, group);
                transformed.Freeze();
                source = transformed;
            }
            _displayImage = source;
            _emptyVisibility = Visibility.Collapsed;
        }
        catch (Exception)
        {
            _displayImage = null;
            _emptyVisibility = Visibility.Visible;
        }
    }

    private static string StatusLabel(string value) => value.ToUpperInvariant() switch
    {
        "CONNECTING" => "CONNESSIONE",
        "LIVE" => "LIVE",
        "OFFLINE" => "OFFLINE",
        "RECONNECTING" => "RICONNESSIONE",
        "DISABLED" => "DISABILITATA",
        "ERROR" => "ERRORE",
        "NOT_CONFIGURED" => "NON CONFIGURATA",
        _ => value,
    };

    private static SolidColorBrush CreateBrush(Color color)
    {
        var brush = new SolidColorBrush(color);
        brush.Freeze();
        return brush;
    }

    public void Dispose()
    {
        _frameReader.Dispose();
        _rawFrame = null;
        _displayImage = null;
    }

    private void OnPropertyChanged([CallerMemberName] string? name = null) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}
