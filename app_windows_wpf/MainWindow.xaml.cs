using System.Globalization;
using System.IO;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;
using Microsoft.Win32;
using LocalSecurityMonitor.Wpf.Models;
using LocalSecurityMonitor.Wpf.Services;
using IOPath = System.IO.Path;

namespace LocalSecurityMonitor.Wpf;

public partial class MainWindow : Window
{
    private readonly MainViewModel _viewModel = new();
    private readonly BackendBridge _bridge;
    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
    };
    private readonly DispatcherTimer _cameraSaveTimer;
    private readonly DispatcherTimer _personSaveTimer;
    private readonly DispatcherTimer _faceSaveTimer;
    private readonly string _repoRoot;
    private readonly Dictionary<string, (int Rotation, bool Mirrored)> _transforms = new();
    private bool _suppressCameraEditor;
    private bool _suppressPersonEditor;
    private bool _suppressFaceEditor;
    private bool _suppressTransformEditor;
    private bool _closing;
    private bool _loaded;
    private bool _rememberWindowGeometry;
    private string? _selectedCameraId;
    private PersonDetectionData? _personSnapshot;
    private FaceRecognitionData? _faceSnapshot;
    private double _faceDetectorConfidenceExact = 0.5;
    private bool _faceDetectorConfidenceEdited;
    private double? _faceRecognitionThresholdExact;
    private bool _faceRecognitionThresholdEdited;

    public MainWindow(LaunchOptions options)
    {
        InitializeComponent();
        DataContext = _viewModel;
        _repoRoot = ResolveRepoRoot(options);

        var configPath = string.IsNullOrWhiteSpace(options.ConfigPath)
            ? null
            : IOPath.GetFullPath(IOPath.IsPathRooted(options.ConfigPath)
                ? options.ConfigPath
                : IOPath.Combine(_repoRoot, options.ConfigPath));
        _bridge = new BackendBridge(
            new BackendBridgeOptions(
                _repoRoot,
                configPath,
                options.FakeCameras,
                options.FakeOfflineCamera,
                options.FakeReconnectCamera));
        _bridge.MessageReceived += Bridge_MessageReceived;
        _bridge.ErrorReceived += Bridge_ErrorReceived;
        _bridge.ProcessExited += Bridge_ProcessExited;

        _cameraSaveTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(500) };
        _cameraSaveTimer.Tick += CameraSaveTimer_Tick;
        _personSaveTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(500) };
        _personSaveTimer.Tick += PersonSaveTimer_Tick;
        _faceSaveTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(500) };
        _faceSaveTimer.Tick += FaceSaveTimer_Tick;
    }

    private async void Window_Loaded(object sender, RoutedEventArgs e)
    {
        if (_loaded)
        {
            return;
        }
        _loaded = true;
        try
        {
            await _bridge.StartAsync();
        }
        catch (Exception ex)
        {
            SetBackendError($"Avvio backend non riuscito: {ex.Message}");
        }
    }

    private void Bridge_MessageReceived(BridgeMessage message)
    {
        _ = Dispatcher.InvokeAsync(() =>
        {
            try
            {
                HandleBridgeMessage(message);
            }
            catch (Exception ex)
            {
                SetBackendError($"Risposta backend non gestita: {ex.Message}");
            }
        });
    }

    private void Bridge_ErrorReceived(string message)
    {
        if (string.IsNullOrWhiteSpace(message) || _closing)
        {
            return;
        }
        _ = Dispatcher.InvokeAsync(() =>
        {
            if (!_closing && message.Contains("ERROR", StringComparison.OrdinalIgnoreCase))
            {
                _viewModel.ConnectionStatus = message;
                FooterStatusText.Text = message;
            }
        });
    }

    private void Bridge_ProcessExited()
    {
        _ = Dispatcher.InvokeAsync(() =>
        {
            if (!_closing)
            {
                SetBackendError("Il backend locale si è arrestato.");
            }
        });
    }

    private void HandleBridgeMessage(BridgeMessage message)
    {
        switch (message.Type)
        {
            case "hello":
                HandleHello(Deserialize<HelloData>(message.Data));
                break;
            case "snapshot":
                HandleSnapshot(Deserialize<SnapshotData>(message.Data));
                break;
            case "person_detection":
                HandlePersonDetection(Deserialize<PersonDetectionData>(message.Data));
                break;
            case "camera_save_result":
                HandleCameraSaveResult(message.Data);
                break;
            case "camera_reconfigured":
                HandleCameraReconfigured(message.Data);
                break;
            case "connection_test_started":
                TestConnectionButton.IsEnabled = false;
                SetCameraStatus("Test connessione in corso…", error: false);
                break;
            case "connection_test_result":
                HandleConnectionTestResult(message.Data);
                break;
            case "person_settings_saved":
                HandlePersonSettingsSaved(message.Data);
                break;
            case "face_recognition_state":
                HandleFaceRecognition(Deserialize<FaceRecognitionData>(message.Data));
                break;
            case "face_detection_state":
                HandleFaceDetection(Deserialize<FaceDetectionStateData>(message.Data));
                break;
            case "face_gallery_state":
                HandleFaceGallery(Deserialize<FaceGalleryData>(message.Data));
                break;
            case "face_capabilities":
                HandleFaceCapabilities(message.Data);
                break;
            case "face_settings_saved":
                HandleFaceSettingsSaved(message.Data);
                break;
            case "face_enrollment_result":
                FaceStatusText.Text = ReadBool(message.Data, "ok")
                    ? "Enrollment completato"
                    : "Enrollment rifiutato: nessuna immagine accettata";
                break;
            case "face_enrollment_batch_result":
                HandleFaceEnrollmentBatch(message.Data);
                break;
            case "face_gallery_root_saved":
                UpdateConfigPath(message.Data);
                break;
            case "error":
                HandleErrorMessage(message.Data);
                break;
        }
    }

    private void HandleHello(HelloData hello)
    {
        _viewModel.Initialize(hello);
        _rememberWindowGeometry = hello.Ui.RememberWindowGeometry;
        EnvironmentText.Text = hello.Simulation ? "SIMULAZIONE" : "BACKEND LOCALE";
        FooterStatusText.Text = _viewModel.FooterText;
        FooterConfigText.Text = hello.ConfigPath ?? string.Empty;
        Title = hello.Simulation
            ? "Local Security Monitor — SIMULAZIONE"
            : "Local Security Monitor";

        if (hello.Ui.StartMaximized)
        {
            WindowState = WindowState.Maximized;
        }
        else
        {
            RestoreWindowGeometry();
        }
        PopulatePersonSettings(hello.PersonDetection);
        PopulateFaceSettings(hello.FaceDetection);
        HandleFaceGallery(hello.FaceGallery);
        PopulateFaceCapabilities(hello.FaceCapabilities);
    }

    private void HandleSnapshot(SnapshotData snapshot)
    {
        var camera = _viewModel.FindCamera(snapshot.CameraId);
        if (camera is null)
        {
            return;
        }
        camera.ApplySnapshot(snapshot);
        if (snapshot.CameraId == _selectedCameraId)
        {
            UpdateFocusVisual(camera);
        }
    }

    private void HandlePersonDetection(PersonDetectionData snapshot)
    {
        if (snapshot.CameraId is not null && snapshot.CameraId != _selectedCameraId)
        {
            return;
        }
        _personSnapshot = snapshot;
        DetectionStatusText.Text = snapshot.StatusLabel;
        DetectionStatusText.Foreground = snapshot.Status switch
        {
            "RUNNING" => new SolidColorBrush(Color.FromRgb(22, 125, 74)),
            "ERROR" or "MODEL_MISSING" => new SolidColorBrush(Color.FromRgb(173, 64, 58)),
            "LOADING" => new SolidColorBrush(Color.FromRgb(180, 120, 22)),
            _ => new SolidColorBrush(Color.FromRgb(91, 101, 115)),
        };
        DetectionModelTelemetryText.Text = snapshot.ModelName;
        var device = snapshot.ActualDevice is null
            ? $"richiesto {snapshot.RequestedDevice}"
            : $"{snapshot.Backend ?? "backend n/d"} · {snapshot.ActualDevice.ToUpperInvariant()} · {snapshot.Provider ?? "provider n/d"}"
              + (snapshot.DeviceVerified ? " · verificato" : " · candidato");
        DetectionDeviceTelemetryText.Text = device;
        DetectionPerformanceTelemetryText.Text =
            $"{FormatNullable(snapshot.LatencyMs, "0.0", "ms")}  /  {FormatNullable(snapshot.InferenceFps, "0.0", "FPS")}";
        DetectionCountTelemetryText.Text = snapshot.DetectionCount.ToString(CultureInfo.InvariantCulture);
        PersonStatusDetailText.Text = snapshot.Message;
        RenderDetectionOverlay();
    }

    private void HandleCameraSaveResult(JsonElement data)
    {
        var ok = ReadBool(data, "ok");
        var message = ok ? "Configurazione salvata; applicazione backend in corso…" : ReadString(data, "message");
        SetCameraStatus(message, error: !ok);
        if (ok)
        {
            CameraPasswordBox.Clear();
            _suppressCameraEditor = true;
            ClearPasswordCheckBox.IsChecked = false;
            _suppressCameraEditor = false;
            var path = ReadString(data, "path");
            if (!string.IsNullOrWhiteSpace(path))
            {
                _viewModel.ConfigPath = path;
                FooterConfigText.Text = path;
            }
        }
    }

    private void HandleCameraReconfigured(JsonElement data)
    {
        var ok = ReadBool(data, "ok");
        var message = ReadString(data, "message");
        SetCameraStatus(ok ? message : $"Applicazione non riuscita: {message}", error: !ok);
    }

    private void HandleConnectionTestResult(JsonElement data)
    {
        TestConnectionButton.IsEnabled = true;
        var ok = ReadBool(data, "ok");
        SetCameraStatus(ReadString(data, "message"), error: !ok);
    }

    private void HandlePersonSettingsSaved(JsonElement data)
    {
        if (!ReadBool(data, "ok"))
        {
            PersonStatusDetailText.Text = ReadString(data, "message");
            return;
        }
        if (data.TryGetProperty("settings", out var settings))
        {
            _viewModel.PersonSettings = Deserialize<PersonSettingsData>(settings);
        }
        PersonStatusDetailText.Text = "Configurazione salvata; applicazione backend in corso…";
        var path = ReadString(data, "path");
        if (!string.IsNullOrWhiteSpace(path))
        {
            _viewModel.ConfigPath = path;
            FooterConfigText.Text = path;
        }
    }

    private void HandleFaceSettingsSaved(JsonElement data)
    {
        if (!ReadBool(data, "ok"))
        {
            var error = ReadString(data, "message");
            if (string.Equals(ReadString(data, "section"), "detection", StringComparison.OrdinalIgnoreCase))
            {
                FaceDetectionStatusText.Text = error;
            }
            else
            {
                FaceStatusText.Text = error;
            }
            return;
        }
        if (data.TryGetProperty("settings", out var settings))
        {
            _viewModel.FaceSettings = Deserialize<FaceSettingsData>(settings);
            PopulateFaceSettings(_viewModel.FaceSettings);
        }
        if (string.Equals(ReadString(data, "section"), "detection", StringComparison.OrdinalIgnoreCase))
        {
            FaceDetectionStatusText.Text = "Configurazione detection salvata; applicazione in corso…";
        }
        else
        {
            FaceStatusText.Text = "Configurazione recognition salvata; applicazione in corso…";
        }
        UpdateConfigPath(data);
    }

    private void HandleFaceRecognition(FaceRecognitionData snapshot)
    {
        if (snapshot.CameraId is not null && snapshot.CameraId != _selectedCameraId)
        {
            return;
        }
        _faceSnapshot = snapshot;
        FaceStatusText.Text = $"{snapshot.Status.ToUpperInvariant()}: {snapshot.Message}";
        if (!string.IsNullOrWhiteSpace(snapshot.EffectiveRecognizerId)
            && !string.Equals(snapshot.EffectiveRecognizerId, snapshot.RecognizerId, StringComparison.OrdinalIgnoreCase))
        {
            FaceStatusText.Text += $" · modello effettivo {snapshot.EffectiveRecognizerId}";
        }
        var detectorDevice = snapshot.ActualDetectorDevice is null
            ? $"face detector richiesto {snapshot.RequestedDetectorDevice ?? "n/d"}"
            : $"face detector {snapshot.ActualDetectorDevice.ToUpperInvariant()}";
        var recognizerDevice = snapshot.ActualRecognizerDevice is null
            ? $"recognizer richiesto {snapshot.RequestedRecognizerDevice ?? "n/d"}"
            : $"recognizer {snapshot.ActualRecognizerDevice.ToUpperInvariant()}";
        FaceTelemetryText.Text =
            $"faces {snapshot.FaceCount} · known {snapshot.RecognizedCount} · unknown {snapshot.UnknownCount} · "
            + $"{detectorDevice} · {recognizerDevice} · "
            + (snapshot.Telemetry.Count == 0
                ? "telemetria in attesa"
                : string.Join(" · ", snapshot.Telemetry.Select(item => $"{item.Key}={item.Value}")));
        RenderDetectionOverlay();
    }

    private void HandleFaceDetection(FaceDetectionStateData snapshot)
    {
        if (snapshot.CameraId is not null && snapshot.CameraId != _selectedCameraId)
        {
            return;
        }
        FaceDetectionStatusText.Text = $"{snapshot.StatusLabel}: {snapshot.Message}";
        var device = snapshot.ActualDevice is null
            ? $"richiesto {snapshot.RequestedDevice ?? "n/d"}"
            : $"{snapshot.ActualDevice.ToUpperInvariant()}"
              + (snapshot.DeviceVerified ? " · verificato" : " · non verificato");
        FaceTelemetryText.Text =
            $"detector {snapshot.DetectorId ?? "n/d"} · backend {snapshot.DetectorBackend ?? "n/d"} · "
            + $"modello {snapshot.DetectorModel ?? "n/d"} · device {device} · "
            + $"faces {snapshot.FaceCount}";
        RenderDetectionOverlay();
    }

    private void HandleFaceGallery(FaceGalleryData gallery)
    {
        _viewModel.ApplyFaceGallery(gallery);
        FaceGalleryListEmptyText.Visibility = gallery.EnrollmentPeople.Count == 0
            ? Visibility.Visible
            : Visibility.Collapsed;
        FaceGalleryPeopleCountText.Text = FormatPeopleCount(gallery.EnrollmentPeople.Count);
        FaceGalleryActiveCountText.Text = $"Gallery attiva: {gallery.Persons.Count}";
        FaceGalleryRecordCountText.Text = $"Record attivi: {gallery.EnrollmentPeople.Count(value => value.Active)}";
        var source = gallery.EnrollmentRootPresent
            ? "Sorgente gallery disponibile"
            : "Sorgente gallery non trovata";
        FaceGalleryText.Text = source;
        if (!string.IsNullOrWhiteSpace(gallery.Message)
            && !gallery.Message.Equals("gallery refreshed", StringComparison.OrdinalIgnoreCase))
        {
            FaceGalleryText.Text += $" · {gallery.Message}";
        }
        FaceGalleryRootButton.ToolTip = string.IsNullOrWhiteSpace(gallery.EnrollmentRoot)
            ? "Seleziona la cartella sorgente della Face gallery"
            : gallery.EnrollmentRoot;
        _viewModel.FaceGalleryBusy = false;
    }

    private void HandleFaceEnrollmentBatch(JsonElement data)
    {
        var accepted = data.TryGetProperty("accepted_count", out var acceptedValue)
            && acceptedValue.TryGetInt32(out var acceptedCount)
            ? acceptedCount
            : 0;
        FaceGalleryText.Text = ReadBool(data, "ok")
            ? $"Attivazione completata: {accepted} embedding accettati"
            : "Attivazione completata con errori; controllare il dettaglio del backend.";
        _viewModel.FaceGalleryBusy = false;
    }

    private static string FormatPeopleCount(int count) =>
        count == 1 ? "1 persona" : $"{count} persone";

    private void HandleFaceCapabilities(JsonElement data)
    {
        var payload = Deserialize<FaceCapabilitiesData>(data);
        PopulateFaceCapabilities(payload.Items);
    }

    private void UpdateConfigPath(JsonElement data)
    {
        var path = ReadString(data, "path");
        if (!string.IsNullOrWhiteSpace(path))
        {
            _viewModel.ConfigPath = path;
            FooterConfigText.Text = path;
        }
    }

    private void HandleErrorMessage(JsonElement data)
    {
        var message = ReadString(data, "message");
        if (!string.IsNullOrWhiteSpace(message))
        {
            var command = ReadString(data, "command");
            if (command.Contains("gallery", StringComparison.OrdinalIgnoreCase)
                || command.Contains("enrollment", StringComparison.OrdinalIgnoreCase)
                || command.Equals("remove_person", StringComparison.OrdinalIgnoreCase))
            {
                _viewModel.FaceGalleryBusy = false;
            }
            if (command.Contains("person", StringComparison.OrdinalIgnoreCase))
            {
                PersonStatusDetailText.Text = message;
            }
            else if (command.Contains("face_detection", StringComparison.OrdinalIgnoreCase))
            {
                FaceDetectionStatusText.Text = message;
            }
            else if (command.Contains("face", StringComparison.OrdinalIgnoreCase))
            {
                FaceStatusText.Text = message;
            }
            else
            {
                SetCameraStatus(message, error: true);
            }
        }
    }

    private void CameraTile_Click(object sender, RoutedEventArgs e)
    {
        if (sender is Button { Tag: string cameraId })
        {
            OpenFocus(cameraId);
        }
    }

    private void OpenFocus(string cameraId)
    {
        var camera = _viewModel.FindCamera(cameraId);
        if (camera is null)
        {
            return;
        }
        _selectedCameraId = cameraId;
        _viewModel.SelectedCamera = camera;
        GridView.Visibility = Visibility.Collapsed;
        FocusView.Visibility = Visibility.Visible;
        PopulateCameraEditor(camera);
        UpdateFocusVisual(camera);
        _ = SendActiveCameraAsync(cameraId);
    }

    private void BackToGrid_Click(object sender, RoutedEventArgs e)
    {
        CloseFocus();
    }

    private void CloseFocus()
    {
        _cameraSaveTimer.Stop();
        _personSaveTimer.Stop();
        _faceSaveTimer.Stop();
        _selectedCameraId = null;
        _viewModel.SelectedCamera = null;
        FocusView.Visibility = Visibility.Collapsed;
        GridView.Visibility = Visibility.Visible;
        DetectionCanvas.Children.Clear();
        _faceSnapshot = null;
        _ = SendActiveCameraAsync(null);
    }

    private async Task SendActiveCameraAsync(string? cameraId)
    {
        try
        {
            await _bridge.SendCommandAsync(
                "set_active_camera",
                new Dictionary<string, object?> { ["camera_id"] = cameraId });
        }
        catch (Exception ex)
        {
            SetBackendError(ex.Message);
        }
    }

    private void PopulateCameraEditor(CameraViewModel camera)
    {
        _suppressCameraEditor = true;
        try
        {
            CameraIdText.Text = camera.CameraId;
            CameraNameBox.Text = camera.Name;
            CameraEnabledCheckBox.IsChecked = camera.Enabled;
            SelectTag(CameraSchemeCombo, camera.Editor.Scheme);
            CameraHostBox.Text = camera.Editor.Host;
            CameraPortBox.Text = camera.Editor.Port?.ToString(CultureInfo.InvariantCulture) ?? string.Empty;
            CameraPathBox.Text = camera.Editor.Path;
            CameraUsernameBox.Text = camera.Editor.Username;
            CameraPasswordBox.Clear();
            CameraPasswordBox.ToolTip = camera.Editor.PasswordStored
                ? "Vuoto = conserva la password salvata"
                : "Password opzionale";
            ClearPasswordCheckBox.IsChecked = false;
            ClearPasswordCheckBox.IsEnabled = camera.Editor.PasswordStored;
            SelectTag(CameraTransportCombo, camera.Editor.Transport);
            CameraConfigStatusText.Text = "Le modifiche valide vengono applicate automaticamente.";
            UpdateUrlPreview();
        }
        finally
        {
            _suppressCameraEditor = false;
        }
    }

    private void CameraEditorChanged(object sender, RoutedEventArgs e)
    {
        if (_suppressCameraEditor || _selectedCameraId is null)
        {
            return;
        }
        UpdateUrlPreview();
        SetCameraStatus("Modifica in attesa di applicazione…", error: false);
        _cameraSaveTimer.Stop();
        _cameraSaveTimer.Start();
    }

    private void CameraSaveTimer_Tick(object? sender, EventArgs e)
    {
        _cameraSaveTimer.Stop();
        _ = SendCameraDraftAsync("save_camera");
    }

    private void ApplyCameraNow_Click(object sender, RoutedEventArgs e)
    {
        _cameraSaveTimer.Stop();
        _ = SendCameraDraftAsync("save_camera");
    }

    private void TestConnection_Click(object sender, RoutedEventArgs e)
    {
        _cameraSaveTimer.Stop();
        _ = SendCameraDraftAsync("test_connection");
    }

    private async Task SendCameraDraftAsync(string command)
    {
        if (_selectedCameraId is null || _viewModel.FindCamera(_selectedCameraId) is not { } camera)
        {
            return;
        }
        var data = new Dictionary<string, object?>
        {
            ["camera_id"] = camera.CameraId,
            ["slot_index"] = camera.SlotIndex,
            ["name"] = CameraNameBox.Text,
            ["enabled"] = CameraEnabledCheckBox.IsChecked == true,
            ["scheme"] = SelectedTag(CameraSchemeCombo),
            ["host"] = CameraHostBox.Text,
            ["port"] = CameraPortBox.Text,
            ["path"] = CameraPathBox.Text,
            ["username"] = CameraUsernameBox.Text,
            ["password"] = CameraPasswordBox.Password,
            ["clear_password"] = ClearPasswordCheckBox.IsChecked == true,
            ["transport"] = SelectedTag(CameraTransportCombo),
        };
        try
        {
            if (command == "test_connection")
            {
                SetCameraStatus("Connessione in corso…", error: false);
            }
            await _bridge.SendCommandAsync(command, data);
        }
        catch (Exception ex)
        {
            if (command == "test_connection")
            {
                TestConnectionButton.IsEnabled = true;
            }
            SetCameraStatus(ex.Message, error: true);
        }
    }

    private void UpdateUrlPreview()
    {
        var scheme = SelectedTag(CameraSchemeCombo);
        var host = CameraHostBox.Text.Trim();
        if (string.IsNullOrWhiteSpace(host))
        {
            CameraUrlPreviewText.Text = "URL non configurato";
            return;
        }
        if (host.Contains(':') && !host.StartsWith("[", StringComparison.Ordinal))
        {
            host = $"[{host}]";
        }
        var port = CameraPortBox.Text.Trim();
        var path = CameraPathBox.Text.Trim();
        if (string.IsNullOrWhiteSpace(path))
        {
            path = "/";
        }
        else if (!path.StartsWith("/", StringComparison.Ordinal))
        {
            path = "/" + path;
        }
        var username = CameraUsernameBox.Text.Trim();
        var userInfo = string.IsNullOrWhiteSpace(username) ? string.Empty : $"{username}:***@";
        CameraUrlPreviewText.Text = $"{scheme}://{userInfo}{host}"
            + (string.IsNullOrWhiteSpace(port) ? string.Empty : $":{port}")
            + path;
    }

    private void PopulatePersonSettings(PersonSettingsData settings)
    {
        _suppressPersonEditor = true;
        try
        {
            DetectionEnabledCheckBox.IsChecked = settings.Enabled;
            SelectTag(DetectionBackendCombo, settings.Backend);
            PopulateDetectionModels(settings.Model);
            DetectionPromptsBox.Text = string.Join(", ", settings.Prompts);
            SelectTag(DetectionDeviceCombo, settings.Device);
            SelectTag(DetectionPrecisionCombo, settings.Precision);
            SelectTag(DetectionFallbackCombo, settings.FallbackDevice);
            DetectionConfidenceSlider.Value = settings.ConfidenceThreshold;
            DetectionConfidenceValue.Text = $"{settings.ConfidenceThreshold:P0}";
            DetectionFpsBox.Text = settings.InferenceFps.ToString("0.##", CultureInfo.InvariantCulture);
            DetectionImageSizeBox.Text = settings.ImageSize.ToString(CultureInfo.InvariantCulture);
            DetectionShowBoxesCheckBox.IsChecked = settings.ShowBoxes;
            DetectionShowMasksCheckBox.IsChecked = settings.ShowMasks;
        }
        finally
        {
            _suppressPersonEditor = false;
        }
    }

    private void PopulateFaceSettings(FaceSettingsData settings)
    {
        _suppressFaceEditor = true;
        try
        {
            FaceDetectionEnabledCheckBox.IsChecked = settings.FaceDetectionEnabled;
            FaceLandmarksEnabledCheckBox.IsChecked = settings.LandmarksEnabled;
            FaceRecognitionEnabledCheckBox.IsChecked = settings.RecognitionEnabled;
            PopulateFaceModelCombos(settings);
            SelectTag(FaceDetectorBackendCombo, settings.DetectorBackend);
            SelectTag(FaceDetectorDeviceCombo, settings.DetectorDevice);
            SelectTag(FaceLandmarkerDeviceCombo, settings.LandmarkerDevice);
            SelectTag(FaceRecognizerBackendCombo, settings.RecognizerBackend);
            SelectTag(FaceRecognizerDeviceCombo, settings.RecognizerDevice);
            FaceDetectorModelBox.Text = settings.DetectorModel ?? string.Empty;
            _faceDetectorConfidenceExact = settings.DetectorConfidenceThreshold;
            _faceDetectorConfidenceEdited = false;
            FaceDetectorConfidenceSlider.Value = ToPercentSliderValue(settings.DetectorConfidenceThreshold, minimum: 1);
            FaceDetectorConfidenceValue.Text = $"{FaceDetectorConfidenceSlider.Value:0}%";
            FaceDetectorFpsBox.Text = settings.DetectorInferenceFps.ToString("0.##", CultureInfo.InvariantCulture);
            FaceRecognizerModelBox.Text = settings.RecognizerModel ?? string.Empty;
            _faceRecognitionThresholdExact = settings.RecognitionThreshold;
            _faceRecognitionThresholdEdited = false;
            FaceRecognitionThresholdEnabledCheckBox.IsChecked = settings.RecognitionThreshold is not null;
            FaceRecognitionThresholdSlider.Value = ToPercentSliderValue(settings.RecognitionThreshold ?? 0, minimum: 0);
            FaceRecognitionThresholdSlider.IsEnabled = settings.RecognitionThreshold is not null;
            FaceRecognitionThresholdValue.Text = settings.RecognitionThreshold is null
                ? "disattivata"
                : $"{FaceRecognitionThresholdSlider.Value:0}%";
            FaceRecognitionConfirmationsBox.Text = settings.MinConfirmations.ToString(CultureInfo.InvariantCulture);
            FaceRecognitionWindowBox.Text = settings.ConfirmationWindowSeconds.ToString("0.##", CultureInfo.InvariantCulture);
            FaceRecognitionFpsBox.Text = settings.RecognitionInferenceFps.ToString("0.##", CultureInfo.InvariantCulture);
        }
        finally
        {
            _suppressFaceEditor = false;
        }
    }

    private void PopulateFaceCapabilities(IEnumerable<FaceCapabilityData> capabilities)
    {
        _viewModel.FaceCapabilities.Clear();
        foreach (var capability in capabilities)
        {
            _viewModel.FaceCapabilities.Add(capability);
        }
        var wasSuppressed = _suppressFaceEditor;
        _suppressFaceEditor = true;
        try
        {
            PopulateFaceModelCombos(_viewModel.FaceSettings);
        }
        finally
        {
            _suppressFaceEditor = wasSuppressed;
        }
    }

    private void PopulateFaceModelCombos(FaceSettingsData settings)
    {
        var detectorId = InferFaceModelId(settings.DetectorId, settings.DetectorModel);
        var recognizerId = InferFaceModelId(settings.RecognizerId, settings.RecognizerModel);
        var detectorRows = _viewModel.FaceCapabilities
            .Where(value => value.Component == "face_detection")
            .ToArray();
        var landmarkRows = _viewModel.FaceCapabilities
            .Where(value => value.Component == "face_landmarks")
            .ToArray();
        var recognizerRows = _viewModel.FaceCapabilities
            .Where(value => value.Component == "recognition")
            .ToArray();
        var availableDetectorRows = detectorRows.Where(value => value.Available).ToArray();
        var availableLandmarkRows = landmarkRows.Where(value => value.Available).ToArray();
        var availableRecognizerRows = recognizerRows.Where(value => value.Available).ToArray();
        PopulateFaceModels(FaceDetectorCombo, detectorRows, detectorId);
        PopulateFaceModels(FaceLandmarkerCombo, landmarkRows, settings.LandmarkerId, includeUnavailable: true);
        PopulateFaceModels(FaceRecognizerCombo, recognizerRows, recognizerId);
        PopulateFaceBackends(FaceDetectorBackendCombo, availableDetectorRows, settings.DetectorBackend);
        PopulateFaceBackends(FaceRecognizerBackendCombo, availableRecognizerRows, settings.RecognizerBackend);
        PopulateFaceDevices(FaceDetectorDeviceCombo, availableDetectorRows, detectorId, settings.DetectorBackend, settings.DetectorDevice);
        PopulateFaceDevices(FaceLandmarkerDeviceCombo, availableLandmarkRows, settings.LandmarkerId, "openvino", settings.LandmarkerDevice);
        PopulateFaceDevices(FaceRecognizerDeviceCombo, availableRecognizerRows, recognizerId, settings.RecognizerBackend, settings.RecognizerDevice);
    }

    private static void PopulateFaceModels(
        ComboBox combo,
        IEnumerable<FaceCapabilityData> rows,
        string? configured,
        bool includeUnavailable = false)
    {
        combo.Items.Clear();
        var candidates = rows
            .Where(value => !string.IsNullOrWhiteSpace(value.ModelId))
            .GroupBy(value => value.ModelId, StringComparer.OrdinalIgnoreCase)
            .Select(group => includeUnavailable
                ? group.FirstOrDefault(value => value.Available) ?? group.First()
                : group.FirstOrDefault(value => value.Available))
            .Where(value => value is not null)
            .Select(value => value!)
            .OrderBy(value => value.DisplayName, StringComparer.OrdinalIgnoreCase)
            .ThenBy(value => value.ModelId, StringComparer.OrdinalIgnoreCase)
            .ToList();
        foreach (var capability in candidates)
        {
            var displayName = ShortFaceModelName(capability);
            combo.Items.Add(new ComboBoxItem
            {
                Content = capability.Available || !includeUnavailable
                    ? displayName
                    : $"{displayName} (non disponibile)",
                Tag = capability.ModelId,
                IsEnabled = capability.Available,
                ToolTip = capability.Available
                    ? $"{capability.DisplayName} · {capability.ModelId}"
                    : $"{capability.DisplayName} · {capability.ModelId}: {capability.Reason}",
            });
        }
        if (!string.IsNullOrWhiteSpace(configured)
            && !combo.Items.OfType<ComboBoxItem>().Any(value =>
                string.Equals(value.Tag?.ToString(), configured, StringComparison.OrdinalIgnoreCase)))
        {
            combo.Items.Insert(0, new ComboBoxItem
            {
                Content = $"{configured} (non disponibile)",
                Tag = configured,
                IsEnabled = false,
                ToolTip = "Selezione configurata non presente nella capability matrix",
            });
        }
        if (combo.Items.Count > 0 && !string.IsNullOrWhiteSpace(configured))
        {
            SelectTag(combo, configured);
        }
        else
        {
            combo.SelectedIndex = -1;
        }
    }

    private static string ShortFaceModelName(FaceCapabilityData capability)
    {
        return capability.ModelId.ToLowerInvariant() switch
        {
            "landmarks-regression-retail-0009" => "Landmark 0009",
            "face-reidentification-retail-0095" => "OpenVINO retail-0095",
            "facenet-20180402-vggface2" => "FaceNet VGGFace2",
            "arcface-resnet50-webface600k" => "ArcFace WebFace600K",
            "face_detection_0205" => "Intel face detector 0205",
            "yunet_2023mar" => "YuNet 2023mar",
            "scrfd_2.5g_kps" => "SCRFD 2.5G KPS",
            _ => string.IsNullOrWhiteSpace(capability.DisplayName)
                ? capability.ModelId
                : capability.DisplayName,
        };
    }

    private FaceCapabilityData? FindFaceDetectorCapability(string modelId)
    {
        return _viewModel.FaceCapabilities
            .Where(value => value.Component == "face_detection"
                && string.Equals(value.ModelId, modelId, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(value => value.Available)
            .ThenBy(value => value.Device, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault();
    }

    private void ApplyFaceDetectorSelection()
    {
        var modelId = SelectedTag(FaceDetectorCombo);
        if (string.IsNullOrWhiteSpace(modelId))
        {
            return;
        }

        var capability = FindFaceDetectorCapability(modelId);
        if (capability is null || !capability.Available)
        {
            return;
        }

        var detectorRows = _viewModel.FaceCapabilities
            .Where(value => value.Component == "face_detection" && value.Available)
            .ToArray();
        var compatibleDevices = detectorRows
            .Where(value => string.Equals(value.ModelId, capability.ModelId, StringComparison.OrdinalIgnoreCase)
                && string.Equals(value.Backend, capability.Backend, StringComparison.OrdinalIgnoreCase))
            .Select(value => value.Device)
            .Where(value => !string.IsNullOrWhiteSpace(value))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(value => value, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        var currentDevice = SelectedTag(FaceDetectorDeviceCombo);
        var selectedDevice = compatibleDevices.Contains(currentDevice, StringComparer.OrdinalIgnoreCase)
            ? currentDevice
            : compatibleDevices.FirstOrDefault(value =>
                string.Equals(value, "auto", StringComparison.OrdinalIgnoreCase))
              ?? compatibleDevices.FirstOrDefault()
              ?? "auto";

        var wasSuppressed = _suppressFaceEditor;
        _suppressFaceEditor = true;
        try
        {
            if (!string.IsNullOrWhiteSpace(capability.ModelPath))
            {
                FaceDetectorModelBox.Text = capability.ModelPath;
            }
            PopulateFaceBackends(FaceDetectorBackendCombo, detectorRows, capability.Backend);
            PopulateFaceDevices(
                FaceDetectorDeviceCombo,
                detectorRows,
                capability.ModelId,
                capability.Backend,
                selectedDevice);
        }
        finally
        {
            _suppressFaceEditor = wasSuppressed;
        }
    }

    private FaceCapabilityData? FindFaceRecognizerCapability(string modelId)
    {
        return _viewModel.FaceCapabilities
            .Where(value => value.Component == "recognition"
                && string.Equals(value.ModelId, modelId, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(value => value.Available)
            .ThenBy(value => value.Device, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault();
    }

    private void ApplyFaceRecognizerSelection()
    {
        var modelId = SelectedTag(FaceRecognizerCombo);
        if (string.IsNullOrWhiteSpace(modelId))
        {
            return;
        }

        var capability = FindFaceRecognizerCapability(modelId);
        if (capability is null || !capability.Available)
        {
            return;
        }

        var recognizerRows = _viewModel.FaceCapabilities
            .Where(value => value.Component == "recognition" && value.Available)
            .ToArray();
        var compatibleDevices = recognizerRows
            .Where(value => string.Equals(value.ModelId, capability.ModelId, StringComparison.OrdinalIgnoreCase)
                && string.Equals(value.Backend, capability.Backend, StringComparison.OrdinalIgnoreCase))
            .Select(value => value.Device)
            .Where(value => !string.IsNullOrWhiteSpace(value))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(value => value, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        var currentDevice = SelectedTag(FaceRecognizerDeviceCombo);
        var selectedDevice = compatibleDevices.Contains(currentDevice, StringComparer.OrdinalIgnoreCase)
            ? currentDevice
            : compatibleDevices.FirstOrDefault(value =>
                string.Equals(value, "auto", StringComparison.OrdinalIgnoreCase))
              ?? compatibleDevices.FirstOrDefault()
              ?? "auto";

        var wasSuppressed = _suppressFaceEditor;
        _suppressFaceEditor = true;
        try
        {
            if (!string.IsNullOrWhiteSpace(capability.ModelPath))
            {
                FaceRecognizerModelBox.Text = capability.ModelPath;
            }
            PopulateFaceBackends(FaceRecognizerBackendCombo, recognizerRows, capability.Backend);
            PopulateFaceDevices(
                FaceRecognizerDeviceCombo,
                recognizerRows,
                capability.ModelId,
                capability.Backend,
                selectedDevice);
        }
        finally
        {
            _suppressFaceEditor = wasSuppressed;
        }
    }

    private static string? InferFaceModelId(string? configured, string? model)
    {
        if (!string.IsNullOrWhiteSpace(configured))
        {
            return configured.Trim().ToLowerInvariant() switch
            {
                "face-detection-0205" => "face_detection_0205",
                _ => configured.Trim(),
            };
        }

        var value = (model ?? string.Empty).ToLowerInvariant();
        if (value.Contains("scrfd", StringComparison.Ordinal))
        {
            return "scrfd_2.5g_kps";
        }
        if (value.Contains("0205", StringComparison.Ordinal))
        {
            return "face_detection_0205";
        }
        if (value.Contains("yunet", StringComparison.Ordinal))
        {
            return "yunet_2023mar";
        }
        if (value.Contains("retail-0095", StringComparison.Ordinal))
        {
            return "face-reidentification-retail-0095";
        }
        if (value.Contains("facenet", StringComparison.Ordinal))
        {
            return "facenet-20180402-vggface2";
        }
        if (value.Contains("arcface", StringComparison.Ordinal)
            || value.Contains("w600k", StringComparison.Ordinal))
        {
            return "arcface-resnet50-webface600k";
        }
        return null;
    }

    private static void PopulateFaceBackends(
        ComboBox combo,
        IEnumerable<FaceCapabilityData> rows,
        string configured)
    {
        combo.Items.Clear();
        var values = rows.Select(value => value.Backend)
            .Where(value => !string.IsNullOrWhiteSpace(value))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(value => value, StringComparer.OrdinalIgnoreCase)
            .ToList();
        if (!values.Contains("auto", StringComparer.OrdinalIgnoreCase))
        {
            values.Insert(0, "auto");
        }
        if (!values.Contains(configured, StringComparer.OrdinalIgnoreCase))
        {
            values.Insert(0, configured);
        }
        foreach (var value in values)
        {
            combo.Items.Add(new ComboBoxItem { Content = value, Tag = value });
        }
        SelectTag(combo, configured);
    }

    private static void PopulateFaceDevices(
        ComboBox combo,
        IEnumerable<FaceCapabilityData> rows,
        string? modelId,
        string backend,
        string configured)
    {
        combo.Items.Clear();
        var values = rows
            .Where(value => string.IsNullOrWhiteSpace(modelId)
                || string.Equals(value.ModelId, modelId, StringComparison.OrdinalIgnoreCase))
            .Where(value => string.IsNullOrWhiteSpace(backend)
                || backend == "auto"
                || string.Equals(value.Backend, backend, StringComparison.OrdinalIgnoreCase))
            .Select(value => value.Device)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(value => value, StringComparer.OrdinalIgnoreCase)
            .ToList();
        if (values.Count == 0)
        {
            values.Add(configured);
        }
        if (!values.Contains(configured, StringComparer.OrdinalIgnoreCase))
        {
            values.Insert(0, configured);
        }
        foreach (var value in values)
        {
            combo.Items.Add(new ComboBoxItem { Content = value.ToUpperInvariant(), Tag = value });
        }
        SelectTag(combo, configured);
    }

    private void FaceSettingsChanged(object sender, RoutedEventArgs e) => QueueFaceSave();

    private void FaceSettingsSelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_suppressFaceEditor)
        {
            return;
        }
        if (ReferenceEquals(sender, FaceDetectorCombo))
        {
            ApplyFaceDetectorSelection();
        }
        if (ReferenceEquals(sender, FaceRecognizerCombo))
        {
            ApplyFaceRecognizerSelection();
        }
        QueueFaceSave();
    }

    private void FaceSettingsTextChanged(object sender, TextChangedEventArgs e) => QueueFaceSave();

    private void FaceConfidenceSliderChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (_suppressFaceEditor || FaceDetectorConfidenceValue is null)
        {
            return;
        }
        _faceDetectorConfidenceExact = e.NewValue / 100.0;
        _faceDetectorConfidenceEdited = true;
        FaceDetectorConfidenceValue.Text = $"{e.NewValue:0}%";
        QueueFaceSave();
    }

    private void FaceRecognitionThresholdSliderChanged(
        object sender,
        RoutedPropertyChangedEventArgs<double> e)
    {
        if (_suppressFaceEditor
            || FaceRecognitionThresholdEnabledCheckBox is null
            || FaceRecognitionThresholdValue is null)
        {
            return;
        }
        _faceRecognitionThresholdExact = e.NewValue / 100.0;
        _faceRecognitionThresholdEdited = true;
        FaceRecognitionThresholdValue.Text = $"{e.NewValue:0}%";
        QueueFaceSave();
    }

    private void FaceRecognitionThresholdToggleChanged(object sender, RoutedEventArgs e)
    {
        if (_suppressFaceEditor
            || FaceRecognitionThresholdEnabledCheckBox is null
            || FaceRecognitionThresholdSlider is null
            || FaceRecognitionThresholdValue is null)
        {
            return;
        }
        var enabled = FaceRecognitionThresholdEnabledCheckBox.IsChecked == true;
        FaceRecognitionThresholdSlider.IsEnabled = enabled;
        _faceRecognitionThresholdEdited = true;
        if (enabled)
        {
            _faceRecognitionThresholdExact = FaceRecognitionThresholdSlider.Value / 100.0;
            FaceRecognitionThresholdValue.Text = $"{FaceRecognitionThresholdSlider.Value:0}%";
        }
        else
        {
            _faceRecognitionThresholdExact = null;
            FaceRecognitionThresholdValue.Text = "disattivata";
        }
        QueueFaceSave();
    }

    private void QueueFaceSave()
    {
        if (_suppressFaceEditor)
        {
            return;
        }
        FaceDetectionStatusText.Text = "Modifica configurazione face in attesa…";
        FaceStatusText.Text = "Modifica recognition in attesa…";
        _faceSaveTimer.Stop();
        _faceSaveTimer.Start();
    }

    private void FaceSaveTimer_Tick(object? sender, EventArgs e)
    {
        _faceSaveTimer.Stop();
        _ = SendFaceSettingsAsync();
    }

    private async Task SendFaceSettingsAsync()
    {
        if (!TryParseFaceDouble(FaceDetectorFpsBox.Text, "Sampling detector", minimum: 0, out var detectorFps))
        {
            return;
        }
        if (!TryParseFaceInt(FaceRecognitionConfirmationsBox.Text, "Conferme consecutive", minimum: 1, out var confirmations))
        {
            return;
        }
        if (!TryParseFaceDouble(FaceRecognitionWindowBox.Text, "Finestra conferme", minimum: 0, out var confirmationWindow))
        {
            return;
        }
        if (!TryParseFaceDouble(FaceRecognitionFpsBox.Text, "Sampling recognition", minimum: 0, out var recognitionFps))
        {
            return;
        }
        var threshold = FaceRecognitionThresholdEnabledCheckBox.IsChecked == true
            ? (_faceRecognitionThresholdEdited
                ? FaceRecognitionThresholdSlider.Value / 100.0
                : _faceRecognitionThresholdExact)
            : null;
        var detection = new Dictionary<string, object?>
        {
            ["enabled"] = FaceDetectionEnabledCheckBox.IsChecked == true,
            ["detector_id"] = SelectedTag(FaceDetectorCombo),
            ["backend"] = SelectedTag(FaceDetectorBackendCombo),
            ["model"] = FaceDetectorModelBox.Text.Trim(),
            ["device"] = SelectedTag(FaceDetectorDeviceCombo),
            ["confidence_threshold"] = _faceDetectorConfidenceEdited
                ? FaceDetectorConfidenceSlider.Value / 100.0
                : _faceDetectorConfidenceExact,
            ["inference_fps"] = detectorFps,
            ["landmarks_enabled"] = FaceLandmarksEnabledCheckBox.IsChecked == true,
            ["landmarker_id"] = SelectedTag(FaceLandmarkerCombo),
            ["landmarker_device"] = SelectedTag(FaceLandmarkerDeviceCombo),
        };
        var recognition = new Dictionary<string, object?>
        {
            ["enabled"] = FaceRecognitionEnabledCheckBox.IsChecked == true,
            ["recognizer_id"] = SelectedTag(FaceRecognizerCombo),
            ["backend"] = SelectedTag(FaceRecognizerBackendCombo),
            ["model"] = FaceRecognizerModelBox.Text.Trim(),
            ["device"] = SelectedTag(FaceRecognizerDeviceCombo),
            ["threshold"] = threshold,
            ["min_confirmations"] = confirmations,
            ["confirmation_window_seconds"] = confirmationWindow,
            ["inference_fps"] = recognitionFps,
        };
        try
        {
            await _bridge.SendCommandAsync("set_face_detection", detection);
            await _bridge.SendCommandAsync("set_face_recognition", recognition);
        }
        catch (Exception ex)
        {
            FaceDetectionStatusText.Text = $"Configurazione face non applicata: {ex.Message}";
            FaceStatusText.Text = $"Configurazione recognition non applicata: {ex.Message}";
        }
    }

    private async void RefreshFaceGallery_Click(object sender, RoutedEventArgs e)
    {
        await SendFaceGalleryCommandAsync(
            "refresh_face_gallery",
            new Dictionary<string, object?>(),
            "Scansione persone in corso…");
    }

    private void RefreshFaceCapabilities_Click(object sender, RoutedEventArgs e)
    {
        _ = _bridge.SendCommandAsync("refresh_face_capabilities", new Dictionary<string, object?>());
    }

    private async void RemoveFacePerson_Click(object sender, RoutedEventArgs e)
    {
        var person = _viewModel.SelectedEnrollmentPerson;
        var personId = person?.PersonId?.Trim();
        if (string.IsNullOrWhiteSpace(personId) || person?.Active != true)
        {
            FaceGalleryText.Text = "Selezionare un record attivo da eliminare.";
            return;
        }
        var displayName = string.IsNullOrWhiteSpace(person.Name)
            ? personId
            : person.Name.Trim();
        var confirmation = MessageBox.Show(
            this,
            $"Eliminare il record biometrico di '{displayName}'?\nLa cartella sorgente non verrà eliminata.",
            "Conferma eliminazione persona",
            MessageBoxButton.YesNo,
            MessageBoxImage.Warning);
        if (confirmation != MessageBoxResult.Yes)
        {
            return;
        }
        await SendFaceGalleryCommandAsync(
            "remove_person",
            new Dictionary<string, object?> { ["person_id"] = personId },
            "Eliminazione persona in corso…");
    }

    private async void EnrollFacePerson_Click(object sender, RoutedEventArgs e)
    {
        var person = _viewModel.SelectedEnrollmentPerson;
        if (person is null)
        {
            FaceGalleryText.Text = "Selezionare una cartella enrollment valida.";
            return;
        }
        try
        {
            await _bridge.SendCommandAsync(
                "enroll_person",
                new Dictionary<string, object?>
                {
                    ["person_id"] = person.PersonId,
                });
        }
        catch (Exception ex)
        {
            FaceStatusText.Text = ex.Message;
        }
    }

    private async void ActivateFacePeople_Click(object sender, RoutedEventArgs e)
    {
        await SendFaceGalleryCommandAsync(
            "import_enrollment",
            new Dictionary<string, object?>(),
            "Attivazione persone in corso…");
    }

    private async void SelectFaceGalleryRoot_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFolderDialog
        {
            Title = "Seleziona cartella madre delle persone",
        };
        var configuredRoot = _viewModel.FaceGallery.EnrollmentRoot;
        if (!string.IsNullOrWhiteSpace(configuredRoot) && Directory.Exists(configuredRoot))
        {
            dialog.InitialDirectory = configuredRoot;
        }
        if (dialog.ShowDialog() != true || string.IsNullOrWhiteSpace(dialog.FolderName))
        {
            return;
        }
        await SendFaceGalleryCommandAsync(
            "set_face_gallery_root",
            new Dictionary<string, object?> { ["root"] = dialog.FolderName },
            "Salvataggio cartella gallery in corso…");
    }

    private async Task SendFaceGalleryCommandAsync(
        string command,
        IReadOnlyDictionary<string, object?> data,
        string pendingMessage)
    {
        if (_viewModel.FaceGalleryBusy)
        {
            return;
        }
        _viewModel.FaceGalleryBusy = true;
        FaceGalleryText.Text = pendingMessage;
        try
        {
            await _bridge.SendCommandAsync(command, data);
        }
        catch (Exception ex)
        {
            _viewModel.FaceGalleryBusy = false;
            FaceGalleryText.Text = ex.Message;
        }
    }

    private void DetectionBackendChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_suppressPersonEditor)
        {
            return;
        }

        var configuredModel = SelectedDetectionModel() ?? _viewModel.PersonSettings.Model;
        PopulateDetectionModels(configuredModel);
        PersonSettingsChanged(sender, e);
    }

    private void PopulateDetectionModels(string? configuredModel)
    {
        var wasSuppressed = _suppressPersonEditor;
        _suppressPersonEditor = true;
        try
        {
            var backend = EffectiveDetectionBackend(configuredModel);
            var options = DetectionModelCatalog.Discover(backend, configuredModel, _repoRoot);
            DetectionModelCombo.Items.Clear();

            if (options.Count == 0)
            {
                var label = backend switch
                {
                    "onnx" => "Nessun modello ONNX disponibile",
                    _ => "Nessun modello compatibile disponibile",
                };
                DetectionModelCombo.Items.Add(new ComboBoxItem { Content = label });
                DetectionModelCombo.SelectedIndex = 0;
                DetectionModelCombo.IsEnabled = false;
                return;
            }

            foreach (var option in options)
            {
                DetectionModelCombo.Items.Add(new ComboBoxItem
                {
                    Content = option.Label,
                    Tag = option.Value,
                });
            }

            var selectedIndex = -1;
            if (!string.IsNullOrWhiteSpace(configuredModel))
            {
                for (var index = 0; index < DetectionModelCombo.Items.Count; index++)
                {
                    if (DetectionModelCombo.Items[index] is ComboBoxItem item
                        && string.Equals(item.Tag?.ToString(), configuredModel, StringComparison.OrdinalIgnoreCase))
                    {
                        selectedIndex = index;
                        break;
                    }
                }
            }
            DetectionModelCombo.SelectedIndex = selectedIndex >= 0 ? selectedIndex : 0;
            DetectionModelCombo.IsEnabled = !string.Equals(backend, "fake", StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            _suppressPersonEditor = wasSuppressed;
        }
    }

    private string EffectiveDetectionBackend(string? configuredModel)
    {
        var selectedBackend = SelectedTag(DetectionBackendCombo);
        if (!string.Equals(selectedBackend, "auto", StringComparison.OrdinalIgnoreCase))
        {
            return selectedBackend;
        }

        var model = SelectedDetectionModel() ?? configuredModel ?? _viewModel.PersonSettings.Model;
        var modelName = model?.Trim().ToLowerInvariant() ?? string.Empty;
        if (modelName.EndsWith(".onnx", StringComparison.Ordinal))
        {
            return "onnx";
        }
        if (modelName.EndsWith("yolo26s.pt", StringComparison.Ordinal)
            || modelName.EndsWith("yolo26n.pt", StringComparison.Ordinal)
            || modelName.EndsWith("_openvino_model", StringComparison.Ordinal)
            || modelName.EndsWith(".xml", StringComparison.Ordinal))
        {
            return "openvino";
        }
        return "yoloe";
    }

    private string? SelectedDetectionModel()
    {
        return (DetectionModelCombo.SelectedItem as ComboBoxItem)?.Tag?.ToString() is { Length: > 0 } value
            ? value
            : null;
    }

    private void PersonSettingsChanged(object sender, RoutedEventArgs e)
    {
        if (_suppressPersonEditor)
        {
            return;
        }
        DetectionConfidenceValue.Text = $"{DetectionConfidenceSlider.Value:P0}";
        PersonStatusDetailText.Text = "Modifica in attesa di applicazione…";
        _personSaveTimer.Stop();
        _personSaveTimer.Start();
    }

    private void PersonSaveTimer_Tick(object? sender, EventArgs e)
    {
        _personSaveTimer.Stop();
        _ = SendPersonSettingsAsync();
    }

    private async Task SendPersonSettingsAsync()
    {
        var prompts = DetectionPromptsBox.Text
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(value => !string.IsNullOrWhiteSpace(value))
            .ToArray();
        var data = new Dictionary<string, object?>
        {
            ["enabled"] = DetectionEnabledCheckBox.IsChecked == true,
            ["backend"] = SelectedTag(DetectionBackendCombo),
            ["model"] = SelectedDetectionModel(),
            ["prompts"] = prompts.Length == 0 ? new[] { "person" } : prompts,
            ["device"] = SelectedTag(DetectionDeviceCombo),
            ["precision"] = SelectedTag(DetectionPrecisionCombo),
            ["fallback_device"] = SelectedTag(DetectionFallbackCombo),
            ["confidence_threshold"] = DetectionConfidenceSlider.Value,
            ["inference_fps"] = ParseDouble(DetectionFpsBox.Text, 2),
            ["image_size"] = ParseInt(DetectionImageSizeBox.Text, 640),
            ["show_boxes"] = DetectionShowBoxesCheckBox.IsChecked == true,
            ["show_masks"] = DetectionShowMasksCheckBox.IsChecked == true,
        };
        try
        {
            await _bridge.SendCommandAsync("set_person_detection", data);
        }
        catch (Exception ex)
        {
            PersonStatusDetailText.Text = ex.Message;
        }
    }

    private void RotateButton_Click(object sender, RoutedEventArgs e)
    {
        if (_viewModel.SelectedCamera is not { } camera)
        {
            return;
        }
        camera.RotateCounterClockwise();
        _transforms[camera.CameraId] = (camera.RotationDegrees, camera.IsMirrored);
        _suppressTransformEditor = true;
        MirrorCheckBox.IsChecked = camera.IsMirrored;
        _suppressTransformEditor = false;
        UpdateFocusVisual(camera);
    }

    private void MirrorCheckBox_Changed(object sender, RoutedEventArgs e)
    {
        if (_suppressTransformEditor || _viewModel.SelectedCamera is not { } camera)
        {
            return;
        }
        camera.SetMirrored(MirrorCheckBox.IsChecked == true);
        _transforms[camera.CameraId] = (camera.RotationDegrees, camera.IsMirrored);
        UpdateFocusVisual(camera);
    }

    private void UpdateFocusVisual(CameraViewModel camera)
    {
        FocusTitleText.Text = $"{camera.Name}  ·  {camera.CameraId}";
        FocusStatusText.Text = camera.Status;
        FocusStatusText.Foreground = camera.StatusBrush;
        FocusMetaText.Text = camera.MetaText;
        FocusImage.Source = camera.DisplayImage;
        FocusEmptyText.Text = camera.EmptyMessage;
        FocusEmptyText.Visibility = camera.EmptyVisibility;
        _suppressTransformEditor = true;
        MirrorCheckBox.IsChecked = camera.IsMirrored;
        _suppressTransformEditor = false;
        RenderDetectionOverlay();
    }

    private void RenderDetectionOverlay()
    {
        DetectionOverlayRenderer.Render(
            DetectionCanvas,
            _viewModel.SelectedCamera,
            _personSnapshot,
            _faceSnapshot,
            _viewModel.PersonSettings);
    }

    private void Refresh_Click(object sender, RoutedEventArgs e)
    {
        _ = SendPingAsync();
    }

    private async Task SendPingAsync()
    {
        try
        {
            await _bridge.SendCommandAsync("ping", new Dictionary<string, object?>());
            _viewModel.ConnectionStatus = "Backend locale raggiungibile";
            FooterStatusText.Text = _viewModel.ConnectionStatus;
        }
        catch (Exception ex)
        {
            SetBackendError(ex.Message);
        }
    }

    private void Exit_Click(object sender, RoutedEventArgs e) => Close();

    private void Window_KeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Escape && FocusView.Visibility == Visibility.Visible)
        {
            CloseFocus();
            e.Handled = true;
        }
    }

    private void Window_Closing(object? sender, System.ComponentModel.CancelEventArgs e)
    {
        if (_closing)
        {
            return;
        }
        _closing = true;
        _cameraSaveTimer.Stop();
        _personSaveTimer.Stop();
        _faceSaveTimer.Stop();
        SaveWindowGeometry();
        _bridge.DisposeAsync().AsTask().GetAwaiter().GetResult();
    }

    private void SetCameraStatus(string message, bool error)
    {
        CameraConfigStatusText.Text = message;
        CameraConfigStatusText.Foreground = error
            ? new SolidColorBrush(Color.FromRgb(173, 64, 58))
            : new SolidColorBrush(Color.FromRgb(91, 101, 115));
    }

    private void SetBackendError(string message)
    {
        _viewModel.ConnectionStatus = message;
        FooterStatusText.Text = message;
        FooterStatusText.Foreground = new SolidColorBrush(Color.FromRgb(173, 64, 58));
    }

    private static string ResolveRepoRoot(LaunchOptions options)
    {
        if (!string.IsNullOrWhiteSpace(options.RepoRoot))
        {
            return IOPath.GetFullPath(options.RepoRoot);
        }
        foreach (var start in new[] { Directory.GetCurrentDirectory(), AppContext.BaseDirectory })
        {
            var directory = new DirectoryInfo(start);
            while (directory is not null)
            {
                if (File.Exists(IOPath.Combine(directory.FullName, "pyproject.toml")))
                {
                    return directory.FullName;
                }
                directory = directory.Parent;
            }
        }
        return Directory.GetCurrentDirectory();
    }

    private void RestoreWindowGeometry()
    {
        if (!_rememberWindowGeometry)
        {
            return;
        }
        try
        {
            var path = IOPath.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "LocalSecurityMonitor",
                "window.json");
            if (!File.Exists(path))
            {
                return;
            }
            var geometry = JsonSerializer.Deserialize<WindowGeometry>(File.ReadAllText(path));
            if (geometry is null || geometry.Width < MinWidth || geometry.Height < MinHeight)
            {
                return;
            }
            Width = geometry.Width;
            Height = geometry.Height;
            Left = geometry.Left;
            Top = geometry.Top;
        }
        catch
        {
            // A stale or inaccessible geometry file must never block the monitor.
        }
    }

    private void SaveWindowGeometry()
    {
        if (!_rememberWindowGeometry || WindowState == WindowState.Minimized)
        {
            return;
        }
        try
        {
            var path = IOPath.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "LocalSecurityMonitor",
                "window.json");
            Directory.CreateDirectory(IOPath.GetDirectoryName(path)!);
            var bounds = WindowState == WindowState.Normal
                ? this
                : null;
            var geometry = new WindowGeometry(
                bounds?.Width ?? RestoreBounds.Width,
                bounds?.Height ?? RestoreBounds.Height,
                bounds?.Left ?? RestoreBounds.Left,
                bounds?.Top ?? RestoreBounds.Top);
            File.WriteAllText(path, JsonSerializer.Serialize(geometry));
        }
        catch
        {
            // Geometry persistence is optional.
        }
    }

    private static string SelectedTag(ComboBox combo)
    {
        return (combo.SelectedItem as ComboBoxItem)?.Tag?.ToString()
            ?? combo.Text.Trim().ToLowerInvariant();
    }

    private static void SelectTag(ComboBox combo, string? value)
    {
        for (var index = 0; index < combo.Items.Count; index++)
        {
            if (combo.Items[index] is ComboBoxItem item
                && string.Equals(item.Tag?.ToString(), value, StringComparison.OrdinalIgnoreCase))
            {
                combo.SelectedIndex = index;
                return;
            }
        }
        if (combo.Items.Count > 0)
        {
            combo.SelectedIndex = 0;
        }
    }

    private T Deserialize<T>(JsonElement element) =>
        JsonSerializer.Deserialize<T>(element.GetRawText(), _jsonOptions)
        ?? throw new InvalidOperationException($"Payload {typeof(T).Name} vuoto.");

    private static string ReadString(JsonElement data, string property)
    {
        return data.TryGetProperty(property, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString() ?? string.Empty
            : string.Empty;
    }

    private static bool ReadBool(JsonElement data, string property)
    {
        return data.TryGetProperty(property, out var value)
               && value.ValueKind is JsonValueKind.True or JsonValueKind.False
               && value.GetBoolean();
    }

    private static int ParseInt(string value, int fallback) =>
        int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var result)
            ? result
            : fallback;

    private static double ParseDouble(string value, double fallback) =>
        double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var result)
            ? result
            : fallback;

    private static double ToPercentSliderValue(double fraction, double minimum)
    {
        if (!double.IsFinite(fraction))
        {
            return minimum;
        }
        return Math.Clamp(
            Math.Round(fraction * 100.0, MidpointRounding.AwayFromZero),
            minimum,
            100.0);
    }

    private bool TryParseFaceDouble(
        string value,
        string label,
        double minimum,
        out double result)
    {
        if (double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out result)
            && double.IsFinite(result)
            && result > minimum)
        {
            return true;
        }
        var message = $"{label}: inserire un valore maggiore di {minimum.ToString("0.##", CultureInfo.InvariantCulture)}.";
        if (label.Contains("detector", StringComparison.OrdinalIgnoreCase))
        {
            FaceDetectionStatusText.Text = message;
        }
        else
        {
            FaceStatusText.Text = message;
        }
        return false;
    }

    private bool TryParseFaceInt(
        string value,
        string label,
        int minimum,
        out int result)
    {
        if (int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out result)
            && result >= minimum)
        {
            return true;
        }
        FaceStatusText.Text = $"{label}: inserire un intero maggiore o uguale a {minimum}.";
        return false;
    }

    private static string FormatNullable(double? value, string format, string suffix)
    {
        return value is null ? "—" : $"{value.Value.ToString(format, CultureInfo.InvariantCulture)} {suffix}";
    }

    private sealed record WindowGeometry(double Width, double Height, double Left, double Top);
}
