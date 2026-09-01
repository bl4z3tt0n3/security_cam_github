using System.Text.Json;
using System.Text.Json.Serialization;

namespace LocalSecurityMonitor.Wpf.Models;

public sealed class BridgeMessage
{
    public BridgeMessage(string type, JsonElement data)
    {
        Type = type;
        Data = data;
    }

    public string Type { get; }
    public JsonElement Data { get; }
}

public sealed class HelloData
{
    [JsonPropertyName("simulation")]
    public bool Simulation { get; set; }

    [JsonPropertyName("config_path")]
    public string? ConfigPath { get; set; }

    [JsonPropertyName("ui")]
    public UiData Ui { get; set; } = new();

    [JsonPropertyName("cameras")]
    public List<CameraInfo> Cameras { get; set; } = new();

    [JsonPropertyName("person_detection")]
    public PersonSettingsData PersonDetection { get; set; } = new();

    [JsonPropertyName("face_detection")]
    public FaceSettingsData FaceDetection { get; set; } = new();

    [JsonPropertyName("face_recognition")]
    public FaceSettingsData FaceRecognition { get; set; } = new();

    [JsonPropertyName("face_gallery")]
    public FaceGalleryData FaceGallery { get; set; } = new();

    [JsonPropertyName("face_capabilities")]
    public List<FaceCapabilityData> FaceCapabilities { get; set; } = new();
}

public sealed class UiData
{
    [JsonPropertyName("start_maximized")]
    public bool StartMaximized { get; set; }

    [JsonPropertyName("remember_window_geometry")]
    public bool RememberWindowGeometry { get; set; }

    [JsonPropertyName("display_fps")]
    public double DisplayFps { get; set; }
}

public sealed class CameraInfo
{
    [JsonPropertyName("slot_index")]
    public int SlotIndex { get; set; }

    [JsonPropertyName("camera_id")]
    public string CameraId { get; set; } = string.Empty;

    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("enabled")]
    public bool Enabled { get; set; }

    [JsonPropertyName("configured")]
    public bool Configured { get; set; }

    [JsonPropertyName("stream_url")]
    public string? StreamUrl { get; set; }

    [JsonPropertyName("rtsp_transport")]
    public string Transport { get; set; } = "tcp";

    [JsonPropertyName("editor")]
    public EditorData Editor { get; set; } = new();
}

public sealed class EditorData
{
    [JsonPropertyName("scheme")]
    public string Scheme { get; set; } = "rtsp";

    [JsonPropertyName("host")]
    public string Host { get; set; } = string.Empty;

    [JsonPropertyName("port")]
    public int? Port { get; set; }

    [JsonPropertyName("path")]
    public string Path { get; set; } = string.Empty;

    [JsonPropertyName("username")]
    public string Username { get; set; } = string.Empty;

    [JsonPropertyName("transport")]
    public string Transport { get; set; } = "tcp";

    [JsonPropertyName("query")]
    public string Query { get; set; } = string.Empty;

    [JsonPropertyName("fragment")]
    public string Fragment { get; set; } = string.Empty;

    [JsonPropertyName("password_stored")]
    public bool PasswordStored { get; set; }
}

public sealed class SnapshotData
{
    [JsonPropertyName("camera_id")]
    public string CameraId { get; set; } = string.Empty;

    [JsonPropertyName("slot_index")]
    public int SlotIndex { get; set; }

    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("enabled")]
    public bool Enabled { get; set; }

    [JsonPropertyName("configured")]
    public bool Configured { get; set; }

    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;

    [JsonPropertyName("message")]
    public string Message { get; set; } = string.Empty;

    [JsonPropertyName("last_frame_age_s")]
    public double? LastFrameAgeSeconds { get; set; }

    [JsonPropertyName("display_fps")]
    public double? DisplayFps { get; set; }

    [JsonPropertyName("frame_sequence")]
    public long? FrameSequence { get; set; }

    [JsonPropertyName("frame_shm_name")]
    public string? FrameSharedMemoryName { get; set; }

    [JsonPropertyName("frame_byte_count")]
    public int? FrameByteCount { get; set; }

    [JsonPropertyName("frame_stride")]
    public int? FrameStride { get; set; }

    [JsonPropertyName("frame_width")]
    public int? FrameWidth { get; set; }

    [JsonPropertyName("frame_height")]
    public int? FrameHeight { get; set; }

    [JsonPropertyName("stream_width")]
    public int? StreamWidth { get; set; }

    [JsonPropertyName("stream_height")]
    public int? StreamHeight { get; set; }

    [JsonPropertyName("codec")]
    public string? Codec { get; set; }

    [JsonPropertyName("hardware_acceleration")]
    public string? HardwareAcceleration { get; set; }

    [JsonPropertyName("dropped_frames")]
    public int DroppedFrames { get; set; }
}

public sealed class PersonSettingsData
{
    [JsonPropertyName("enabled")]
    public bool Enabled { get; set; }

    [JsonPropertyName("backend")]
    public string Backend { get; set; } = "yoloe";

    [JsonPropertyName("model")]
    public string? Model { get; set; }

    [JsonPropertyName("confidence_threshold")]
    public double ConfidenceThreshold { get; set; } = 0.5;

    [JsonPropertyName("inference_fps")]
    public double InferenceFps { get; set; } = 2;

    [JsonPropertyName("device")]
    public string Device { get; set; } = "auto";

    [JsonPropertyName("precision")]
    public string Precision { get; set; } = "fp16";

    [JsonPropertyName("fallback_device")]
    public string FallbackDevice { get; set; } = "none";

    [JsonPropertyName("image_size")]
    public int ImageSize { get; set; } = 640;

    [JsonPropertyName("classes")]
    public List<string> Classes { get; set; } = new() { "person" };

    [JsonPropertyName("prompts")]
    public List<string> Prompts { get; set; } = new() { "person" };

    [JsonPropertyName("show_boxes")]
    public bool ShowBoxes { get; set; } = true;

    [JsonPropertyName("show_masks")]
    public bool ShowMasks { get; set; }
}

public sealed class PersonDetectionData
{
    [JsonPropertyName("camera_id")]
    public string? CameraId { get; set; }

    [JsonPropertyName("status")]
    public string Status { get; set; } = "DISABLED";

    [JsonPropertyName("status_label")]
    public string StatusLabel { get; set; } = "DISABILITATO";

    [JsonPropertyName("message")]
    public string Message { get; set; } = string.Empty;

    [JsonPropertyName("model_name")]
    public string ModelName { get; set; } = "—";

    [JsonPropertyName("requested_device")]
    public string RequestedDevice { get; set; } = "auto";

    [JsonPropertyName("actual_device")]
    public string? ActualDevice { get; set; }

    [JsonPropertyName("device_verified")]
    public bool DeviceVerified { get; set; }

    [JsonPropertyName("provider")]
    public string? Provider { get; set; }

    [JsonPropertyName("backend")]
    public string? Backend { get; set; }

    [JsonPropertyName("precision")]
    public string? Precision { get; set; }

    [JsonPropertyName("inference_fps")]
    public double? InferenceFps { get; set; }

    [JsonPropertyName("latency_ms")]
    public double? LatencyMs { get; set; }

    [JsonPropertyName("person_count")]
    public int PersonCount { get; set; }

    [JsonPropertyName("detection_count")]
    public int DetectionCount { get; set; }

    [JsonPropertyName("source_width")]
    public int? SourceWidth { get; set; }

    [JsonPropertyName("source_height")]
    public int? SourceHeight { get; set; }

    [JsonPropertyName("detections")]
    public List<DetectionData> Detections { get; set; } = new();
}

public sealed class DetectionData
{
    [JsonPropertyName("bbox")]
    public double[] Bbox { get; set; } = Array.Empty<double>();

    [JsonPropertyName("confidence")]
    public double Confidence { get; set; }

    [JsonPropertyName("label")]
    public string Label { get; set; } = "person";

    [JsonPropertyName("mask_polygon")]
    public List<double[]>? MaskPolygon { get; set; }
}

public sealed class FaceSettingsData
{
    [JsonPropertyName("face_detection_enabled")]
    public bool FaceDetectionEnabled { get; set; }

    [JsonPropertyName("detector_id")]
    public string? DetectorId { get; set; }

    [JsonPropertyName("detector_backend")]
    public string DetectorBackend { get; set; } = "auto";

    [JsonPropertyName("detector_model")]
    public string? DetectorModel { get; set; }

    [JsonPropertyName("detector_device")]
    public string DetectorDevice { get; set; } = "auto";

    [JsonPropertyName("detector_confidence_threshold")]
    public double DetectorConfidenceThreshold { get; set; } = 0.5;

    [JsonPropertyName("detector_inference_fps")]
    public double DetectorInferenceFps { get; set; } = 2;

    [JsonPropertyName("landmarks_enabled")]
    public bool LandmarksEnabled { get; set; }

    [JsonPropertyName("landmarker_id")]
    public string LandmarkerId { get; set; } = "landmarks-regression-retail-0009";

    [JsonPropertyName("landmarker_model")]
    public string? LandmarkerModel { get; set; }

    [JsonPropertyName("landmarker_device")]
    public string LandmarkerDevice { get; set; } = "auto";

    [JsonPropertyName("recognition_enabled")]
    public bool RecognitionEnabled { get; set; }

    [JsonPropertyName("recognizer_id")]
    public string? RecognizerId { get; set; }

    [JsonPropertyName("recognizer_backend")]
    public string RecognizerBackend { get; set; } = "auto";

    [JsonPropertyName("recognizer_model")]
    public string? RecognizerModel { get; set; }

    [JsonPropertyName("recognizer_device")]
    public string RecognizerDevice { get; set; } = "auto";

    [JsonPropertyName("recognition_threshold")]
    public double? RecognitionThreshold { get; set; }

    [JsonPropertyName("recognition_inference_fps")]
    public double RecognitionInferenceFps { get; set; } = 1;

    [JsonPropertyName("min_confirmations")]
    public int MinConfirmations { get; set; } = 2;

    [JsonPropertyName("confirmation_window_seconds")]
    public double ConfirmationWindowSeconds { get; set; } = 10;
}

public sealed class FaceOverlayData
{
    [JsonPropertyName("camera_id")]
    public string CameraId { get; set; } = string.Empty;

    [JsonPropertyName("track_id")]
    public int TrackId { get; set; }

    [JsonPropertyName("bbox")]
    public double[] Bbox { get; set; } = Array.Empty<double>();

    [JsonPropertyName("landmarks")]
    public List<double[]> Landmarks { get; set; } = new();

    [JsonPropertyName("recognition_status")]
    public string RecognitionStatus { get; set; } = "unknown";

    [JsonPropertyName("person_id")]
    public string? PersonId { get; set; }

    [JsonPropertyName("person_name")]
    public string? PersonName { get; set; }

    [JsonPropertyName("score")]
    public double? Score { get; set; }

    [JsonPropertyName("threshold")]
    public double? Threshold { get; set; }
}

public sealed class FaceRecognitionData
{
    [JsonPropertyName("camera_id")]
    public string? CameraId { get; set; }

    [JsonPropertyName("status")]
    public string Status { get; set; } = "disabled";

    [JsonPropertyName("message")]
    public string Message { get; set; } = string.Empty;

    [JsonPropertyName("error")]
    public string? Error { get; set; }

    [JsonPropertyName("requested_detector_device")]
    public string? RequestedDetectorDevice { get; set; }

    [JsonPropertyName("actual_detector_device")]
    public string? ActualDetectorDevice { get; set; }

    [JsonPropertyName("requested_recognizer_device")]
    public string? RequestedRecognizerDevice { get; set; }

    [JsonPropertyName("actual_recognizer_device")]
    public string? ActualRecognizerDevice { get; set; }

    [JsonPropertyName("detector_id")]
    public string? DetectorId { get; set; }

    [JsonPropertyName("recognizer_id")]
    public string? RecognizerId { get; set; }

    [JsonPropertyName("effective_recognizer_id")]
    public string? EffectiveRecognizerId { get; set; }

    [JsonPropertyName("detector_backend")]
    public string? DetectorBackend { get; set; }

    [JsonPropertyName("detector_model")]
    public string? DetectorModel { get; set; }

    [JsonPropertyName("recognizer_backend")]
    public string? RecognizerBackend { get; set; }

    [JsonPropertyName("recognizer_model")]
    public string? RecognizerModel { get; set; }

    [JsonPropertyName("frame_sequence")]
    public long? FrameSequence { get; set; }

    [JsonPropertyName("face_count")]
    public int FaceCount { get; set; }

    [JsonPropertyName("recognized_count")]
    public int RecognizedCount { get; set; }

    [JsonPropertyName("unknown_count")]
    public int UnknownCount { get; set; }

    [JsonPropertyName("overlays")]
    public List<FaceOverlayData> Overlays { get; set; } = new();

    [JsonPropertyName("telemetry")]
    public Dictionary<string, JsonElement> Telemetry { get; set; } = new();
}

public sealed class FaceDetectionStateData
{
    [JsonPropertyName("camera_id")]
    public string? CameraId { get; set; }

    [JsonPropertyName("status")]
    public string Status { get; set; } = "disabled";

    [JsonPropertyName("status_label")]
    public string StatusLabel { get; set; } = "DISABILITATO";

    [JsonPropertyName("message")]
    public string Message { get; set; } = string.Empty;

    [JsonPropertyName("error")]
    public string? Error { get; set; }

    [JsonPropertyName("detector_id")]
    public string? DetectorId { get; set; }

    [JsonPropertyName("detector_backend")]
    public string? DetectorBackend { get; set; }

    [JsonPropertyName("detector_model")]
    public string? DetectorModel { get; set; }

    [JsonPropertyName("requested_device")]
    public string? RequestedDevice { get; set; }

    [JsonPropertyName("actual_device")]
    public string? ActualDevice { get; set; }

    [JsonPropertyName("device_verified")]
    public bool DeviceVerified { get; set; }

    [JsonPropertyName("landmarker_id")]
    public string? LandmarkerId { get; set; }

    [JsonPropertyName("landmarker_model")]
    public string? LandmarkerModel { get; set; }

    [JsonPropertyName("landmarker_device")]
    public string? LandmarkerDevice { get; set; }

    [JsonPropertyName("frame_sequence")]
    public long? FrameSequence { get; set; }

    [JsonPropertyName("face_count")]
    public int FaceCount { get; set; }

    [JsonPropertyName("overlays")]
    public List<FaceOverlayData> Overlays { get; set; } = new();
}

public sealed class FaceCapabilitiesData
{
    [JsonPropertyName("items")]
    public List<FaceCapabilityData> Items { get; set; } = new();
}

public sealed class FaceCapabilityData
{
    [JsonPropertyName("component")]
    public string Component { get; set; } = string.Empty;

    [JsonPropertyName("model_id")]
    public string ModelId { get; set; } = string.Empty;

    [JsonPropertyName("display_name")]
    public string DisplayName { get; set; } = string.Empty;

    [JsonPropertyName("source")]
    public string Source { get; set; } = string.Empty;

    [JsonPropertyName("license")]
    public string License { get; set; } = string.Empty;

    [JsonPropertyName("model_path")]
    public string ModelPath { get; set; } = string.Empty;

    [JsonPropertyName("backend")]
    public string Backend { get; set; } = string.Empty;

    [JsonPropertyName("device")]
    public string Device { get; set; } = string.Empty;

    [JsonPropertyName("available")]
    public bool Available { get; set; }

    [JsonPropertyName("artifact_present")]
    public bool ArtifactPresent { get; set; }

    [JsonPropertyName("reason")]
    public string Reason { get; set; } = string.Empty;

    [JsonPropertyName("embedding_dimension")]
    public int? EmbeddingDimension { get; set; }

    [JsonPropertyName("probed")]
    public bool Probed { get; set; }

    [JsonPropertyName("actual_device")]
    public string? ActualDevice { get; set; }

    [JsonIgnore]
    public string StatusLabel => Available ? "READY" : "NOT READY";

    [JsonIgnore]
    public string EffectiveDeviceLabel => ActualDevice ?? "—";
}

public sealed class FaceGalleryData
{
    [JsonPropertyName("recognizer_id")]
    public string? RecognizerId { get; set; }

    [JsonPropertyName("effective_recognizer_id")]
    public string? EffectiveRecognizerId { get; set; }

    [JsonPropertyName("fingerprint")]
    public string? Fingerprint { get; set; }

    [JsonPropertyName("status")]
    public string Status { get; set; } = "ready";

    [JsonPropertyName("message")]
    public string Message { get; set; } = string.Empty;

    [JsonPropertyName("error")]
    public string? Error { get; set; }

    [JsonPropertyName("persons")]
    public List<FacePersonData> Persons { get; set; } = new();

    [JsonPropertyName("enrollment_people")]
    public List<FaceEnrollmentPersonData> EnrollmentPeople { get; set; } = new();

    [JsonPropertyName("enrollment_root")]
    public string? EnrollmentRoot { get; set; }

    [JsonPropertyName("enrollment_root_present")]
    public bool EnrollmentRootPresent { get; set; } = true;
}

public sealed class FacePersonData
{
    [JsonPropertyName("person_id")]
    public string PersonId { get; set; } = string.Empty;

    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("embedding_count")]
    public int EmbeddingCount { get; set; }

    [JsonPropertyName("fingerprint")]
    public string? Fingerprint { get; set; }
}

public sealed class FaceEnrollmentPersonData
{
    [JsonPropertyName("person_id")]
    public string PersonId { get; set; } = string.Empty;

    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("image_count")]
    public int ImageCount { get; set; }

    [JsonPropertyName("embedding_count")]
    public int EmbeddingCount { get; set; }

    [JsonPropertyName("active")]
    public bool Active { get; set; }

    [JsonPropertyName("valid")]
    public bool Valid { get; set; }

    [JsonPropertyName("source_available")]
    public bool SourceAvailable { get; set; }

    [JsonPropertyName("status")]
    public string Status { get; set; } = "not_active";

    [JsonIgnore]
    public string StatusLabel => Status.ToLowerInvariant() switch
    {
        "active" => "ATTIVA",
        "not_active" => "NON ATTIVA",
        "empty" => "CARTELLA VUOTA",
        "invalid" => "ID NON VALIDO",
        "unreadable" => "NON LEGGIBILE",
        "missing" => "SORGENTE ASSENTE",
        _ => Status.ToUpperInvariant(),
    };

    [JsonIgnore]
    public string DetailsLabel =>
        $"{ImageCount} immagini · {EmbeddingCount} embedding";

    [JsonIgnore]
    public string InitialLabel
    {
        get
        {
            var value = string.IsNullOrWhiteSpace(Name) ? PersonId.Trim() : Name.Trim();
            return string.IsNullOrWhiteSpace(value)
                ? "?"
                : value[..1].ToUpperInvariant();
        }
    }
}
