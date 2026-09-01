using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;

namespace LocalSecurityMonitor.Wpf.Models;

public sealed class MainViewModel : INotifyPropertyChanged, IDisposable
{
    private CameraViewModel? _selectedCamera;
    private string _connectionStatus = "Avvio del backend locale…";
    private string _configPath = string.Empty;
    private string _environmentLabel = "BACKEND LOCALE";
    private string _footerText = "In attesa del backend";
    private PersonSettingsData _personSettings = new();
    private FaceSettingsData _faceSettings = new();
    private FaceRecognitionData _faceState = new();
    private FaceGalleryData _faceGallery = new();
    private FaceEnrollmentPersonData? _selectedEnrollmentPerson;
    private bool _faceGalleryBusy;

    public ObservableCollection<CameraViewModel> Cameras { get; } = new();
    public event PropertyChangedEventHandler? PropertyChanged;

    public CameraViewModel? SelectedCamera
    {
        get => _selectedCamera;
        set => SetField(ref _selectedCamera, value);
    }

    public string ConnectionStatus
    {
        get => _connectionStatus;
        set => SetField(ref _connectionStatus, value);
    }

    public string ConfigPath
    {
        get => _configPath;
        set => SetField(ref _configPath, value);
    }

    public string EnvironmentLabel
    {
        get => _environmentLabel;
        set => SetField(ref _environmentLabel, value);
    }

    public string FooterText
    {
        get => _footerText;
        set => SetField(ref _footerText, value);
    }

    public PersonSettingsData PersonSettings
    {
        get => _personSettings;
        set => SetField(ref _personSettings, value);
    }

    public FaceSettingsData FaceSettings
    {
        get => _faceSettings;
        set => SetField(ref _faceSettings, value);
    }

    public FaceRecognitionData FaceState
    {
        get => _faceState;
        set => SetField(ref _faceState, value);
    }

    public FaceGalleryData FaceGallery
    {
        get => _faceGallery;
        set => SetField(ref _faceGallery, value);
    }

    public ObservableCollection<FaceCapabilityData> FaceCapabilities { get; } = new();
    public ObservableCollection<FaceEnrollmentPersonData> EnrollmentPeople { get; } = new();

    public FaceEnrollmentPersonData? SelectedEnrollmentPerson
    {
        get => _selectedEnrollmentPerson;
        set
        {
            if (!SetField(ref _selectedEnrollmentPerson, value))
            {
                return;
            }
            OnPropertyChanged(nameof(CanEnrollSelectedPerson));
            OnPropertyChanged(nameof(CanRemoveSelectedPerson));
            OnPropertyChanged(nameof(CanActivateFacePeople));
        }
    }

    public bool FaceGalleryBusy
    {
        get => _faceGalleryBusy;
        set
        {
            if (!SetField(ref _faceGalleryBusy, value))
            {
                return;
            }
            OnPropertyChanged(nameof(CanRefreshFaceGallery));
            OnPropertyChanged(nameof(CanActivateFacePeople));
            OnPropertyChanged(nameof(CanRemoveSelectedPerson));
            OnPropertyChanged(nameof(CanSelectFaceGalleryRoot));
        }
    }

    public bool CanRefreshFaceGallery => !FaceGalleryBusy;

    public bool CanActivateFacePeople =>
        !FaceGalleryBusy
        && EnrollmentPeople.Any(value => value.Valid && value.SourceAvailable && value.ImageCount > 0);

    public bool CanSelectFaceGalleryRoot => !FaceGalleryBusy;

    public bool CanEnrollSelectedPerson =>
        SelectedEnrollmentPerson is { Valid: true, SourceAvailable: true, ImageCount: > 0 };

    public bool CanRemoveSelectedPerson => !FaceGalleryBusy && SelectedEnrollmentPerson?.Active == true;

    public void Initialize(HelloData hello)
    {
        foreach (var camera in Cameras)
        {
            camera.Dispose();
        }
        Cameras.Clear();
        foreach (var camera in hello.Cameras.OrderBy(value => value.SlotIndex))
        {
            Cameras.Add(new CameraViewModel(camera));
        }
        ConfigPath = hello.ConfigPath ?? string.Empty;
        EnvironmentLabel = hello.Simulation ? "SIMULAZIONE" : "BACKEND LOCALE";
        FooterText = hello.Simulation
            ? "Frame sintetici locali — nessun flusso RTSP reale"
            : "Acquisizione locale attiva";
        PersonSettings = hello.PersonDetection;
        FaceSettings = hello.FaceDetection;
        ApplyFaceGallery(hello.FaceGallery);
        FaceCapabilities.Clear();
        foreach (var capability in hello.FaceCapabilities)
        {
            FaceCapabilities.Add(capability);
        }
        OnPropertyChanged(nameof(Cameras));
    }

    public void ApplyFaceGallery(FaceGalleryData gallery)
    {
        FaceGallery = gallery;
        var selectedId = SelectedEnrollmentPerson?.PersonId;
        EnrollmentPeople.Clear();
        foreach (var person in gallery.EnrollmentPeople)
        {
            EnrollmentPeople.Add(person);
        }
        SelectedEnrollmentPerson = selectedId is null
            ? null
            : EnrollmentPeople.FirstOrDefault(value =>
                string.Equals(value.PersonId, selectedId, StringComparison.OrdinalIgnoreCase));
        OnPropertyChanged(nameof(CanEnrollSelectedPerson));
        OnPropertyChanged(nameof(CanRemoveSelectedPerson));
        OnPropertyChanged(nameof(CanActivateFacePeople));
    }

    public CameraViewModel? FindCamera(string? cameraId) =>
        cameraId is null ? null : Cameras.FirstOrDefault(value => value.CameraId == cameraId);

    private bool SetField<T>(ref T field, T value, [CallerMemberName] string? propertyName = null)
    {
        if (EqualityComparer<T>.Default.Equals(field, value))
        {
            return false;
        }
        field = value;
        OnPropertyChanged(propertyName);
        return true;
    }

    public void Dispose()
    {
        foreach (var camera in Cameras)
        {
            camera.Dispose();
        }
        Cameras.Clear();
    }

    private void OnPropertyChanged([CallerMemberName] string? propertyName = null) =>
        PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
}
