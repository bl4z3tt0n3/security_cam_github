# Local Security Monitor WPF

La shell `LocalSecurityMonitor.Wpf` è il frontend Windows nativo. Non apre
stream RTSP e non contiene la logica di acquisizione: avvia
`app_windows.wpf_bridge`, che riusa provider, `CameraMonitorController`,
persistenza YAML/DPAPI e rilevamento persone già presenti nel progetto.

Compilazione dalla root:

```powershell
dotnet build app_windows_wpf\LocalSecurityMonitor.Wpf.csproj -c Release
```

Avvio di verifica senza hardware:

```powershell
app_windows_wpf\bin\Release\net8.0-windows\LocalSecurityMonitor.Wpf.exe --fake-cameras
```

Il protocollo bridge è locale e newline-delimited JSON su standard input/output;
non sostituisce né modifica il protocollo RTSP/TCP del backend.
