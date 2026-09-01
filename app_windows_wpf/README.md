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

## Trasporto preview

Il canale di controllo tra WPF e Python resta newline-delimited JSON, ma i pixel
video non vengono più serializzati come JPEG/Base64 dentro JSON.

Per ogni camera il bridge Python pubblica l'ultimo frame BGR in una named memory
mapping Windows. Il messaggio JSON contiene soltanto nome della mapping,
sequenza, dimensioni, stride e byte count. L'header usa un contatore odd/even:
il reader WPF accetta il frame soltanto se il contatore resta pari e invariato
prima e dopo la copia, evitando frame parzialmente sovrascritti.

La camera in focus viene pubblicata alla frequenza UI; le miniature non in focus
sono limitate a 5 FPS e, sopra 480 px di larghezza, ridotte prima della
pubblicazione. Questo evita JPEG encode/decode, Base64 e il relativo aumento di
banda/memoria tra i due processi.

I componenti sono separati in:

- `app_windows/shared_preview.py`: writer della named mapping;
- `Services/SharedFrameReader.cs`: reader WPF e gestione lifetime;
- `Services/DetectionOverlayRenderer.cs`: rendering degli overlay separato da
  `MainWindow`.

