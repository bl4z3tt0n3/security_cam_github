# Monitor Windows locale

Frontend WPF + C#/.NET per visualizzare fino a sei camere del backend video
esistente. La shell nativa comunica con un bridge locale senza spostare
`CameraWorker`, `CameraMonitorController` o la logica di acquisizione. La GUI
usa un `CameraWorker` indipendente per ogni stream configurato,
con rilevamento persone opzionale nella vista focus (OpenVINO YOLO26, YOLOE,
ONNX legacy o fake offline) e una superficie face/recognition opt-in che
riusa tracking, buffer e capability matrix del core.

## REQUISITI

- Windows 10/11;
- Python 3.11–3.13 a 64 bit;
- dipendenze del progetto, inclusi OpenCV e PySide6;
- FFmpeg disponibile nel `PATH` per il backend RTSP/OpenCV;
- Huawei e PC sulla stessa LAN per il test reale.

## INSTALLAZIONE

Dalla root del progetto:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,windows]"
```

Per abilitare l'inferenza YOLOE locale installare anche l'extra opzionale:

```powershell
python -m pip install -e ".[dev,windows,yoloe]"
```

Per il profilo Intel OpenVINO installare invece/anche:

```powershell
python -m pip install -e ".[dev,windows,person-detection,openvino]"
```

## AVVIO

Avvio consigliato su Windows: fare doppio click su [Avvia_Monitor_Windows.bat](../Avvia_Monitor_Windows.bat)
nella root del progetto. Il launcher preferisce l'eseguibile WPF già compilato,
altrimenti usa `dotnet run`; se .NET non è disponibile mantiene la GUI PySide6
come fallback. Gli argomenti vengono inoltrati alla shell WPF e al bridge.

Esempi:

```bat
Avvia_Monitor_Windows.bat
Avvia_Monitor_Windows.bat --config config\config.local.yaml
Avvia_Monitor_Windows.bat --fake-cameras
```

L'entrypoint Python resta disponibile per diagnostica e test:

```powershell
python -m app_windows.main
```

L'entrypoint Python resta disponibile per diagnostica e per il fallback legacy;
la nuova shell può essere compilata direttamente con:

```powershell
dotnet build app_windows_wpf\LocalSecurityMonitor.Wpf.csproj -c Release
```

La configurazione scelta in automatico è `config/config.local.yaml`; se assente,
viene usato `config/config.example.yaml`. È possibile indicare un file diverso:

```powershell
python -m app_windows.main --config config\config.local.yaml
```

## CONFIGURAZIONE CAMERE

Riutilizzare `config/config.local.yaml`, ignorato da Git. I nomi provengono dalla
sezione centrale `cameras`; se assenti, la GUI usa `Camera 1` … `Camera 6`.

Esempio minimo:

```yaml
cameras:
  - id: huawei_p30
    name: "Huawei P30 Pro"
    enabled: true
    source_type: opencv
    stream_url: "${CAMERA_HUAWEI_URL}"
  - id: cam_2
    name: "Camera 2"
    enabled: false
```

Per RTSP Huawei mantenere:

```yaml
video:
  rtsp_transport: tcp
  reconnect_delay_seconds: 2
  # 0 = retry illimitati fino a riattivazione, disabilitazione o shutdown.
  max_reconnect_attempts: 0
```

Le URL con credenziali non vengono stampate in chiaro nei log.

## CONFIGURAZIONE DALLA GUI

La griglia principale non espone più un pulsante globale di configurazione.
Selezionare una tile per aprire la vista focus: a destra viene mostrata la
configurazione della sola videocamera selezionata, mentre il video resta a
sinistra. Il divisore tra le due aree è ridimensionabile e il pannello di
configurazione può essere compresso completamente.

Il pannello permette di modificare ID visualizzato, nome, abilitazione, schema,
host/IP, porta, path, username, password e trasporto RTSP. Le modifiche valide
vengono salvate e applicate automaticamente 500 ms dopo l'ultima modifica; un
errore di validazione non modifica la configurazione attiva.

L'URL mostrato nel pannello è un'anteprima redatta. La sorgente unica resta
`CameraConfig.stream_url`: i campi host/porta/path servono soltanto a costruire
quella URL, non una seconda lista di stream.

La prima modifica fatta quando è caricato `config/config.example.yaml` crea
`config/config.local.yaml`
senza modificare il file di esempio. Le altre sezioni YAML, compresa
`windows_ui.display_fps`, restano intatte e il file viene pubblicato
atomicamente.

Le password inserite dalla GUI non vengono scritte nel YAML: su Windows sono
protette con DPAPI nel sidecar locale `config/config.local.secrets.json`,
ignorato da Git. Il campo vuoto conserva la password esistente; il checkbox
`Rimuovi password salvata` la elimina. Password, URL complete e credenziali non
vengono stampate nei log.

`Test connessione` usa il provider attivo quando disponibile; altrimenti crea
un test temporaneo tramite lo stesso `VideoSource`/`OpenCVVideoSource`, con
timeout bounded e chiusura garantita. Un test offline non blocca la GUI né le
altre camere.

## RILEVAMENTO PERSONE NELLA VISTA FOCUS

Aprendo una camera dalla griglia, il pannello contestuale consente di abilitare
OpenVINO, YOLOE, ONNX legacy o fake offline, scegliere backend/checkpoint,
dispositivo, precisione, fallback, soglia, FPS e dimensione `imgsz`. Il profilo
OpenVINO usa esclusivamente
`models/yolo26s.pt` o `models/yolo26n.pt`; sono scaricati al primo uso e
convertiti una volta in OpenVINO. Il checkpoint YOLOE installato è
`models/yoloe-26n-seg.pt`; sono disponibili anche
`models/yoloe-26s-seg.pt` e `models/yoloe-26l-seg.pt`. Questi sono checkpoint
prompted e usano le categorie inserite nella GUI. Per i prompt è richiesto
anche l'asset locale `models/mobileclip2_b.ts`. Il modello viene caricato su
un thread separato e
usa il provider video già attivo: non viene aperta una seconda connessione RTSP
e le altre camere non vengono riavviate.

La telemetria mostra sempre modello, backend, device/provider realmente usato,
precisione, latenza, FPS e detection rilevate. Box e maschere vengono disegnati nel video dopo la stessa
rotazione, eventuale specchio e geometria FIT_CENTER del frame. Il messaggio
`Identità persone non attiva` resta intenzionale per le sole categorie prompt:
YOLOE non attiva nomi, embedding, matching o eventi. Il pannello separato
Face/Recognition dispone invece della propria configurazione opt-in.

Il pacchetto opzionale per YOLOE è `.[windows,yoloe]`; per OpenVINO usare
`.[windows,person-detection,openvino]`. Se checkpoint
o encoder prompt mancano, la UI mostra `MODELLO MANCANTE` o `ERRORE` e mantiene
comunque attivo lo stream video.

## FACE DETECTION E RECOGNITION OPT-IN

Il pannello face usa la capability matrix prodotta dal core e filtra modello,
backend e device sulle sole combinazioni con artefatto e probe locali validi.
Face detection, landmark, alignment, quality, embedding, gallery e conferma
temporale vengono eseguiti sul tracking persona già prodotto dalla vista focus;
il controller Windows non crea un secondo detector o tracker. Le gallery sono
scopate per recognizer e fingerprint. `set_face_detection` e
`set_face_recognition` non modificano il percorso person detection quando la
funzione è disabilitata.

L'enrollment passa esclusivamente immagini locali a
`scripts/enroll_person.py`; frame live e auto-enrollment sono fuori scope.
Quando un modello o landmarker non è presente, lo stato resta
`model_missing`/`unsupported` e il video continua.

## MODALITÀ FAKE

Avvia sei sorgenti sintetiche senza hardware o rete:

```powershell
Avvia_Monitor_Windows.bat --fake-cameras
```

Per simulare guasti:

```powershell
Avvia_Monitor_Windows.bat --fake-cameras --fake-offline-camera cam_3
Avvia_Monitor_Windows.bat --fake-cameras --fake-reconnect-camera cam_3
```

La modalità è una simulazione esplicita e non dimostra il supporto RTSP reale.

La finestra mostra `SIMULAZIONE` nel titolo e nella barra di stato quando questa
modalita e attiva; non dimostra il supporto RTSP reale.

## VISTA 6 CAMERE

La finestra mantiene sempre sei slot in disposizione 3×2. Ogni tile mostra nome,
stato e ultimo frame disponibile. Le camere disabilitate o non configurate
restano visibili senza avviare worker.

## VISTA INGRANDITA

Fare click su un tile. Il passaggio alla vista focus riutilizza lo stesso worker
e lo stesso flusso; non apre una seconda connessione. Usare `Esc` oppure
`Torna alle 6 camere` per rientrare nella griglia.

## HUAWEI ANDROID: ENDPOINT RTSP/TCP

Con l'app Android inclusa, avviare `Start Stream` e attendere
`SERVER ATTIVO`. Nella configurazione della Camera 1 inserire:

```text
rtsp://<utente>:<password>@<IP-del-telefono>:8554/stream
```

L'IP e' quello mostrato da Diagnostics Android. Mantenere
`video.rtsp_transport: tcp`; il server Android senza client non e' un errore,
mentre la tile passa a `LIVE` dopo l'arrivo del primo frame. Non abilitare
RTSPS/SRTP/E2EE: non sono implementati dal trasporto Android.

## TEST CON HUAWEI

1. Avviare l’app IP-camera sul Huawei.
2. Mettere Huawei e PC sulla stessa LAN.
3. Impostare localmente `CAMERA_HUAWEI_URL`.
4. Lasciare le camere 2–6 disabilitate.
5. Eseguire prima la diagnostica backend:

```powershell
python scripts\diagnose_stream.py --config config\config.local.yaml --camera-id huawei_p30
python scripts\test_stream.py --config config\config.local.yaml --camera-id huawei_p30 --duration 60
```

6. Avviare `Avvia_Monitor_Windows.bat --config config\config.local.yaml` e
   verificare griglia, focus ed `Esc`.
7. Disattivare temporaneamente il Wi-Fi del Huawei: solo Camera 1 deve andare
   offline/reconnecting.
8. Riattivare il Wi-Fi e verificare il recupero.

Questa procedura non certifica automaticamente sei stream reali; aggiungerli uno
alla volta e misurare carico e stabilità.

## TEST MANUALE HUAWEI DALLA GUI

Per la verifica reale della Camera 1:

1. avviare lo stream RTSP nell'app Android/Huawei;
2. collegare PC e Huawei alla stessa LAN;
3. aprire `Avvia_Monitor_Windows.bat`;
4. selezionare Camera 1/Huawei dalla griglia;
5. inserire schema, host/IP, porta e path mostrati dall'app nel pannello destro;
6. inserire username e password solo nei campi mascherati, se richiesti;
7. lasciare TCP come trasporto predefinito;
8. premere `Test connessione` e attendere `Connessione riuscita`;
9. attendere l'applicazione automatica e verificare il live nella Camera 1;
10. aprire la vista ingrandita, quindi premere `Esc`;
11. arrestare il server RTSP nell'app Android, senza modificare la configurazione Windows;
12. verificare che solo Camera 1 passi a `OFFLINE`/`RICONNESSIONE`;
13. attendere almeno 10 secondi senza premere pulsanti o `Test connessione`;
14. riavviare il server RTSP Android sullo stesso IP, porta e path;
15. verificare che il video torni automaticamente entro pochi secondi;
16. disabilitare Camera 1 e verificare che i retry si interrompano.

Fake provider, test asincroni e test offline non dimostrano il supporto Huawei
reale. Senza dispositivo e URL RTSP disponibili, il test reale resta non verificato.

## DIAGNOSTICA

Gli stati visuali sono `CONNESSIONE`, `LIVE`, `OFFLINE`, `RICONNESSIONE`,
`DISABILITATA`, `ERRORE` e `NON CONFIGURATA`. I dettagli tecnici restano nei log;
il tile mostra messaggi brevi e non bloccanti.

## TROUBLESHOOTING

- `PySide6 non è installato`: eseguire l’installazione `[dev,windows]`.
- `OpenCV is not installed`: installare le dipendenze del progetto nell’ambiente
  virtuale usato per avviare la GUI.
- `NON CONFIGURATA`: controllare variabile d’ambiente e URL nella configurazione
  locale.
- `OFFLINE`: verificare Huawei, LAN, IP, porta, firewall, codec e percorso RTSP.
- Un tile offline non deve impedire agli altri cinque di aggiornarsi.

## PACKAGING

Il packaging PyInstaller è fuori dalla prima verifica. Va preparato soltanto
dopo un avvio reale su Windows, includendo DLL Qt/OpenCV/FFmpeg necessarie e
senza segreti nel pacchetto.

## LIMITI NOTI

- PySide6 e OpenCV devono essere installati per l’esecuzione reale.
- Il repository non espone ancora un bus interprocesso per condividere frame con
  un altro processo già attivo; la GUI riusa le astrazioni backend e crea un solo
  worker per camera nel proprio processo.
- Nessun supporto dichiarato per NVR, registrazioni, PTZ, cloud o notifiche.
  Il pannello face/recognition supporta solo gli artefatti locali verificati:
  le combinazioni mancanti o non probed restano `unsupported`/`model_missing`.
- L'enrollment è un comando esplicito su immagini locali; non esiste
  auto-enrollment da frame live.
- I test automatici e fake non sostituiscono la verifica Huawei/RTSP reale.

## VERIFICA

TEST AUTOMATICI (eseguiti offline):

```powershell
python -m pytest -q
python -m compileall -q app app_windows scripts
git diff --check
```

TEST CON FAKE PROVIDER: `Avvia_Monitor_Windows.bat --fake-cameras`.

TEST CON HUAWEI REALE: diagnostica `diagnose_stream.py`, test sostenuto
`test_stream.py --duration 60`, quindi verifica manuale della GUI.

Multi-camera reale: non verificato finché gli stream non vengono aggiunti e
provati progressivamente.

NON TESTATO: Huawei reale, raggiungibilita RTSP/TCP sulla LAN, recupero dopo
disconnessione Wi-Fi e stabilita di sei stream reali.
