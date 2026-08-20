# Sistema di videosorveglianza locale low-cost

Prototipo local-first per ricevere e diagnosticare uno stream video LAN, inizialmente fornito da un Huawei P30 Pro. Questa consegna implementa le Fasi A/B e le Fasi 1-11, inclusi sampling, person detection YOLO26s/OpenVINO FP16 opzionale, YOLOE promptata, backend ONNX legacy, tracking, analisi facciale, enrollment locale, matching, conferma temporale, eventi locali, metriche, motion detection opt-in, benchmark di scalabilità, calibrazione offline e hardening locale.

Il riconoscimento locale con soglia e conferma temporale è disponibile come
componente opt-in della pipeline live. Face detection e recognition restano
disabilitati per default; quando abilitati, l'orchestratore riusa il tracking
persona e il buffer latest-frame esistenti. Il layer degli eventi è collegabile
alla pipeline tramite un publisher opzionale. La configurazione di esempio
mantiene i modelli disabilitati finché non vengono installati localmente.

## Pipeline face corrente

Il percorso implementato è:

```text
VideoSource -> CameraWorker -> latest frame -> FrameSampler -> person detector
-> IoUGreedyTracker -> FaceRecognitionOrchestrator -> face detector
-> landmarker -> alignment -> quality -> embedding -> gallery/matcher
-> conferma per camera_id + track_id -> stato/telemetria/UI/bridge
```

La factory e la capability matrix verificano artefatto, runtime, input/output
e device prima di costruire detector, landmarker o recognizer. Non esiste
fallback implicito a CPU. Le gallery sono isolate per `recognizer_id` e
fingerprint; gli embedding di modelli diversi non vengono confrontati.

## Stato della prima consegna

- sorgente astratta `VideoSource`, indipendente dal dispositivo;
- `OpenCVVideoSource` con reader thread e buffer limitato all’ultimo frame;
- `FakeVideoSource` per test senza Huawei;
- `CameraWorker` indipendente con reconnect, watchdog, statistiche e shutdown controllato;
- trasporto RTSP predefinito TCP applicato solo durante l’apertura della capture FFmpeg;
- diagnostica codec, risoluzione, FPS dichiarati/reali, timeout, frame corrotti, disconnessioni e reconnect;
- integrazione opzionale con `ffprobe`;
- nessun cloud, port forwarding o server web;
- person detection sostituibile eseguita soltanto sui frame campionati, con OpenVINO YOLO26s, YOLOE e backend ONNX legacy;
- modello e runtime separati dal repository: i pesi restano nella directory locale `models/`.

La latenza riportata dagli script è la durata locale della lettura/decodifica. Non è una misura end-to-end tra fotocamera e computer.

## Requisiti

Su Windows sono richiesti:

- Python 3.11–3.13 a 64 bit;
- FFmpeg con `ffprobe` consigliato e `ffplay` opzionale;
- rete locale condivisa tra Huawei e computer.

Nell’ambiente di sviluppo rilevato per questo progetto: Windows 11, Python 3.13.14,
FFmpeg/ffprobe/ffplay disponibili, PyTorch CPU-only e Intel Iris Xe Graphics
esposta da OpenVINO come `GPU`. L’inferenza YOLO26s OpenVINO FP16 è stata
verificata su CPU (`EXECUTION_DEVICES=['CPU']`) e GPU Iris Xe
(`['GPU.0']`); la telemetria GPU resta non disponibile senza un tool Intel
locale.

## Installazione Windows

Aprire PowerShell nella directory del progetto:

```powershell
Set-Location D:\CODEX\security_cam
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,windows]"
```

Per avviare il monitor Windows con doppio click usare
`Avvia_Monitor_Windows.bat` nella root del progetto. Il launcher preferisce la
nuova shell WPF/.NET e il bridge locale verso il backend Python; usa la GUI
PySide6 soltanto come fallback se .NET non è disponibile. Gli argomenti vengono
inoltrati, per esempio `Avvia_Monitor_Windows.bat --fake-cameras`.

La shell WPF è in `app_windows_wpf`; il frontend non duplica streaming,
protocolli RTSP/TCP, configurazione o inferenza. Per una compilazione esplicita:

```powershell
dotnet build app_windows_wpf\LocalSecurityMonitor.Wpf.csproj -c Release
```

Se PowerShell impedisce l’attivazione dello script, usare temporaneamente:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\Activate.ps1
```

L’installazione base non scarica PyTorch o modelli. L’extra
`person-detection` installa Ultralytics e PyTorch; se la person detection è
abilitata e `model` è un identificativo ufficiale, Ultralytics può scaricare il
checkpoint al primo caricamento.

Per il backend Intel OpenVINO installare l’extra separato:

```powershell
python -m pip install -e ".[dev,person-detection,openvino]"
```

L’extra OpenVINO non è richiesto dalle installazioni YOLOE/ONNX esistenti.

## Installazione FFmpeg

Verificare l’installazione:

```powershell
ffmpeg -version
ffprobe -version
ffplay -version
```

Se i comandi non sono trovati, installare FFmpeg con un metodo approvato per il computer e aggiungere la directory `bin` al `PATH`. `ffprobe` migliora la diagnostica ma OpenCV resta il ricevitore principale.

## Configurazione

Creare la configurazione locale ignorata da Git:

```powershell
Copy-Item config\config.example.yaml config\config.local.yaml
```

Impostare l’URL solo nella sessione PowerShell, senza inserirlo nel repository:

```powershell
$env:CAMERA_HUAWEI_URL = "INSERISCI QUI L'URL DELLO STREAM DEL TUO HUAWEI"
$env:CAMERA_CAM_2_URL = "INSERISCI QUI L'URL DELLO STREAM DELLA SECONDA CAMERA"
```

Il trasporto RTSP TCP è configurato in `config/config.local.yaml` tramite `video.rtsp_transport: tcp`; non è più necessario impostare manualmente `OPENCV_FFMPEG_CAPTURE_OPTIONS`.

Sono accettati URL `rtsp://`, `rtsps://`, `http://` e `https://`. RTSP con H.264 è il formato preferito. Non inserire password reali in `.env.example`, `config.example.yaml` o nei commit.

Su Windows la configurazione si modifica nella vista focus: selezionare una
camera dalla griglia e usare il pannello laterale della sola camera selezionata.
Le modifiche valide vengono applicate automaticamente dopo una breve pausa e
salvate in `config/config.local.yaml`; le password inserite dalla GUI sono
protette con DPAPI nel sidecar locale ignorato da Git e non vengono scritte nel
YAML o nei log.

## Health check locale

Eseguire dalla root del progetto:

```powershell
python scripts\check_environment.py
```

Il comando verifica Python, OpenCV, FFmpeg, ffprobe, CPU, RAM, storage e configurazione. Se la person detection è abilitata controlla anche runtime, provider e presenza del modello; quando è disabilitata conferma che non viene caricato alcun modello. Per aggiungere una prova breve dello stream:

```powershell
python scripts\check_environment.py --url "INSERISCI QUI L'URL DELLO STREAM DEL TUO HUAWEI"
```

## Huawei Android integrato: endpoint RTSP/TCP

Il repository include ora l'app Android `app_android`, che avvia sul telefono
una pipeline Camera2 -> MediaCodec H.264 -> RTP -> RTSP/TCP. Per il monitor
Windows l'endpoint predefinito e':

```text
rtsp://<utente>:<password>@<IP-del-telefono>:8554/stream
```

Avviare `Start Stream` sull'app, attendere `SERVER ATTIVO`, poi configurare
solo la Camera 1 nel pannello Windows con IP, porta, path e credenziali
mascherate. Il server senza client e' uno stato atteso; la tile Windows diventa
`LIVE` quando riceve frame. Durante uno stream attivo l'app Android riprende
automaticamente dopo il ritorno della LAN; uno stop manuale annulla il recupero.

La password Android e' cifrata a riposo tramite Android Keystore/AES-GCM e le
password Windows usano DPAPI. Il trasporto video resta RTSP/TCP con Basic Auth:
TLS, RTSPS, SRTP, DTLS ed E2EE sono **NOT IMPLEMENTED BY DESIGN**. Non esporre
la porta alla rete Internet e non configurare port forwarding.

La procedura seguente resta utile solo per sorgenti IP-camera esterne diverse
dall'app Android inclusa.

## Configurazione manuale del Huawei

Il progetto non sceglie né hardcodifica una specifica app Android. L’app IP-camera deve preferibilmente offrire:

- stream solo sulla LAN;
- RTSP oppure HTTP/MJPEG leggibile da FFmpeg/OpenCV;
- risoluzione, FPS e bitrate configurabili;
- codec H.264 preferibilmente;
- autenticazione locale opzionale;
- funzionamento prolungato senza cloud o relay Internet obbligatorio.

Procedura:

1. Collegare Huawei e computer allo stesso SSID/rete locale.
2. Disabilitare, se presente, AP/client isolation sul router.
3. Avviare la funzione IP camera nell’app e annotare l’IP locale mostrato dal telefono o dal router.
4. Annotare porta e percorso dello stream. L’URL finale avrà una forma simile a `rtsp://IP:PORT/PERCORSO`, ma il percorso dipende dall’app e non va inventato.
5. Collegare il Huawei all’alimentazione USB-C e disabilitare il risparmio energetico per l’app durante il test.
6. Creare una DHCP reservation nel router per l’indirizzo MAC del telefono, così l’IP non cambia dopo il riavvio.
7. Non creare port forwarding verso Internet.

Se l’app non fornisce un URL esplicito, usare la sua documentazione o la schermata “server/IP camera”. Il programma non può ricavare il percorso RTSP senza questa informazione.

## Test della rete e dello stream

Sostituire i placeholder soltanto nella sessione locale:

```powershell
$cameraIp = "INSERISCI_IP_LOCALE_DEL_HUAWEI"
$cameraPort = 8554
Test-NetConnection $cameraIp -Port $cameraPort
```

Poi provare lo stream con FFmpeg:

```powershell
ffprobe -v error -show_streams -show_format "INSERISCI QUI L'URL DELLO STREAM DEL TUO HUAWEI"
ffplay "INSERISCI QUI L'URL DELLO STREAM DEL TUO HUAWEI"
```

Se `ffplay` non è disponibile, usare VLC o un altro player locale. Una connessione TCP riuscita non garantisce che il percorso dello stream o il codec siano corretti.

Diagnostica dettagliata di dieci secondi:

```powershell
python scripts\diagnose_stream.py --url "INSERISCI QUI L'URL DELLO STREAM DEL TUO HUAWEI"
```

Test sostenuto di un minuto:

```powershell
python scripts\test_stream.py --url "INSERISCI QUI L'URL DELLO STREAM DEL TUO HUAWEI" --duration 60
```

Per avviare il solo worker di acquisizione e visualizzare le metriche live:

```powershell
python scripts\run_camera_worker.py --config config\config.local.yaml --camera-id huawei_p30
```

Il comando mostra stato, connessione stabilita, FPS, reconnect count, frame scartati e dimensione del buffer. Terminare con `Ctrl+C` per una chiusura pulita.

Risultato atteso:

- `RESULT: OK`;
- risoluzione e codec coerenti con la configurazione dell’app;
- FPS reali vicini a quelli dichiarati;
- zero o pochi frame corrotti;
- nessuna disconnessione non spiegata;
- `dropped frames` limitati e nessuna crescita indefinita della latenza.

Le CLI supportano anche una configurazione:

```powershell
python scripts\diagnose_stream.py --config config\config.local.yaml --camera-id huawei_p30
python scripts\test_stream.py --config config\config.local.yaml --camera-id huawei_p30 --duration 60
```

## Diagnostica degli errori

Gli script non riportano soltanto `VideoCapture failed`. In caso di errore mostrano l’URL redatto e controlli per:

- Huawei spento, sospeso o app terminata;
- telefono e computer su reti diverse;
- IP o porta errati;
- IP cambiato senza DHCP reservation;
- firewall o AP isolation;
- stream non avviato;
- protocollo, URL o codec non supportati;
- timeout durante apertura o lettura.

Un errore di `ffprobe` non è automaticamente fatale se OpenCV riesce a ricevere frame. Se invece OpenCV non riceve frame, il test termina con codice di errore e le possibili cause vengono stampate.

## Test automatici offline

I test non richiedono il Huawei:

```powershell
python -m pytest -q
python -m compileall app scripts
git diff --check
```

`FakeVideoSource` permette di verificare frame, timeout, disconnessioni e reconnect senza rete reale. I test OpenCV usano una capture finta iniettata, quindi non aprono indirizzi esterni.

## Codici di uscita

- `0`: controllo o test riuscito;
- `1`: prerequisito mancante o stream non ricevuto;
- `2`: argomenti o configurazione non validi.

## Sicurezza e privacy

Il sistema è local-first. Non espone API, stream, immagini o database e non configura porte Internet. Le credenziali vengono oscurate nei report; per la GUI Windows usare il dialog e il sidecar DPAPI locale, mentre per le CLI continuare a preferire variabili d’ambiente o `config.local.yaml` ignorato.

## Fase 2 - frame sampling e metriche

Il livello di sampling e' configurabile senza modificare il codice:

```yaml
inference:
  person_detection_fps: 2
```

Il sampler legge il buffer live del `CameraWorker` su un thread indipendente e
pubblica solo l'ultimo frame campionato. Usa un clock monotono, non dipende dagli
FPS dichiarati dalla sorgente e non carica ancora alcun modello di detection.

La chiave precedente `person_detection.inference_fps` resta supportata come alias
legacy; se sono presenti entrambe le chiavi, prevale `inference.person_detection_fps`.

Il runner mostra separatamente:

- `stream fps`: FPS dichiarati dalla sorgente, oppure `n/d`;
- `decoded fps`: frame realmente ricevuti dal worker;
- `sampled fps`: frame inoltrati dal sampler;
- frame persi nei buffer e frame esclusi intenzionalmente dal sampling;
- dimensione delle code di acquisizione e campionamento;
- latenza locale approssimativa dal timestamp di ricezione al consumo nel sampler.

La latenza non e' una misura end-to-end dalla telecamera al computer.

## Fase 3 - person detection sostituibile

Il detector riceve soltanto i frame pubblicati dal `FrameSampler`. Il worker di
acquisizione non carica modelli e non esegue inferenza. Il backend candidato
nell’esempio è OpenVINO con YOLO26s, FP16, GPU e fallback CPU; la sezione resta
disabilitata finché il runtime non viene installato e il device non viene
verificato da `EXECUTION_DEVICES`. YOLOE e `OnnxPersonDetector` restano
disponibili senza cambiare il loro contratto.

OpenVINO accetta esclusivamente `yolo26s.pt` e `yolo26n.pt` per il download
ufficiale. L’export viene eseguito una sola volta nella cache
`models/yolo26*_openvino_model`, con `quantize=16` per FP16; l’eventuale cache
CPU FP32 usa lo stem separato `*_fp32_openvino_model`. Metadata e marker locali
impediscono di riusare una cache con task, precisione o `imgsz` diversi.
Il risultato filtra la sola classe `person`, ignora maschere, clippa bbox
finite/valide e conserva il timestamp del frame.

YOLOE usa il checkpoint testuale `yoloe-26n-seg.pt` e `set_classes(["person"])`
una sola volta durante la costruzione dell’adapter. L’output utilizzato è solo
la bounding box, non la maschera. Le detection restituite hanno coordinate
`x1,y1,x2,y2` nel sistema di pixel del frame originale, confidence e timestamp.
Le coordinate e le classi vengono comunque filtrate difensivamente prima di
essere propagate.

### Installazione del runtime

Il runtime di detection è opzionale, quindi il progetto continua a funzionare
con `person_detection.enabled: false` senza importare Ultralytics, PyTorch o
caricare un modello:

```powershell
python -m pip install -e ".[dev,person-detection]"
```

I pesi non vengono committati. Per scaricare il checkpoint ufficiale YOLOE:

```powershell
python -m pip install -e ".[dev,person-detection]"
python -c "from ultralytics import YOLOE; YOLOE('models/yoloe-26n-seg.pt')"
```

È anche possibile impostare `model` a un file `.pt` locale nella directory
`models/`. Un modello `.onnx` va usato con `backend: onnx`: gli export YOLOE
non accettano nuovi prompt runtime dopo l’export.

### Configurazione

```yaml
person_detection:
  enabled: false
  backend: "openvino"
  model: "models/yolo26s.pt"
  prompts:
    - "person"
  classes:
    - "person"
  confidence_threshold: 0.45
  precision: fp16
  device: gpu
  fallback_device: cpu
  image_size: 640

inference:
  person_detection_fps: 2
```

Per OpenVINO i device validi sono `auto`, `cpu` e `gpu`; vengono scelti soltanto
tra i nomi restituiti da `Core.available_devices`. `fallback_device: cpu` abilita
il solo percorso GPU→CPU e viene registrato nel log. La riuscita GPU non viene
dichiarata finché una vera inferenza non espone una voce GPU in
`EXECUTION_DEVICES`. YOLOE/ONNX mantengono i device PyTorch/ONNX originali
(`auto`, `cpu`, `cuda`). Il runner mostra modello, backend, precisione, provider,
device richiesto e device verificato.

### Esecuzione sullo stream Huawei

Senza preview, con log delle metriche:

```powershell
python scripts\run_person_detection.py --config config\config.local.yaml --camera-id huawei_p30 --device cpu --duration 60
```

Per selezionare esplicitamente OpenVINO senza modificare il file:

```powershell
python scripts\run_person_detection.py --config config\config.local.yaml --camera-id huawei_p30 --detector-backend openvino --model models\yolo26s.pt --device gpu --precision fp16 --fallback-device cpu --imgsz 640 --duration 60
```

Con preview opzionale e bounding box:

```powershell
python scripts\run_person_detection.py --config config\config.local.yaml --camera-id huawei_p30 --preview
```

Il comando mostra backend, prompt, sampled FPS, tempo medio di inferenza,
persone nell'ultimo frame, persone totali, numero di frame processati, stato
della camera, track attivi e provider/device realmente utilizzato. `q` o `Esc`
chiudono la preview; `Ctrl+C` esegue lo shutdown pulito.

Per una prova CPU senza configurazione YAML è possibile passare direttamente
l'URL, mantenendo il modello locale configurato di default:

```powershell
python scripts\run_person_detection.py --url "$env:CAMERA_HUAWEI_URL" --device cpu --duration 60
```

Non inserire URL con credenziali nei commit o nei log.

## Fase 4 - tracking e macchina a stati

Ogni camera possiede una `CameraTrackingPipeline` indipendente, con tracker e
macchina a stati propri. Il tracker associa le detection consecutive con IoU
greedy e usa la distanza dei centri come fallback. Gli ID sono temporanei,
locali alla pipeline e non vengono riutilizzati durante la sua vita.

La configurazione è nella sezione `tracking`:

```yaml
tracking:
  iou_threshold: 0.30
  max_center_distance_px: 100
  max_missed_samples: 3
```

Un track resta vivo fino a `max_missed_samples` campioni senza detection; al
campione successivo viene chiuso. Gli stati camera sono `PERSON_SCAN`,
`PERSON_DETECTED`, `TRACKING`, `FACE_ANALYSIS`, `KNOWN`, `UNKNOWN` e
`COOLDOWN`. `PERSON_DETECTED` è registrato come transizione verso `TRACKING`.

La Fase 5 aggiunge contratti sostituibili per face detector, landmarker,
alignment e qualità. Il servizio analizza soltanto i crop dei track attivi
della camera che sta elaborando; una camera senza persone non invoca il face
detector. Sono disponibili adapter locali per SCRFD, face-detection-0205 e
YuNet, oltre ai fake per i test. Il runtime costruisce la catena reale soltanto
quando le sezioni face sono abilitate e la capability probe è positiva.

La configurazione `face_quality` verifica dimensioni, varianza del Laplaciano,
luminosità, bbox parziali e confidence. Ogni rifiuto espone i relativi motivi
per logging e calibrazione.

### Verifica

I test automatici usano `FakePersonDetector`, un modello YOLOE falso iniettato
e sessioni ONNX finte, quindi non richiedono né il Huawei né i pesi. La prova
YOLOE reale CPU richiede Ultralytics, PyTorch e il checkpoint; la prova live
richiede anche lo stream Huawei raggiungibile sulla LAN. Se uno di questi
prerequisiti manca, il risultato resta non verificato.

## Stato dopo la Fase 6

La Fase 6 aggiunge enrollment e storage locale degli embedding. Il runtime
live usa lo stesso servizio e lo stesso tracking del percorso persona; i fake
restano confinati a test e demo. La prova Huawei/RTSP rimane separata dalla
verifica locale dei contratti e dei modelli.

## Fase 6 - enrollment locale

### Setup esplicito dei modelli face

`scripts/setup_face_models.py` è l'unico punto che può scaricare un artefatto,
e lo fa soltanto con `--download`. Senza quel flag esegue verifiche locali,
stampa il digest dichiarato dall'operatore con `--sha256`, la licenza registrata
con `--show-license` e il probe reale di runtime con `--probe`. La conversione
ONNX → OpenVINO IR richiede esplicitamente `--convert-onnx --output-xml`.
L'applicazione, i test normali e il bridge non invocano nessuna di queste
operazioni.

Per una verifica read-only dei modelli già presenti:

```powershell
python scripts\inspect_face_models.py --all
python scripts\setup_face_models.py --all --probe
```

Un modello mancante o non compilabile resta `NOT READY` e non viene registrato
come capability disponibile.

Installare il runtime opzionale e predisporre localmente due modelli ONNX: un
face detector con output `[x1, y1, x2, y2, confidence]` e un face embedder con
input NCHW a tre canali e output di un embedding.

```powershell
python -m pip install -e ".[dev,face-embedding]"
Copy-Item config\config.example.yaml config\config.local.yaml
```

Impostare in `config/config.local.yaml` i percorsi locali e abilitare
`face_detection.enabled` e `recognition.enabled`. L'adapter embedder usa RGB,
resize alla dimensione dichiarata dal modello e normalizzazione
`(pixel - 127.5) / 128`; il contratto completo viene salvato nei metadata.

Eseguire l'enrollment:

```powershell
python scripts\enroll_person.py --name "Mario Rossi" --images .\enrollment\mario\
```

Sono disponibili anche `--config`, `--face-model`, `--embedding-model`,
`--persons-dir`, `--person-id` e `--overwrite`. Il comando non stampa
embedding completi e rifiuta immagini senza volto, con più volti o di qualità
insufficiente.

Il risultato viene scritto in
`persons/<recognizer_id>/<fingerprint>/<person_id>/metadata.json` e
`persons/<recognizer_id>/<fingerprint>/<person_id>/embeddings.npz`. La
directory `persons/` contiene dati
biometrici sensibili: non versionarla, limita l'accesso all'account locale e
non copiarla su servizi cloud. Su Windows è possibile applicare ACL locali con
un comando eseguito intenzionalmente dall'utente, dalla root del progetto:

```powershell
icacls .\persons /inheritance:r /grant:r "$env:USERNAME:(OI)(CI)F"
```

Il modello viene identificato tramite backend, id, versione, preprocessing,
dimensione, provider e SHA-256 quando il file è disponibile. Record prodotti da
contratti incompatibili devono essere rigenerati; non vengono confrontati tra
loro. Se il modello o il runtime opzionale non sono presenti, la CLI termina
con un errore esplicito e non salva dati parziali.

## Fase 7 - matching locale e conferma temporale

Il matcher locale è separato dall'embedder. Carica soltanto i record presenti
nello scope `persons/<recognizer_id>/<fingerprint>/` attivo e verifica che il
contratto del modello coincida esattamente con
quello dell'embedder attivo: backend, modello, versione, dimensione,
preprocessing, provider e SHA-256 quando disponibile. Un record incompatibile o
malformato blocca il matching invece di essere ignorato silenziosamente.

Gli embedding sono confrontati con similarità coseno, calcolata come prodotto
scalare dopo normalizzazione L2. Lo score è quindi migliore quando è più alto;
per ogni persona viene usato il punteggio massimo tra i suoi embedding di
enrollment. La persona più simile viene restituita come `known` solo quando:

```text
score >= recognition.threshold
```

Se la soglia è `null`, il risultato resta `unknown` anche con una similarità
alta. Un risultato `unknown` non contiene mai `person_id` o `person_name`, pur
potendo esporre lo score del miglior candidato per calibrazione.

La conferma temporale è mantenuta per ogni track e per ogni camera. Il valore
`recognition.min_confirmations` richiede osservazioni consecutive coerenti; un
risultato incoerente azzera solo la streak pendente del track. Un'identità già
confermata resta stabile finché una nuova identità non raggiunge nuovamente la
soglia di conferme. Quando il track termina, tutto il relativo stato viene
eliminato.

L'integrazione della pipeline collega `FACE_ANALYSIS` a `KNOWN` o `UNKNOWN`
soltanto dopo la conferma. Track contemporanei e camere diverse non condividono
identità, contatori o stato. Il publisher eventi, se configurato, riceve solo
riconoscimenti già confermati e delega a EventManager la deduplica per camera,
track e tipo di evento.

La verifica automatica usa anche `FakeEmbedder`, `PersonStore` e risultati
sintetici; non dimostra la qualità di un modello facciale reale. Le prove dei
modelli concreti sono esplicite e richiedono artefatti locali e runtime
compatibili.

## Fase 8 - eventi locali e snapshot

`EventManager` salva gli eventi in `events/YYYY/MM/DD/<event_id>/`, con un
`metadata.json` e, quando `events.save_snapshot` è attivo e viene fornito un
frame, uno `snapshot.jpg`. I metadata usano path relativi e contengono camera,
track, tipo `known_person` o `unknown_person`, identità quando disponibile e
score di recognition.

La deduplica è indipendente per camera, track e tipo. Le ripetizioni dello
stesso track rispettano `known_person_cooldown_seconds` o
`unknown_person_cooldown_seconds`; un nuovo track della stessa persona può
generare un nuovo evento. Gli snapshot sono scritti su una coda bounded in
background e gli errori di storage vengono isolati dalla pipeline.

La pipeline accetta un publisher eventi opzionale e lo invoca soltanto dopo la
conferma temporale della recognition. Face analysis, matcher e conferma sono
ora cablati nel runtime opt-in; gli eventi restano un confine separato.

La registrazione video resta disabilitata per default. `RecordingController`
definisce il confine per un futuro buffer circolare e per segmenti pre/post
evento, ma con `recording.enabled: false` non avvia backend o thread.

## Fase 9 - metriche, health reporting e benchmark

Le metriche di ogni camera sono raccolte in snapshot thread-safe e bounded.
Comprendono stream FPS, decoded FPS, sampled FPS, person/face detection FPS,
frame persi, reconnect, coda, latenza di pipeline, track attivi, rifiuti di
qualita' facciale, tentativi di recognition ed eventi generati. I contatori
sono cumulativi dall'avvio o dall'ultimo reset; i rate usano un clock monotono
e non conservano una lista illimitata di campioni.

Quando viene eseguito il runner live con person detection e, separatamente,
con la pipeline face opt-in, il report periodico
include una riga per-camera:

```text
[huawei_p30] stream: 24.80 FPS | decoded: 24.10 FPS | sampled: 2.00 FPS |
person detector: 2.00 FPS | face detector: n/d | queue: 0 | dropped: 127 |
reconnects: 0 | latency: 142.00 ms | active tracks: 1 | face rejects: 0 |
recognition attempts: 0 | events: 0
```

Il benchmark offline predefinito usa componenti fake deterministici e non
richiede Huawei, stream, GPU, pesi o runtime ONNX:

```powershell
python scripts\benchmark.py
python scripts\benchmark.py --iterations 500 --json
```

Misura separatamente person detection, face detection, embedding, matching e
pipeline, oltre a CPU, RAM e stato GPU/VRAM. I risultati indicano sempre se il
numero e' misurato su componenti simulati, misurato su componenti reali oppure
non disponibile. Il benchmark usa una directory temporanea locale e non
salva dati biometrici permanenti.

Per una prova reale esplicita si puo' usare un'immagine locale oppure una
configurazione con modelli e camera abilitati:

```powershell
python scripts\benchmark.py --mode real --image .\frame.jpg
python scripts\benchmark.py --mode real --config config\config.local.yaml --camera-id huawei_p30
```

Il benchmark dedicato YOLO26 richiede almeno 100 iterazioni temporizzate e
include load/export, warm-up, media, p50, p95, FPS, CPU, RAM, precisione e
device reale:

```powershell
python scripts\benchmark_person_detection.py --iterations 100 --warmup 10 --json
```

Esegue YOLO26s PyTorch CPU, OpenVINO CPU FP16 e OpenVINO GPU FP16. Utilizzo e
memoria GPU restano `null` con motivazione se OpenVINO o un tool Intel locale non
forniscono una telemetria affidabile.

La seconda forma apre un solo frame dello stream e lo classifica come
`real_stream`; richiede runtime, modelli, directory `persons/` compatibili e
stream LAN disponibili. L'output del benchmark non dimostra supporto a un
numero specifico di camere.

## Fase 10 - runtime multi-camera isolato

Il runner di person detection avvia tutte le camere abilitate presenti nel file
YAML quando `--camera-id` non viene specificato:

```powershell
python scripts\run_person_detection.py --config config\config.local.yaml --duration 60
```

La configurazione di esempio contiene `huawei_p30` e `cam_2`, con URL separati
e credenziali da mantenere soltanto nelle variabili d'ambiente o nella
configurazione locale ignorata da Git. Per eseguire una sola camera resta
disponibile:

```powershell
python scripts\run_person_detection.py --config config\config.local.yaml --camera-id huawei_p30 --duration 60
```

Ogni camera possiede source, worker, buffer, sampler, tracker, macchina a
stati, metriche, reconnect e stato degli errori separati. Il person detector è
caricato una sola volta e le chiamate condivise sono serializzate da un lock;
un errore su un frame viene registrato nella camera interessata e non arresta
le altre.

Il runtime espone la pipeline face opt-in per i track persona attivi. Face
detection, embedding e recognition non vengono eseguiti quando non esiste una
persona o quando le relative sezioni sono disabilitate; i publisher eventi
restano opzionali e per-camera.

I test con `FakeVideoSource` dimostrano l'isolamento offline. Non costituiscono
una verifica di due stream RTSP reali; per quella prova servono due stream locali
raggiungibili e hardware disponibile.

## Fase 11 - motion detection, scalabilità e hardening

### Motion detection opt-in

Il filtro motion è disabilitato per default nella configurazione d'esempio:

```yaml
motion_detection:
  enabled: false
  pixel_threshold: 25
  min_changed_fraction: 0.01
  resize_width: 320
  warmup_frames: 1
```

Quando è abilitato, ogni camera mantiene un detector indipendente. Il detector
converte i frame in grayscale, li ridimensiona alla larghezza configurata e
confronta frame consecutivi con una differenza assoluta. Il primo frame e il
primo frame dopo un reconnect sono considerati movimento. Una scena statica
aggiorna il frame più recente, salta la person detection e preserva i track
attivi; un errore del filtro applica un fallback fail-open alla person
detection.

Il filtro riduce il carico soltanto quando una scena statica è un'ipotesi
accettabile. Prima di abilitarlo su una camera reale occorre validare soglia,
percentuale minima e rischio di perdere movimenti piccoli o lenti.

### Benchmark della flotta

Il benchmark di scalabilità esegue livelli sequenziali da 1 a 6. La modalità
fake è deterministica e non dimostra supporto hardware:

```powershell
python scripts\benchmark_scalability.py --mode fake --max-cameras 6
python scripts\benchmark_scalability.py --mode fake --max-cameras 6 --scenario one_person --duration 10 --warmup 2 --json
python scripts\benchmark_scalability.py --mode fake --max-cameras 6 --scenario two_persons --json
```

Gli scenari sono `none`, `one_person` e `two_persons`. Ogni livello crea nuovi
runtime. `measured` significa che tutte le camere fake previste hanno prodotto
campioni per la durata richiesta; in modalità real significa che tutte le
camere configurate hanno funzionato realmente per quella durata. Un livello
incompleto è `unavailable` e non contribuisce a una dichiarazione di supporto.

La modalità real usa le prime camere abilitate con URL validi nel file YAML e
richiede modello ONNX, runtime, URL LAN raggiungibili e camere disponibili:

```powershell
python scripts\benchmark_scalability.py --mode real --config config\config.local.yaml --max-cameras 6 --duration 60 --warmup 5 --json
```

Il report include CPU, RAM, GPU/VRAM quando disponibili, FPS stream/decoded,
sampled, person e face, frame persi, coda massima, latenza, reconnect e stato
per camera. I tempi e i contatori face/landmark/alignment/embedding/matching
sono esposti dalla telemetria quando la pipeline è attiva.

### Calibrazione della soglia di recognition

La calibrazione usa solo score già raccolti e non genera dati biometrici. CSV e
JSONL devono contenere i campi obbligatori `label` e `score`; `label` può essere
solo `genuine` o `impostor`, mentre `score` deve essere finito e nell'intervallo
della similarità coseno `[-1, 1]`.

Esempio CSV:

```csv
label,score
genuine,0.91
genuine,0.86
impostor,0.22
impostor,0.48
```

Esempio JSONL:

```jsonl
{"label":"genuine","score":0.91}
{"label":"impostor","score":0.22}
```

Eseguire il tool con una soglia FAR obiettivo oppure ottenere soltanto il
suggerimento EER-like:

```powershell
python scripts\calibrate_recognition.py --input .\scores.csv --target-far 0.01
python scripts\calibrate_recognition.py --input .\scores.jsonl --format jsonl --json
```

Il report calcola `FAR = impostor accettati / impostor totali` e
`FRR = genuine rifiutati / genuine totali` per ogni soglia candidata. Con
`--target-far` seleziona la soglia più bassa compatibile con il limite; senza
il parametro mostra una soglia EER-like marcata soltanto come suggerimento.
Il comando non modifica mai `config.local.yaml`. La calibrazione va rifatta
per ogni modello, camera e condizioni di luce diverse.

### Controllo ambiente e report finale

Il controllo hardening locale verifica redazione di URL/password, regole Git
per configurazioni locali, directory scrivibili, database persone vuoto,
componenti disabilitati, compatibilità degli embedding, isolamento di errori
storage/snapshot, camera offline e shutdown dei thread:

```powershell
python scripts\check_environment.py --config config\config.example.yaml --hardening
python scripts\check_environment.py --config config\config.example.yaml --hardening --json
```

Gli stati strutturati sono `PASS`, `INFO`, `DEFERRED` e `FAIL`. `PASS` indica
un probe completato; `INFO` è informativo; `DEFERRED` richiede una risorsa
esterna, come uno stream LAN; `FAIL` rende non riuscito il controllo.

Il report della Fase 11 può raccogliere un benchmark fake e i probe hardening
senza modificare la configurazione:

```powershell
python scripts\report_phase11.py --run-fake --run-hardening --max-cameras 6 --duration 5 --warmup 1 --pytest-passed 170 --compileall-passed --output reports\phase_11_report.md
```

Il file generato contiene sempre le sezioni `IMPLEMENTATO`, `TESTATO
AUTOMATICAMENTE`, `TESTATO SU HARDWARE REALE`, `NON TESTABILE NELL'AMBIENTE
ATTUALE`, `LIMITI MISURATI`, `CONFIGURAZIONE CONSIGLIATA` e `PROBLEMI NOTI`.
Il benchmark fake viene marcato come simulazione; nessun livello reale,
supporto stabile a N camere o capacità GPU viene dichiarato senza evidenza
completa su hardware reale.
