# LocalCam Android - Camera2 to RTSP/TCP

L'app Android usa il package originale `com.localsecuritycam.android` e
trasforma il telefono in una sorgente RTSP locale per il monitor Windows. Non e'
una demo CameraX: la pipeline reale e'

```text
Camera2 -> VideoFrameRenderer -> MediaCodec H.264 -> RTP -> RTSP/TCP
```

`CameraStreamingService` e' un foreground service ed e' l'unico proprietario di
Camera2, encoder, server RTSP e socket. La UI contiene soltanto una `SurfaceView`
per la preview e si collega al servizio tramite binder/snapshot; una rotazione o
la compressione del pannello ricreano solo la UI e non riavviano lo stream.

## Stato visibile

`ServiceSnapshot` e' la sorgente di verita'. Il presenter genera i controlli:

| Stato del servizio | Pulsante |
| --- | --- |
| `STOPPED` | `Start Stream` |
| `STARTING` / `WAITING_NETWORK` | `Stop Stream` (annulla) |
| `STREAMING` | `Stop Stream` |
| `STOPPING` | `Arresto...` disabilitato |
| `ERROR` | `Riprova` |

Un server RTSP attivo senza client e' normale: la UI mostra `SERVER ATTIVO` /
`Waiting for client`. Solo quando il monitor Windows si connette diventa
`CLIENT CONNESSO` / `LIVE`.

## Setup e diagnostica

Il foglio `Setup` salva lente, risoluzione, FPS, bitrate, porta, path e Basic
Auth validati. La password resta nel `CredentialStore` storico, protetta a
riposo con Android Keystore + AES-GCM; i campi sono mascherati e URL/log sono
redatti. Il foglio `Diagnostics` mostra IP, porta, stato, server/client e
metriche essenziali del servizio.

La perdita del Wi-Fi durante uno stream attivo passa a `WAITING_NETWORK` e il
servizio prova automaticamente a riprendere quando torna una LAN valida. Uno
`Stop Stream` manuale annulla questo recupero.

## Endpoint Windows

Con le impostazioni predefinite, dopo `Start Stream` il monitor Windows deve
usare:

```text
rtsp://<utente>:<password>@<IP-del-telefono>:8554/stream
```

Usare l'IP mostrato dalla diagnostica Android e inserire l'URL nella
configurazione della sola Camera 1 sul monitor Windows. Se username o password
contengono caratteri riservati per URL, codificarli prima di comporre l'URL.
Il client usa RTSP interleaved su TCP; non configurare port forwarding.

## Sicurezza del trasporto

Basic Auth e cifratura della password a riposo sono implementati. TLS, RTSPS,
SRTP, DTLS, E2EE e nuovi handshake di trasporto sono **NOT IMPLEMENTED BY
DESIGN**. Usare quindi la stessa LAN fidata e non esporre la porta RTSP a
Internet.

## Requisiti e build

- JDK 17;
- Android SDK Platform 35;
- Android 9 / API 28 o superiore;
- telefono e PC sulla stessa LAN per la prova reale.

Da questa directory:

```powershell
$env:GRADLE_USER_HOME = (Join-Path $PWD '.gradle-user-home')
$env:ANDROID_USER_HOME = (Join-Path $PWD '.android-user-home')
.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug
```

L'APK debug e' in:

```text
app/build/outputs/apk/debug/app-debug.apk
```

Per un dispositivo collegato via ADB:

```powershell
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

## Verifica reale richiesta

1. Installare l'APK su Huawei e concedere fotocamera/notifiche.
2. In `Setup`, verificare credenziali, porta `8554` e path `stream`.
3. Premere `Start Stream`; attendere `SERVER ATTIVO`.
4. Configurare Windows con l'endpoint sopra e attendere `LIVE`.
5. Provare stop/start, disconnessione e ripresa Wi-Fi, rotazione e pannello
   compresso senza riavviare il servizio.

I test unitari verificano contratti Camera2/pipeline, H.264/RTP/RTSP,
backpressure, lifecycle e mapping UI. Non dimostrano compatibilita' reale con
Huawei, MediaCodec o la LAN: senza il telefono, il risultato resta
`NOT TESTED - PHYSICAL HUAWEI REQUIRED`.
