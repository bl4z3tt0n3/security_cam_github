# Report Fase 11

## IMPLEMENTATO

- Motion detection opt-in, indipendente per camera, fail-open e con reset dopo reconnect.
- Runtime multi-camera con buffer bounded, tracking preservato su scena statica e metriche motion.
- Benchmark di scalabilità fake/real con livelli sequenziali, risorse, code, reconnect e stato camera.
- Calibrazione offline da CSV/JSONL con distribuzioni, FAR, FRR e soglie candidate.
- Hardening locale strutturato con redazione segreti, probe storage, compatibilità embedding, isolamento e shutdown.
- Nessuna modifica automatica a `config.local.yaml` e nessun wiring live face-analysis -> recognition -> eventi.

## TESTATO AUTOMATICAMENTE

- Pytest: `170 passed`.
- `compileall`: completato con esito positivo.
- Hardening locale: stato aggregato `PASS` (PASS=9, INFO=2, DEFERRED=1, FAIL=0).
- `Secrets ignore rules`: `PASS` — local config and .env patterns present.
- `Secret redaction`: `PASS` — url=rtsp://admin:***@camera.local:8554/live?password=*** password=*** token=***.
- `Storage write checks`: `PASS` — 7/7 directories writable.
- `Persons database`: `INFO` — empty; recognition safely returns UNKNOWN.
- `Disabled components`: `PASS` — no model adapters constructed for: person detection, face detection, recognition.
- `Embedding compatibility`: `PASS` — synthetic incompatible record rejected before matching.
- `Configured embedding records`: `INFO` — recognition disabled; no live records loaded.
- `Storage failure isolation`: `PASS` — event metadata persisted while snapshot failure stayed isolated.
- `Offline camera isolation`: `PASS` — offline camera failed independently; healthy camera received frames.
- `Graceful shutdown`: `PASS` — runtime, sampler and worker stopped without residual project threads.
- `Public API exposure`: `PASS` — no web server or port-forwarding component configured.
- `Live camera isolation`: `DEFERRED` — requires two concrete LAN streams; offline probe is reported separately.
- Benchmark scalabilità: `simulated` con 6 livelli richiesti.

## TESTATO SU HARDWARE REALE

- Nessun livello `real` con stato `measured` è presente nelle evidenze fornite.
- Il benchmark fake/simulato non costituisce prova di supporto hardware.

## NON TESTABILE NELL'AMBIENTE ATTUALE

- URL LAN/RTSP concreti e modelli ONNX reali non sono verificati da questo report.
- GPU/VRAM resta non disponibile quando il controllo ambiente non fornisce uno stato GPU `available`.
- La concorrenza della face pipeline è `n/d`: il wiring live resta fuori scope.

## LIMITI MISURATI

- Livello 1: `measured` (simulated; simulazione se execution=`simulated`): fake_1: sampled=9.99 FPS, person=9.99 FPS, face=0.00 FPS, dropped=0, queue_max=1, face_concurrency=n/d.
- Livello 2: `measured` (simulated; simulazione se execution=`simulated`): fake_1: sampled=9.95 FPS, person=9.95 FPS, face=0.00 FPS, dropped=0, queue_max=1, face_concurrency=n/d; fake_2: sampled=9.96 FPS, person=9.95 FPS, face=0.00 FPS, dropped=0, queue_max=1, face_concurrency=n/d.
- Livello 3: `measured` (simulated; simulazione se execution=`simulated`): fake_1: sampled=9.96 FPS, person=9.95 FPS, face=0.00 FPS, dropped=0, queue_max=1, face_concurrency=n/d; fake_2: sampled=9.96 FPS, person=9.95 FPS, face=0.00 FPS, dropped=0, queue_max=1, face_concurrency=n/d; fake_3: sampled=9.96 FPS, person=9.95 FPS, face=0.00 FPS, dropped=0, queue_max=1, face_concurrency=n/d.
- Livello 4: `measured` (simulated; simulazione se execution=`simulated`): fake_1: sampled=9.95 FPS, person=9.95 FPS, face=0.00 FPS, dropped=0, queue_max=1, face_concurrency=n/d; fake_2: sampled=9.95 FPS, person=9.95 FPS, face=0.00 FPS, dropped=0, queue_max=1, face_concurrency=n/d; fake_3: sampled=9.95 FPS, person=9.95 FPS, face=0.00 FPS, dropped=0, queue_max=1, face_concurrency=n/d; fake_4: sampled=9.95 FPS, person=9.95 FPS, face=0.00 FPS, dropped=0, queue_max=1, face_concurrency=n/d.
- Livello 5: `measured` (simulated; simulazione se execution=`simulated`): fake_1: sampled=9.94 FPS, person=9.94 FPS, face=0.00 FPS, dropped=0, queue_max=1, face_concurrency=n/d; fake_2: sampled=9.94 FPS, person=9.94 FPS, face=0.00 FPS, dropped=0, queue_max=1, face_concurrency=n/d; fake_3: sampled=9.94 FPS, person=9.94 FPS, face=0.00 FPS, dropped=0, queue_max=1, face_concurrency=n/d; fake_4: sampled=9.94 FPS, person=9.94 FPS, face=0.00 FPS, dropped=0, queue_max=1, face_concurrency=n/d; fake_5: sampled=9.94 FPS, person=9.94 FPS, face=0.00 FPS, dropped=0, queue_max=1, face_concurrency=n/d.
- Livello 6: `measured` (simulated; simulazione se execution=`simulated`): fake_1: sampled=9.98 FPS, person=9.98 FPS, face=0.00 FPS, dropped=0, queue_max=1, face_concurrency=n/d; fake_2: sampled=9.98 FPS, person=9.98 FPS, face=0.00 FPS, dropped=0, queue_max=1, face_concurrency=n/d; fake_3: sampled=9.98 FPS, person=9.98 FPS, face=0.00 FPS, dropped=0, queue_max=1, face_concurrency=n/d; fake_4: sampled=9.98 FPS, person=9.98 FPS, face=0.00 FPS, dropped=2, queue_max=1, face_concurrency=n/d; fake_5: sampled=9.98 FPS, person=9.98 FPS, face=0.00 FPS, dropped=0, queue_max=1, face_concurrency=n/d; fake_6: sampled=9.98 FPS, person=9.98 FPS, face=0.00 FPS, dropped=0, queue_max=1, face_concurrency=n/d.
- Non vengono applicate soglie universali per CPU, FPS o numero di camere.

## CONFIGURAZIONE CONSIGLIATA

- Mantenere `motion_detection.enabled: false` finché soglia e percentuale non sono validate sulla scena reale.
- Usare URL e segreti soltanto in `config.local.yaml` o variabili d’ambiente ignorate da Git.
- Abilitare person/face/recognition solo dopo aver installato runtime e modelli locali compatibili.
- Rifare la calibrazione per ogni modello, camera e condizioni di luce; usare la soglia come suggerimento da validare.

## PROBLEMI NOTI

- Le evidenze raccolte non attestano inferenza ONNX, connettività RTSP o supporto GPU reali.
- La pipeline facciale live e la relativa concorrenza restano non implementate (`n/d`).
- Il benchmark fake misura soltanto la riproducibilità del runtime simulato.
