param(
    [Parameter(Mandatory = $true)]
    [int]$AppPid,
    [string]$Executable,
    [string]$ConfigPath,
    [switch]$VerifyReopen,
    [switch]$ExerciseRemoval,
    [string]$OutputDirectory = "artifacts\wpf-ui"
)

$ErrorActionPreference = "Continue"
$WinApp = "C:\Users\rober\AppData\Local\Microsoft\WindowsApps\winapp.exe"
$script:CurrentPid = $AppPid
$script:Pass = 0
$script:Fail = 0
$script:Results = @()

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$nativeSource = 'using System; using System.Runtime.InteropServices; public static class WpfUiSmokeNative { [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags); }'
$nativeType = [System.Management.Automation.PSTypeName]::new("WpfUiSmokeNative")
if ($null -eq $nativeType.Type) {
    Add-Type $nativeSource
}

function Invoke-Ui {
    param([string[]]$Arguments)
    & $WinApp ui @Arguments 2>&1
}

function Assert-Ui {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    $output = Invoke-Ui $Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "${Name}: $($output -join ' ')"
    }
    return $output
}

function Get-MainWindowHandle {
    $windows = (Assert-Ui "list windows" @("list-windows", "-a", "$script:CurrentPid", "--json") -join "`n") | ConvertFrom-Json
    $window = @($windows | Where-Object { $_.title -notlike "PopupHost*" -and $null -ne $_.hwnd } | Select-Object -First 1)
    if ($window.Count -eq 0) {
        throw "Main WPF window non trovato"
    }
    return [IntPtr]$window[0].hwnd
}

function Resize-MainWindowAndCapture {
    param(
        [int]$Width,
        [int]$Height,
        [string]$FileName
    )
    $hwnd = Get-MainWindowHandle
    if (-not [WpfUiSmokeNative]::SetWindowPos($hwnd, [IntPtr]::Zero, 40, 40, $Width, $Height, 0)) {
        throw "Impossibile ridimensionare la finestra WPF a ${Width}x${Height}"
    }
    Start-Sleep -Milliseconds 750
    Assert-Ui "screenshot ${Width}x${Height}" @("screenshot", "-a", "$script:CurrentPid", "-o", (Join-Path $OutputDirectory $FileName)) | Out-Null
}

function Test-Ui {
    param(
        [string]$Name,
        [scriptblock]$Action
    )
    try {
        & $Action
        $script:Pass++
        $script:Results += [ordered]@{ name = $Name; status = "PASS" }
    }
    catch {
        $script:Fail++
        $script:Results += [ordered]@{ name = $Name; status = "FAIL"; detail = "$($_.Exception.Message)" }
        Write-Host "FAIL: $Name - $($_.Exception.Message)" -ForegroundColor Red
    }
}

Test-Ui "Open camera focus panel" {
    Assert-Ui "fake camera tile" @("wait-for", "huawei_p30", "-a", "$script:CurrentPid", "-t", "30000") | Out-Null
    Assert-Ui "open fake camera" @("invoke", "huawei_p30", "-a", "$script:CurrentPid") | Out-Null
}

Test-Ui "Face detector confidence slider exists" {
    Assert-Ui "FaceDetectorConfidenceSlider" @("wait-for", "FaceDetectorConfidenceSlider", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
}

Test-Ui "Detector slider maps to 1 percent" {
    Assert-Ui "set detector confidence 1" @("set-value", "FaceDetectorConfidenceSlider", "1", "-a", "$script:CurrentPid") | Out-Null
    Assert-Ui "detector value 1%" @("wait-for", "FaceDetectorConfidenceValue", "-a", "$script:CurrentPid", "--value", "1%", "-t", "3000") | Out-Null
}

Test-Ui "Detector slider maps to 100 percent" {
    Assert-Ui "set detector confidence 100" @("set-value", "FaceDetectorConfidenceSlider", "100", "-a", "$script:CurrentPid") | Out-Null
    Assert-Ui "detector value 100%" @("wait-for", "FaceDetectorConfidenceValue", "-a", "$script:CurrentPid", "--value", "100%", "-t", "3000") | Out-Null
}

Test-Ui "Detector percentage is committed for reopen" {
    Assert-Ui "set persisted detector confidence 21" @("set-value", "FaceDetectorConfidenceSlider", "21", "-a", "$script:CurrentPid") | Out-Null
    Assert-Ui "detector value 21%" @("wait-for", "FaceDetectorConfidenceValue", "-a", "$script:CurrentPid", "--value", "21%", "-t", "3000") | Out-Null
    Assert-Ui "detector save acknowledged" @("wait-for", "FaceDetectionStatus", "-a", "$script:CurrentPid", "--value", "Configurazione detection salvata; applicazione in corso…", "-t", "5000") | Out-Null
    Assert-Ui "detection status element" @("scroll-into-view", "FaceDetectionStatus", "-a", "$script:CurrentPid") | Out-Null
    Assert-Ui "detection status visible" @("wait-for", "FaceDetectionStatus", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
}

$detectorCases = @(
    [pscustomobject]@{
        Label = "SCRFD"
        Selection = "SCRFD 2.5G KPS"
        Model = "models/face_detection/scrfd_2.5g_kps/scrfd_2.5g_bnkps.onnx"
        Backend = "onnxruntime"
    },
    [pscustomobject]@{
        Label = "Intel 0205"
        Selection = "Intel face detector 0205"
        Model = "models/face_detection/face_detection_0205_fp32/face-detection-0205.xml"
        Backend = "openvino"
    },
    [pscustomobject]@{
        Label = "YuNet"
        Selection = "YuNet 2023mar"
        Model = "models/face_detection/yunet_2023mar/face_detection_yunet_2023mar.onnx"
        Backend = "opencv_dnn"
    }
)

foreach ($detectorCase in $detectorCases) {
    Test-Ui "Face detector selection: $($detectorCase.Label)" {
        Assert-Ui "scroll detector selector" @("scroll-into-view", "FaceDetectorCombo", "-a", "$script:CurrentPid") | Out-Null
        Assert-Ui "select $($detectorCase.Label)" @("set-value", "FaceDetectorCombo", $detectorCase.Selection, "-a", "$script:CurrentPid") | Out-Null
        Assert-Ui "selected detector $($detectorCase.Label)" @("wait-for", "FaceDetectorCombo", "-a", "$script:CurrentPid", "--value", $detectorCase.Selection, "-t", "5000") | Out-Null
        Assert-Ui "canonical path $($detectorCase.Label)" @("wait-for", "FaceDetectorModel", "-a", "$script:CurrentPid", "--value", $detectorCase.Model, "-t", "5000") | Out-Null
        Assert-Ui "canonical backend $($detectorCase.Label)" @("wait-for", "FaceDetectorBackendCombo", "-a", "$script:CurrentPid", "--value", $detectorCase.Backend, "-t", "5000") | Out-Null
    }
}

Test-Ui "YuNet selection is acknowledged as saved" {
    Assert-Ui "saved detector selection" @("wait-for", "FaceDetectionStatus", "-a", "$script:CurrentPid", "--value", "Configurazione detection salvata; applicazione in corso…", "-t", "5000") | Out-Null
}

$recognizerCases = @(
    [pscustomobject]@{
        Label = "FaceNet"
        Selection = "FaceNet VGGFace2"
        Model = "models/face_embedding/facenet-20180402-vggface2.onnx"
        Backend = "onnxruntime"
    },
    [pscustomobject]@{
        Label = "OpenVINO retail-0095"
        Selection = "OpenVINO retail-0095"
        Model = "models/face_embedding/face-reidentification-retail-0095/face-reidentification-retail-0095.xml"
        Backend = "openvino"
    },
    [pscustomobject]@{
        Label = "ArcFace"
        Selection = "ArcFace WebFace600K"
        Model = "models/face_embedding/arcface-resnet50-webface600k.onnx"
        Backend = "onnxruntime"
    }
)

foreach ($recognizerCase in $recognizerCases) {
    Test-Ui "Face recognizer selection: $($recognizerCase.Label)" {
        Assert-Ui "scroll recognizer selector" @("scroll-into-view", "FaceRecognizerCombo", "-a", "$script:CurrentPid") | Out-Null
        Assert-Ui "select $($recognizerCase.Label)" @("set-value", "FaceRecognizerCombo", $recognizerCase.Selection, "-a", "$script:CurrentPid") | Out-Null
        Assert-Ui "selected recognizer $($recognizerCase.Label)" @("wait-for", "FaceRecognizerCombo", "-a", "$script:CurrentPid", "--value", $recognizerCase.Selection, "-t", "5000") | Out-Null
        Assert-Ui "canonical path $($recognizerCase.Label)" @("wait-for", "FaceRecognizerModel", "-a", "$script:CurrentPid", "--value", $recognizerCase.Model, "-t", "5000") | Out-Null
        Assert-Ui "canonical backend $($recognizerCase.Label)" @("wait-for", "FaceRecognizerBackendCombo", "-a", "$script:CurrentPid", "--value", $recognizerCase.Backend, "-t", "5000") | Out-Null
    }
}

Test-Ui "ArcFace selection is acknowledged as saved" {
    Assert-Ui "saved recognizer selection" @("wait-for", "FaceStatus", "-a", "$script:CurrentPid", "--value", "Configurazione recognition salvata; applicazione in corso…", "-t", "5000") | Out-Null
}

Test-Ui "Sampling labels and landmark selector are visible" {
    Assert-Ui "sampling detector label" @("wait-for", "Sampling detector (FPS)", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
    Assert-Ui "sampling recognition label" @("wait-for", "Sampling recognition (FPS)", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
    Assert-Ui "landmarker selector" @("wait-for", "FaceLandmarkerCombo", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
}

Test-Ui "Threshold and confirmation controls are separated" {
    Assert-Ui "threshold slider" @("wait-for", "FaceRecognitionThresholdSlider", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
    Assert-Ui "threshold toggle" @("wait-for", "FaceRecognitionThresholdEnabled", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
    Assert-Ui "confirmations" @("wait-for", "FaceRecognitionConfirmations", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
    Assert-Ui "confirmation window" @("wait-for", "FaceRecognitionWindow", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
    Assert-Ui "recognition FPS" @("wait-for", "FaceRecognitionFps", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
}

Test-Ui "Gallery list and source state are visible" {
    Assert-Ui "gallery summary" @("wait-for", "FaceGallerySummary", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
    Assert-Ui "people count" @("wait-for", "FaceGalleryPeopleCount", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
    Assert-Ui "active gallery count" @("wait-for", "FaceGalleryActiveCount", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
    Assert-Ui "active record count" @("wait-for", "FaceGalleryRecordCount", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
    Assert-Ui "gallery list" @("wait-for", "FaceEnrollmentList", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
    Assert-Ui "refresh people" @("wait-for", "RefreshFaceGallery", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
    Assert-Ui "activate people" @("wait-for", "ActivateFacePeople", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
    Assert-Ui "delete person" @("wait-for", "RemoveFacePerson", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
    Assert-Ui "select gallery root" @("wait-for", "SelectFaceGalleryRoot", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
}

Test-Ui "Face gallery labels are complete" {
    Assert-Ui "refresh label" @("wait-for", "Aggiorna persone", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
    Assert-Ui "activate label" @("wait-for", "Attiva persone", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
    Assert-Ui "delete label" @("wait-for", "Elimina persona", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
    Assert-Ui "root label" @("wait-for", "Seleziona cartella madre delle persone", "-a", "$script:CurrentPid", "-t", "3000") | Out-Null
}

$listPayload = Invoke-Ui @("inspect", "FaceEnrollmentList", "-a", "$script:CurrentPid", "--json") -join "`n"
if ($LASTEXITCODE -eq 0) {
    try {
        $list = $listPayload | ConvertFrom-Json
        $listElement = @($list.windows[0].elements)[0]
        $activeItem = @($listElement.children | Where-Object {
            @($_.children | Where-Object { $_.name -eq "ATTIVA" }).Count -gt 0
        }) | Select-Object -First 1
        if ($null -ne $activeItem) {
            Test-Ui "Gallery removal starts disabled without selection" {
                $properties = (Assert-Ui "initial remove button properties" @("get-property", "RemoveFacePerson", "-a", "$script:CurrentPid", "--json") -join "`n") | ConvertFrom-Json
                if ($properties.properties.IsEnabled -ne "False") {
                    throw "Elimina persona deve essere disabilitata senza selezione"
                }
            }
            Test-Ui "Gallery selection enables removal" {
                Assert-Ui "scroll selected person" @("scroll-into-view", $activeItem.selector, "-a", "$script:CurrentPid") | Out-Null
                Assert-Ui "select person" @("click", $activeItem.selector, "-a", "$script:CurrentPid") | Out-Null
                $properties = (Assert-Ui "remove button properties" @("get-property", "RemoveFacePerson", "-a", "$script:CurrentPid", "--json") -join "`n") | ConvertFrom-Json
                if ($properties.properties.IsEnabled -ne "True") {
                    throw "Rimuovi record attivo non abilitato per una riga ATTIVA"
                }
            }
            if ($ExerciseRemoval) {
                Test-Ui "Selected active record can be removed" {
                    Assert-Ui "remove selected person" @("invoke", "RemoveFacePerson", "-a", "$script:CurrentPid") | Out-Null
                    Assert-Ui "gallery refresh after remove" @("wait-for", "FaceGallerySummary", "-a", "$script:CurrentPid", "-t", "5000") | Out-Null
                    Assert-Ui "reactivate source enrollment" @("invoke", "ActivateFacePeople", "-a", "$script:CurrentPid") | Out-Null
                }
            }
        }
        else {
            $script:Results += [ordered]@{ name = "Gallery selection enables removal"; status = "SKIP"; detail = "Nessun record ATTIVA presente" }
        }
    }
    catch {
        $script:Fail++
        $script:Results += [ordered]@{ name = "Gallery row inspection"; status = "FAIL"; detail = $_.Exception.Message }
    }
}

Test-Ui "Current-size screenshot" {
    Assert-Ui "screenshot current" @("screenshot", "-a", "$script:CurrentPid", "-o", (Join-Path $OutputDirectory "wpf-face-current.png")) | Out-Null
}

Test-Ui "Responsive gallery screenshots" {
    Resize-MainWindowAndCapture -Width 1080 -Height 760 -FileName "wpf-face-narrow.png"
    Resize-MainWindowAndCapture -Width 1440 -Height 900 -FileName "wpf-face-current-size.png"
    Resize-MainWindowAndCapture -Width 1600 -Height 900 -FileName "wpf-face-wide.png"
}

if ($VerifyReopen) {
    if ([string]::IsNullOrWhiteSpace($Executable) -or [string]::IsNullOrWhiteSpace($ConfigPath)) {
        throw "-VerifyReopen richiede -Executable e -ConfigPath"
    }
    Test-Ui "Configuration survives WPF reopen" {
        Assert-Ui "close first WPF process" @("invoke", "Close", "-a", "$script:CurrentPid") | Out-Null
        Start-Sleep -Seconds 2
        $newProcess = Start-Process -FilePath $Executable -ArgumentList @("--fake-cameras", "--config", $ConfigPath) -WorkingDirectory (Get-Location) -PassThru
        $script:CurrentPid = $newProcess.Id
        Assert-Ui "reopened camera" @("wait-for", "huawei_p30", "-a", "$script:CurrentPid", "-t", "30000") | Out-Null
        Assert-Ui "open camera" @("invoke", "huawei_p30", "-a", "$script:CurrentPid") | Out-Null
        Assert-Ui "reopened detector percentage" @("wait-for", "FaceDetectorConfidenceValue", "-a", "$script:CurrentPid", "--value", "21%", "-t", "30000") | Out-Null
        Assert-Ui "reopened detector selection" @("wait-for", "FaceDetectorCombo", "-a", "$script:CurrentPid", "--value", "YuNet 2023mar", "-t", "30000") | Out-Null
        Assert-Ui "reopened detector path" @("wait-for", "FaceDetectorModel", "-a", "$script:CurrentPid", "--value", "models/face_detection/yunet_2023mar/face_detection_yunet_2023mar.onnx", "-t", "30000") | Out-Null
        Assert-Ui "reopened detector backend" @("wait-for", "FaceDetectorBackendCombo", "-a", "$script:CurrentPid", "--value", "opencv_dnn", "-t", "30000") | Out-Null
        Assert-Ui "reopened recognizer selection" @("wait-for", "FaceRecognizerCombo", "-a", "$script:CurrentPid", "--value", "ArcFace WebFace600K", "-t", "30000") | Out-Null
        Assert-Ui "reopened recognizer path" @("wait-for", "FaceRecognizerModel", "-a", "$script:CurrentPid", "--value", "models/face_embedding/arcface-resnet50-webface600k.onnx", "-t", "30000") | Out-Null
        Assert-Ui "reopened recognizer backend" @("wait-for", "FaceRecognizerBackendCombo", "-a", "$script:CurrentPid", "--value", "onnxruntime", "-t", "30000") | Out-Null
    }
}

Assert-Ui "final screenshot" @("screenshot", "-a", "$script:CurrentPid", "-o", "$OutputDirectory\wpf-face-final.png") | Out-Null
if ($VerifyReopen) {
    Assert-Ui "close reopened WPF process" @("invoke", "Close", "-a", "$script:CurrentPid") | Out-Null
}
$script:Results | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -Path "$OutputDirectory\wpf-face-ui-results.json"
Write-Host "`nPassed: $script:Pass | Failed: $script:Fail"
$script:Results | Where-Object { $_.status -eq "FAIL" } | ForEach-Object {
    Write-Host "  FAIL: $($_.name) - $($_.detail)" -ForegroundColor Red
}
if ($script:Fail -gt 0) { exit 1 }
