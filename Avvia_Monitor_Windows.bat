@echo off
setlocal EnableExtensions

rem Launcher for the native WPF Windows monitor.
rem The script is rooted at its own location so Explorer and a terminal use
rem the same repository-relative backend and configuration paths.
cd /d "%~dp0"

rem Keep the existing Python help/diagnostic contract available to scripts.
set "SHOW_HELP="
for %%A in (%*) do if /I "%%~A"=="--help" set "SHOW_HELP=1"
if defined SHOW_HELP goto run_python

set "DOTNET_EXE=C:\Program Files\dotnet\dotnet.exe"
if exist "%DOTNET_EXE%" goto build_and_run_wpf

set "DOTNET_EXE=dotnet"
where dotnet >nul 2>&1
if not errorlevel 1 goto build_and_run_wpf

echo INFO: WPF/.NET non trovato; uso la GUI Python di fallback.
goto run_python

:run_wpf
"%WPF_EXE%" %*
set "EXIT_CODE=%ERRORLEVEL%"
goto finish

:build_and_run_wpf
echo INFO: ricostruzione WPF dal sorgente corrente...
"%DOTNET_EXE%" build app_windows_wpf\LocalSecurityMonitor.Wpf.csproj -c Release -t:Rebuild --nologo --verbosity minimal
if errorlevel 1 (
    set "EXIT_CODE=%ERRORLEVEL%"
    echo ERROR: build WPF fallita; nessun binario precedente verra' eseguito.
    goto finish
)
set "WPF_EXE=%CD%\app_windows_wpf\bin\Release\net8.0-windows\LocalSecurityMonitor.Wpf.exe"
if not exist "%WPF_EXE%" (
    set "EXIT_CODE=1"
    echo ERROR: eseguibile WPF non prodotto dalla build.
    goto finish
)
"%WPF_EXE%" %*
set "EXIT_CODE=%ERRORLEVEL%"
goto finish

:run_python
set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
set "PYTHON_ARGS="

if exist "%PYTHON_EXE%" goto run_python_entrypoint

where py >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=py"
    set "PYTHON_ARGS=-3"
    goto run_python_entrypoint
)

where python >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=python"
    goto run_python_entrypoint
)

echo ERROR: Python 3.11-3.13 was not found.
echo Create .venv or install Python and add it to PATH.
pause
exit /b 1

:run_python_entrypoint
"%PYTHON_EXE%" %PYTHON_ARGS% -m app_windows.main %*
set "EXIT_CODE=%ERRORLEVEL%"

:finish
if not "%EXIT_CODE%"=="0" (
    echo.
    echo Monitor exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
