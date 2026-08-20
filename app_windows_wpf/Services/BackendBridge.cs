using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using LocalSecurityMonitor.Wpf.Models;

namespace LocalSecurityMonitor.Wpf.Services;

public sealed record BackendBridgeOptions(
    string RepoRoot,
    string? ConfigPath,
    bool FakeCameras,
    string? FakeOfflineCamera,
    string? FakeReconnectCamera);

public sealed class BackendBridge : IAsyncDisposable
{
    private readonly BackendBridgeOptions _options;
    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = false,
    };
    private readonly SemaphoreSlim _writeGate = new(1, 1);
    private Process? _process;
    private bool _disposed;

    public BackendBridge(BackendBridgeOptions options)
    {
        _options = options;
    }

    public event Action<BridgeMessage>? MessageReceived;
    public event Action<string>? ErrorReceived;
    public event Action? ProcessExited;

    public bool IsRunning => _process is { HasExited: false };

    public Task StartAsync()
    {
        if (_process is not null)
        {
            return Task.CompletedTask;
        }

        var startInfo = CreateStartInfo();
        var process = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
        process.Exited += (_, _) => ProcessExited?.Invoke();
        if (!process.Start())
        {
            throw new InvalidOperationException("Impossibile avviare il backend Python locale.");
        }
        _process = process;
        _ = Task.Run(ReadOutputAsync);
        _ = Task.Run(ReadErrorAsync);
        return Task.CompletedTask;
    }

    public async Task SendCommandAsync(string command, IReadOnlyDictionary<string, object?> data)
    {
        var process = _process;
        if (process is null || process.HasExited || process.StandardInput is null)
        {
            throw new InvalidOperationException("Il backend locale non è in esecuzione.");
        }

        var payload = new Dictionary<string, object?>
        {
            ["command"] = command,
            ["data"] = data,
        };
        var json = JsonSerializer.Serialize(payload, _jsonOptions);
        await _writeGate.WaitAsync().ConfigureAwait(false);
        try
        {
            await process.StandardInput.WriteLineAsync(json).ConfigureAwait(false);
            await process.StandardInput.FlushAsync().ConfigureAwait(false);
        }
        finally
        {
            _writeGate.Release();
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        var process = _process;
        if (process is null)
        {
            return;
        }

        try
        {
            if (!process.HasExited)
            {
                await SendCommandAsync("shutdown", new Dictionary<string, object?>())
                    .WaitAsync(TimeSpan.FromMilliseconds(500))
                    .ConfigureAwait(false);
                await process.WaitForExitAsync().WaitAsync(TimeSpan.FromMilliseconds(1800))
                    .ConfigureAwait(false);
            }
        }
        catch
        {
            // Closing the desktop shell must not leave a child process alive.
            try
            {
                if (!process.HasExited)
                {
                    process.Kill(entireProcessTree: true);
                }
            }
            catch
            {
                // The process may have exited between the checks.
            }
        }
        process.Dispose();
        _writeGate.Dispose();
    }

    private ProcessStartInfo CreateStartInfo()
    {
        var python = FindPython(_options.RepoRoot, out var usesPyLauncher);
        var startInfo = new ProcessStartInfo
        {
            FileName = python,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8,
            WorkingDirectory = _options.RepoRoot,
        };
        startInfo.Environment["PYTHONUNBUFFERED"] = "1";
        if (usesPyLauncher)
        {
            startInfo.ArgumentList.Add("-3");
        }
        startInfo.ArgumentList.Add("-m");
        startInfo.ArgumentList.Add("app_windows.wpf_bridge");
        if (!string.IsNullOrWhiteSpace(_options.ConfigPath))
        {
            startInfo.ArgumentList.Add("--config");
            startInfo.ArgumentList.Add(_options.ConfigPath);
        }
        if (_options.FakeCameras)
        {
            startInfo.ArgumentList.Add("--fake-cameras");
        }
        if (!string.IsNullOrWhiteSpace(_options.FakeOfflineCamera))
        {
            startInfo.ArgumentList.Add("--fake-offline-camera");
            startInfo.ArgumentList.Add(_options.FakeOfflineCamera);
        }
        if (!string.IsNullOrWhiteSpace(_options.FakeReconnectCamera))
        {
            startInfo.ArgumentList.Add("--fake-reconnect-camera");
            startInfo.ArgumentList.Add(_options.FakeReconnectCamera);
        }
        return startInfo;
    }

    private static string FindPython(string repoRoot, out bool usesPyLauncher)
    {
        var venvPython = Path.Combine(repoRoot, ".venv", "Scripts", "python.exe");
        if (File.Exists(venvPython))
        {
            usesPyLauncher = false;
            return venvPython;
        }
        usesPyLauncher = true;
        return "py";
    }

    private async Task ReadOutputAsync()
    {
        var process = _process;
        if (process is null)
        {
            return;
        }
        try
        {
            while (await process.StandardOutput.ReadLineAsync().ConfigureAwait(false) is { } line)
            {
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }
                try
                {
                    using var document = JsonDocument.Parse(line);
                    var root = document.RootElement;
                    var type = root.TryGetProperty("type", out var typeValue)
                        ? typeValue.GetString() ?? string.Empty
                        : string.Empty;
                    var data = root.TryGetProperty("data", out var dataValue)
                        ? dataValue.Clone()
                        : JsonDocument.Parse("{}").RootElement.Clone();
                    MessageReceived?.Invoke(new BridgeMessage(type, data));
                }
                catch (JsonException ex)
                {
                    ErrorReceived?.Invoke($"Risposta backend non valida: {ex.Message}");
                }
            }
        }
        catch (ObjectDisposedException)
        {
        }
        catch (IOException ex)
        {
            ErrorReceived?.Invoke($"Lettura backend interrotta: {ex.Message}");
        }
    }

    private async Task ReadErrorAsync()
    {
        var process = _process;
        if (process is null)
        {
            return;
        }
        try
        {
            while (await process.StandardError.ReadLineAsync().ConfigureAwait(false) is { } line)
            {
                if (!string.IsNullOrWhiteSpace(line))
                {
                    ErrorReceived?.Invoke(line);
                }
            }
        }
        catch (ObjectDisposedException)
        {
        }
        catch (IOException ex)
        {
            ErrorReceived?.Invoke($"Log backend interrotto: {ex.Message}");
        }
    }
}
