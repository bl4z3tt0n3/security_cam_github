using System.Windows;

namespace LocalSecurityMonitor.Wpf;

public partial class App : Application
{
    private void Application_Startup(object sender, StartupEventArgs e)
    {
        var options = LaunchOptions.Parse(e.Args);
        var window = new MainWindow(options);
        MainWindow = window;
        window.Show();
    }
}

public sealed class LaunchOptions
{
    public string? ConfigPath { get; init; }
    public string? RepoRoot { get; init; }
    public bool FakeCameras { get; init; }
    public string? FakeOfflineCamera { get; init; }
    public string? FakeReconnectCamera { get; init; }

    public static LaunchOptions Parse(IReadOnlyList<string> args)
    {
        string? config = null;
        string? root = null;
        string? offline = null;
        string? reconnect = null;
        var fake = false;

        for (var index = 0; index < args.Count; index++)
        {
            switch (args[index].ToLowerInvariant())
            {
                case "--config" when index + 1 < args.Count:
                    config = args[++index];
                    break;
                case "--repo-root" when index + 1 < args.Count:
                    root = args[++index];
                    break;
                case "--fake-cameras":
                    fake = true;
                    break;
                case "--fake-offline-camera" when index + 1 < args.Count:
                    offline = args[++index];
                    fake = true;
                    break;
                case "--fake-reconnect-camera" when index + 1 < args.Count:
                    reconnect = args[++index];
                    fake = true;
                    break;
            }
        }

        return new LaunchOptions
        {
            ConfigPath = config,
            RepoRoot = root,
            FakeCameras = fake,
            FakeOfflineCamera = offline,
            FakeReconnectCamera = reconnect,
        };
    }
}
