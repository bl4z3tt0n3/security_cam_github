using System.IO;

namespace LocalSecurityMonitor.Wpf.Models;

public sealed record DetectionModelOption(string Label, string Value);

/// <summary>
/// Lists only models that the selected Windows person-detection backend can use.
/// </summary>
public static class DetectionModelCatalog
{
    private static readonly string[] YoloeModelNames =
    {
        "yoloe-26n-seg.pt",
        "yoloe-26s-seg.pt",
        "yoloe-26l-seg.pt",
    };

    private static readonly string[] OpenVinoModelNames =
    {
        "yolo26s.pt",
        "yolo26n.pt",
    };

    public static IReadOnlyList<DetectionModelOption> Discover(
        string backend,
        string? configuredModel,
        string repoRoot)
    {
        var root = Path.GetFullPath(repoRoot);
        return backend.Trim().ToLowerInvariant() switch
        {
            "openvino" => DiscoverOpenVino(root, configuredModel),
            "yoloe" => DiscoverYoloe(root, configuredModel),
            "onnx" => DiscoverOnnx(root, configuredModel),
            "fake" => new[] { new DetectionModelOption("Nessun modello (fake/offline)", string.Empty) },
            _ => Array.Empty<DetectionModelOption>(),
        };
    }

    private static IReadOnlyList<DetectionModelOption> DiscoverYoloe(
        string root,
        string? configuredModel)
    {
        var candidates = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        AddConfiguredCheckpoint(candidates, root, configuredModel, YoloeModelNames);

        foreach (var path in EnumerateFiles(root, "models", ".pt"))
        {
            if (ContainsName(YoloeModelNames, Path.GetFileName(path)))
            {
                AddCandidate(candidates, root, path, DisplayPath(root, path));
            }
        }

        return candidates
            .OrderBy(item => item.Key, StringComparer.OrdinalIgnoreCase)
            .Select(item => CreateLocalOption(root, item.Key, item.Value, downloadOnFirstUse: false))
            .ToArray();
    }

    private static IReadOnlyList<DetectionModelOption> DiscoverOpenVino(
        string root,
        string? configuredModel)
    {
        var candidates = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var name in OpenVinoModelNames)
        {
            var display = $"models/{name}";
            candidates[display] = display;
        }

        if (TryResolvePath(root, configuredModel, out var configuredPath)
            && (ContainsName(OpenVinoModelNames, Path.GetFileName(configuredPath))
                || Directory.Exists(configuredPath)
                || string.Equals(Path.GetExtension(configuredPath), ".xml", StringComparison.OrdinalIgnoreCase)))
        {
            candidates[DisplayPath(root, configuredPath)] = configuredModel!;
        }

        foreach (var path in EnumerateFiles(root, "models", null))
        {
            if (string.Equals(Path.GetExtension(path), ".pt", StringComparison.OrdinalIgnoreCase)
                && ContainsName(OpenVinoModelNames, Path.GetFileName(path)))
            {
                AddCandidate(candidates, root, path, DisplayPath(root, path));
            }
            else if (string.Equals(Path.GetExtension(path), ".xml", StringComparison.OrdinalIgnoreCase))
            {
                var directory = Path.GetDirectoryName(path);
                if (!string.IsNullOrWhiteSpace(directory))
                {
                    var display = DisplayPath(root, directory);
                    AddCandidate(candidates, root, directory, display);
                }
            }
        }

        return candidates
            .OrderBy(item => item.Key, StringComparer.OrdinalIgnoreCase)
            .Select(item => CreateLocalOption(
                root,
                item.Key,
                item.Value,
                downloadOnFirstUse: ContainsName(OpenVinoModelNames, Path.GetFileName(item.Value))))
            .ToArray();
    }

    private static IReadOnlyList<DetectionModelOption> DiscoverOnnx(
        string root,
        string? configuredModel)
    {
        var candidates = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        if (TryResolvePath(root, configuredModel, out var configuredPath)
            && string.Equals(Path.GetExtension(configuredPath), ".onnx", StringComparison.OrdinalIgnoreCase))
        {
            candidates[DisplayPath(root, configuredPath)] = configuredModel!;
        }

        foreach (var path in EnumerateFiles(root, "models", ".onnx"))
        {
            AddCandidate(candidates, root, path, DisplayPath(root, path));
        }

        return candidates
            .OrderBy(item => item.Key, StringComparer.OrdinalIgnoreCase)
            .Select(item => CreateLocalOption(root, item.Key, item.Value, downloadOnFirstUse: false))
            .ToArray();
    }

    private static void AddConfiguredCheckpoint(
        IDictionary<string, string> candidates,
        string root,
        string? configuredModel,
        IReadOnlyCollection<string> supportedNames)
    {
        if (!TryResolvePath(root, configuredModel, out var configuredPath)
            || !string.Equals(Path.GetExtension(configuredPath), ".pt", StringComparison.OrdinalIgnoreCase)
            || !ContainsName(supportedNames, Path.GetFileName(configuredPath)))
        {
            return;
        }

        candidates[DisplayPath(root, configuredPath)] = configuredModel!;
    }

    private static void AddCandidate(
        IDictionary<string, string> candidates,
        string root,
        string path,
        string value)
    {
        var display = DisplayPath(root, path);
        if (!candidates.ContainsKey(display))
        {
            candidates[display] = value;
        }
    }

    private static DetectionModelOption CreateLocalOption(
        string root,
        string display,
        string value,
        bool downloadOnFirstUse)
    {
        var exists = TryResolvePath(root, value, out var path)
            && (File.Exists(path) || Directory.Exists(path));
        var label = display;
        if (!exists)
        {
            label += downloadOnFirstUse
                ? " (download al primo uso)"
                : " (mancante)";
        }
        return new DetectionModelOption(label, value);
    }

    private static IEnumerable<string> EnumerateFiles(
        string root,
        string relativeDirectory,
        string? extension)
    {
        var directory = Path.Combine(root, relativeDirectory);
        if (!Directory.Exists(directory))
        {
            yield break;
        }

        string[] files;
        try
        {
            files = Directory
                .EnumerateFiles(directory, "*", SearchOption.AllDirectories)
                .ToArray();
        }
        catch (IOException)
        {
            yield break;
        }
        catch (UnauthorizedAccessException)
        {
            yield break;
        }

        foreach (var path in files)
        {
            if (extension is null
                || string.Equals(Path.GetExtension(path), extension, StringComparison.OrdinalIgnoreCase))
            {
                yield return path;
            }
        }
    }

    private static bool TryResolvePath(string root, string? value, out string path)
    {
        path = string.Empty;
        if (string.IsNullOrWhiteSpace(value))
        {
            return false;
        }

        try
        {
            path = Path.GetFullPath(Path.IsPathRooted(value)
                ? value
                : Path.Combine(root, value));
            return true;
        }
        catch (ArgumentException)
        {
            return false;
        }
        catch (NotSupportedException)
        {
            return false;
        }
    }

    private static string DisplayPath(string root, string path)
    {
        try
        {
            var fullRoot = Path.GetFullPath(root)
                .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            var fullPath = Path.GetFullPath(path);
            var relative = Path.GetRelativePath(fullRoot, fullPath);
            if (relative == ".."
                || relative.StartsWith($"..{Path.DirectorySeparatorChar}", StringComparison.Ordinal)
                || Path.IsPathRooted(relative))
            {
                return fullPath;
            }
            return relative.Replace(Path.DirectorySeparatorChar, '/');
        }
        catch (ArgumentException)
        {
            return path.Replace('\\', '/');
        }
    }

    private static bool ContainsName(IEnumerable<string> names, string? candidate) =>
        candidate is not null
        && names.Any(name => string.Equals(name, candidate, StringComparison.OrdinalIgnoreCase));
}
