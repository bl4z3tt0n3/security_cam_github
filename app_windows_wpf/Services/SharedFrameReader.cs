using System.IO.MemoryMappedFiles;
using LocalSecurityMonitor.Wpf.Models;

namespace LocalSecurityMonitor.Wpf.Services;

public sealed record SharedBgrFrame(byte[] Pixels, int Width, int Height, int Stride);

public sealed class SharedFrameReader : IDisposable
{
    private const int HeaderSize = 40;
    private const uint Version = 1;
    private static readonly byte[] Magic = "LSCF"u8.ToArray();

    private MemoryMappedFile? _map;
    private MemoryMappedViewAccessor? _view;
    private string? _mapName;
    private byte[]? _pixels;

    public bool TryRead(SnapshotData snapshot, out SharedBgrFrame? frame)
    {
        frame = null;
        var name = snapshot.FrameSharedMemoryName;
        if (string.IsNullOrWhiteSpace(name))
        {
            return false;
        }

        try
        {
            EnsureMapping(name);
            var view = _view;
            if (view is null)
            {
                return false;
            }

            var observedMagic = new byte[4];
            for (var attempt = 0; attempt < 3; attempt++)
            {
                view.ReadArray(0, observedMagic, 0, observedMagic.Length);
                if (!observedMagic.AsSpan().SequenceEqual(Magic))
                {
                    return false;
                }

                var version = view.ReadUInt32(4);
                var epochBefore = view.ReadUInt64(8);
                if (version != Version || (epochBefore & 1UL) != 0)
                {
                    Thread.Yield();
                    continue;
                }

                var sequence = view.ReadUInt64(16);
                var width = checked((int)view.ReadUInt32(24));
                var height = checked((int)view.ReadUInt32(28));
                var stride = checked((int)view.ReadUInt32(32));
                var byteCount = checked((int)view.ReadUInt32(36));
                if (
                    width <= 0 ||
                    height <= 0 ||
                    stride < width * 3 ||
                    byteCount != stride * height ||
                    (snapshot.FrameSequence is not null &&
                        sequence != checked((ulong)snapshot.FrameSequence.Value))
                )
                {
                    return false;
                }

                if (_pixels is null || _pixels.Length != byteCount)
                {
                    _pixels = new byte[byteCount];
                }
                view.ReadArray(HeaderSize, _pixels, 0, byteCount);
                var epochAfter = view.ReadUInt64(8);
                if (epochBefore == epochAfter && (epochAfter & 1UL) == 0)
                {
                    frame = new SharedBgrFrame(_pixels, width, height, stride);
                    return true;
                }
                Thread.Yield();
            }
        }
        catch (Exception)
        {
            ResetMapping();
        }
        return false;
    }

    public void Dispose()
    {
        ResetMapping();
        _pixels = null;
    }

    private void EnsureMapping(string name)
    {
        if (string.Equals(_mapName, name, StringComparison.Ordinal))
        {
            return;
        }
        ResetMapping();
        _map = MemoryMappedFile.OpenExisting(name, MemoryMappedFileRights.Read);
        _view = _map.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        _mapName = name;
    }

    private void ResetMapping()
    {
        _view?.Dispose();
        _map?.Dispose();
        _view = null;
        _map = null;
        _mapName = null;
    }
}
