using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;
using LocalSecurityMonitor.Wpf.Models;

namespace LocalSecurityMonitor.Wpf.Services;

public static class DetectionOverlayRenderer
{
    public static void Render(
        Canvas canvas,
        CameraViewModel? camera,
        PersonDetectionData? detection,
        FaceRecognitionData? face,
        PersonSettingsData settings)
    {
        canvas.Children.Clear();
        if (camera is null)
        {
            return;
        }

        var personVisible = detection is not null
            && detection.CameraId == camera.CameraId
            && detection.Status == "RUNNING"
            && detection.SourceWidth is not null
            && detection.SourceHeight is not null
            && detection.Detections.Count > 0
            && (settings.ShowBoxes || settings.ShowMasks);
        var faceVisible = face is not null
            && face.CameraId == camera.CameraId
            && face.Status.Equals("running", StringComparison.OrdinalIgnoreCase)
            && face.Overlays.Count > 0;
        if (!personVisible && !faceVisible)
        {
            return;
        }

        var canvasWidth = canvas.ActualWidth;
        var canvasHeight = canvas.ActualHeight;
        if (canvasWidth <= 1 || canvasHeight <= 1)
        {
            return;
        }

        var sourceWidth = detection?.SourceWidth ?? camera.RawWidth;
        var sourceHeight = detection?.SourceHeight ?? camera.RawHeight;
        if (sourceWidth <= 0 || sourceHeight <= 0)
        {
            return;
        }

        var effectiveWidth = camera.RotationDegrees is 90 or 270 ? sourceHeight : sourceWidth;
        var effectiveHeight = camera.RotationDegrees is 90 or 270 ? sourceWidth : sourceHeight;
        var scale = Math.Min(canvasWidth / effectiveWidth, canvasHeight / effectiveHeight);
        var renderWidth = effectiveWidth * scale;
        var renderHeight = effectiveHeight * scale;
        var originX = (canvasWidth - renderWidth) / 2;
        var originY = (canvasHeight - renderHeight) / 2;

        if (personVisible && detection is not null)
        {
            RenderPersons(
                canvas,
                camera,
                detection,
                settings,
                sourceWidth,
                sourceHeight,
                effectiveWidth,
                effectiveHeight,
                originX,
                originY,
                scale);
        }

        if (faceVisible && face is not null)
        {
            RenderFaces(
                canvas,
                camera,
                face,
                sourceWidth,
                sourceHeight,
                effectiveWidth,
                effectiveHeight,
                originX,
                originY,
                scale,
                canvasHeight);
        }
    }

    private static void RenderPersons(
        Canvas canvas,
        CameraViewModel camera,
        PersonDetectionData detection,
        PersonSettingsData settings,
        int sourceWidth,
        int sourceHeight,
        int effectiveWidth,
        int effectiveHeight,
        double originX,
        double originY,
        double scale)
    {
        foreach (var item in detection.Detections)
        {
            if (item.MaskPolygon is not null && settings.ShowMasks)
            {
                var polygon = new Polygon
                {
                    Stroke = new SolidColorBrush(Color.FromRgb(79, 190, 126)),
                    StrokeThickness = 1,
                    Fill = new SolidColorBrush(Color.FromArgb(60, 79, 190, 126)),
                };
                foreach (var point in item.MaskPolygon)
                {
                    if (point.Length >= 2)
                    {
                        var mapped = MapPoint(
                            point[0], point[1], sourceWidth, sourceHeight,
                            effectiveWidth, originX, originY, scale, camera);
                        polygon.Points.Add(new Point(mapped.X, mapped.Y));
                    }
                }
                canvas.Children.Add(polygon);
            }

            if (!settings.ShowBoxes || item.Bbox.Length < 4)
            {
                continue;
            }

            var topLeft = MapPoint(
                item.Bbox[0], item.Bbox[1], sourceWidth, sourceHeight,
                effectiveWidth, originX, originY, scale, camera);
            var bottomRight = MapPoint(
                item.Bbox[2], item.Bbox[3], sourceWidth, sourceHeight,
                effectiveWidth, originX, originY, scale, camera);
            var rectangle = new Rectangle
            {
                Width = Math.Max(1, Math.Abs(bottomRight.X - topLeft.X)),
                Height = Math.Max(1, Math.Abs(bottomRight.Y - topLeft.Y)),
                Stroke = new SolidColorBrush(Color.FromRgb(114, 237, 146)),
                StrokeThickness = 2,
            };
            Canvas.SetLeft(rectangle, Math.Min(topLeft.X, bottomRight.X));
            Canvas.SetTop(rectangle, Math.Min(topLeft.Y, bottomRight.Y));
            canvas.Children.Add(rectangle);

            var label = new Border
            {
                Background = new SolidColorBrush(Color.FromArgb(220, 8, 31, 24)),
                Padding = new Thickness(4, 2, 4, 2),
                Child = new TextBlock
                {
                    Text = $"{item.Label} {item.Confidence:0.00}",
                    Foreground = new SolidColorBrush(Color.FromRgb(217, 255, 226)),
                    FontSize = 11,
                },
            };
            Canvas.SetLeft(label, Math.Min(topLeft.X, bottomRight.X));
            Canvas.SetTop(label, Math.Max(0, Math.Min(topLeft.Y, bottomRight.Y) - 23));
            canvas.Children.Add(label);
        }
    }

    private static void RenderFaces(
        Canvas canvas,
        CameraViewModel camera,
        FaceRecognitionData face,
        int sourceWidth,
        int sourceHeight,
        int effectiveWidth,
        int effectiveHeight,
        double originX,
        double originY,
        double scale,
        double canvasHeight)
    {
        foreach (var item in face.Overlays)
        {
            if (item.Bbox.Length < 4)
            {
                continue;
            }

            var topLeft = MapPoint(
                item.Bbox[0], item.Bbox[1], sourceWidth, sourceHeight,
                effectiveWidth, originX, originY, scale, camera);
            var bottomRight = MapPoint(
                item.Bbox[2], item.Bbox[3], sourceWidth, sourceHeight,
                effectiveWidth, originX, originY, scale, camera);
            var known = string.Equals(item.RecognitionStatus, "known", StringComparison.OrdinalIgnoreCase);
            var color = known ? Color.FromRgb(101, 230, 255) : Color.FromRgb(255, 209, 102);
            var rectangle = new Rectangle
            {
                Width = Math.Max(1, Math.Abs(bottomRight.X - topLeft.X)),
                Height = Math.Max(1, Math.Abs(bottomRight.Y - topLeft.Y)),
                Stroke = new SolidColorBrush(color),
                StrokeThickness = 2,
            };
            Canvas.SetLeft(rectangle, Math.Min(topLeft.X, bottomRight.X));
            Canvas.SetTop(rectangle, Math.Min(topLeft.Y, bottomRight.Y));
            canvas.Children.Add(rectangle);

            foreach (var point in item.Landmarks)
            {
                if (point.Length < 2)
                {
                    continue;
                }
                var mapped = MapPoint(
                    point[0], point[1], sourceWidth, sourceHeight,
                    effectiveWidth, originX, originY, scale, camera);
                var marker = new Ellipse
                {
                    Width = 6,
                    Height = 6,
                    Fill = new SolidColorBrush(color),
                };
                Canvas.SetLeft(marker, mapped.X - 3);
                Canvas.SetTop(marker, mapped.Y - 3);
                canvas.Children.Add(marker);
            }

            var faceLabel = $"track {item.TrackId} · {item.RecognitionStatus.ToUpperInvariant()}";
            if (!string.IsNullOrWhiteSpace(item.PersonName))
            {
                faceLabel += $" · {item.PersonName}";
            }
            if (item.Score is not null)
            {
                faceLabel += $" {item.Score.Value:0.00}";
            }
            var label = new Border
            {
                Background = new SolidColorBrush(Color.FromArgb(220, 8, 26, 35)),
                Padding = new Thickness(4, 2, 4, 2),
                Child = new TextBlock
                {
                    Text = faceLabel,
                    Foreground = new SolidColorBrush(Color.FromRgb(229, 251, 255)),
                    FontSize = 11,
                },
            };
            Canvas.SetLeft(label, Math.Min(topLeft.X, bottomRight.X));
            Canvas.SetTop(
                label,
                Math.Min(canvasHeight - 23, Math.Max(0, Math.Min(topLeft.Y, bottomRight.Y) - 23)));
            canvas.Children.Add(label);
        }
    }

    private static (double X, double Y) MapPoint(
        double x,
        double y,
        int sourceWidth,
        int sourceHeight,
        int effectiveWidth,
        double originX,
        double originY,
        double scale,
        CameraViewModel camera)
    {
        var (rotatedX, rotatedY) = camera.RotationDegrees switch
        {
            90 => (y, sourceWidth - x),
            180 => (sourceWidth - x, sourceHeight - y),
            270 => (sourceHeight - y, x),
            _ => (x, y),
        };
        if (camera.IsMirrored)
        {
            rotatedX = effectiveWidth - rotatedX;
        }
        return (originX + rotatedX * scale, originY + rotatedY * scale);
    }
}
