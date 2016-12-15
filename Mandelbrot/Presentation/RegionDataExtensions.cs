using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Mandelbrot.Framework;

namespace Mandelbrot.Presentation
{
    public static class RegionDataExtensions
    {
        public static WriteableBitmap ToBitmap(this RegionData data)
        {
            var wb = new WriteableBitmap(data.Definition.Width, data.Definition.Height, 100, 100, PixelFormats.Bgra32, null);
            var stride = data.Definition.Width * PixelFormats.Bgra32.BitsPerPixel / 8;
            wb.WritePixels(
              new Int32Rect(0, 0, data.Definition.Width, data.Definition.Height),
               data.Colors, stride, 0, 0);
            return wb;
        }

        public static void Overlay(this WriteableBitmap bitmap, RegionData data, int x = 0, int y = 0)
        {
            var sourceStride = data.Definition.Width * PixelFormats.Bgra32.BitsPerPixel / 8;
            bitmap.WritePixels(new Int32Rect(0, 0, data.Definition.Width, data.Definition.Height), data.Colors, sourceStride, x, y);
        }
      
    }
}
