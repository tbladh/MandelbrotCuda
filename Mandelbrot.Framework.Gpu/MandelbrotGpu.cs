using System;
using Cudafy;
using Mandelbrot.Framework.Gpu.Extensions;

namespace Mandelbrot.Framework.Gpu
{
    public class MandelbrotGpu: Mandelbrot
    {

        public override string Name => "Mandelbrot (GPU)";

        public override Guid Id => new Guid("ed87ad6e2c984ef0aba5cf00f63b85a2");

        public override Guid? LinkedId => new Guid("bba39b3f89e542cfb13139319c46f10b");

        public override RegionData Generate(RegionDefinition definition, byte[] palette)
        {
            var data = new RegionData(definition);

            this.Execute("MandelbrotKernel", data.Levels, data.Colors, palette, definition);

            return data;
        }

        [Cudafy]
        public static void MandelbrotKernel(GThread thread, int[] levels, byte[] colors, byte[] palette, int w, int h, double sx, double sy, double sw, double sh, int maxLevels, double[] parameters)
        {
            var x = thread.blockDim.x * thread.blockIdx.x + thread.threadIdx.x;
            var y = thread.blockDim.y * thread.blockIdx.y + thread.threadIdx.y;
            var offset = x + y * w;
            var xstep = sw/w;
            var ystep = sh/h;
            var cx = sx + x*xstep;
            var cy = sy + y*ystep;
            var colorOffset = offset * 4;
            var level = MSetLevel(cx, cy, maxLevels);
            levels[offset] = level;

            if (level < maxLevels)
            {
                var paletteOffset = level * 3 % palette.Length;
                colors[colorOffset] = palette[paletteOffset + 2];
                colors[colorOffset + 1] = palette[paletteOffset + 1];
                colors[colorOffset + 2] = palette[paletteOffset];
                colors[colorOffset + 3] = 255;
            }
            else
            {
                colors[colorOffset] = 0;
                colors[colorOffset + 1] = 0;
                colors[colorOffset + 2] = 0;
                colors[colorOffset + 3] = 255;
            }

        }

        [Cudafy]
        public static int MSetLevel(double cr, double ci, int max)
        {
            const double bailout = 4.0;
            var zr = 0.0;
            var zi = 0.0;
            var zrs = 0.0;
            var zis = 0.0;
            var i = 0;
            while (i < max && (zis + zrs) < bailout)
            {
                zi = 2.0 * (zr * zi) + ci;
                zr = (zrs - zis) + cr;
                zis = zi * zi;
                zrs = zr * zr;
                i++;
            }
            return i;
        }
    }
}
