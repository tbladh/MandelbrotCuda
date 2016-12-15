using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cudafy;
using Mandelbrot.Framework.Gpu.Extensions;

namespace Mandelbrot.Framework.Gpu
{
    public class JuliaGpu: Julia
    {
        public override string Name => "Julia (GPU)";
        public override Guid Id => new Guid("bba39b3f89e542cfb13139319c46f10b");

        public static byte[] Palette;

        static JuliaGpu()
        {
            Palette = new byte[768];
            var j = 255;
            for (var i = 0; i < Palette.Length; i += 3, j--)
            {
                Palette[i] = (byte)j;
                Palette[i + 1] = (byte)j;
                Palette[i + 2] = (byte)j;
            }
        }

        public override RegionData Generate(RegionDefinition definition, byte[] palette)
        {
            var data = new RegionData(definition);
            palette = Palette;
            this.Execute("JuliaKernel", data.Levels, data.Colors, palette, definition);

            return data;
        }

        [Cudafy]
        public static void JuliaKernel(GThread thread, int[] levels, byte[] colors, byte[] palette, int w, int h, double sx, double sy, double sw, double sh, int maxLevels, double[] parameters)
        {
            var x = thread.blockDim.x * thread.blockIdx.x + thread.threadIdx.x;
            var y = thread.blockDim.y * thread.blockIdx.y + thread.threadIdx.y;
            var offset = x + y * w;
            var xstep = sw / w;
            var ystep = sh / h;
            var cx = sx + x * xstep;
            var cy = sy + y * ystep;
            var colorOffset = offset * 4;
            var level = JSetLevel(cx, cy, parameters[0], parameters[1], maxLevels);
            levels[offset] = level;

            var paletteOffset = (int)((level / (double)maxLevels) * 256.0);
            if (paletteOffset > 255) paletteOffset = 255;
            paletteOffset = (paletteOffset * 3);
            if (level <= maxLevels)
            {
                colors[colorOffset] = palette[paletteOffset + 2];
                colors[colorOffset + 1] = palette[paletteOffset + 1];
                colors[colorOffset + 2] = palette[paletteOffset];
                colors[colorOffset + 3] = 255;
            }

        }

        [Cudafy]
        public static int JSetLevel(double zr, double zi, double cr, double ci, int max)
        {
            const double bailout = 4.0;
            var zrl = zr;
            var zil = zi;
            var zrs = zrl * zrl;
            var zis = zil * zil;
            var i = 0;

            do
            {
                zil = 2 * (zrl * zil) + ci;
                zrl = (zrs - zis) + cr;
                zrs = zrl * zrl;
                zis = zil * zil;
                i++;
            } while (i < max && (zrs + zis) <= bailout);

            return i;
        }

    }
}
