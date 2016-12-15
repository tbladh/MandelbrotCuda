using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mandelbrot.Framework.Cpu
{
    public class JuliaCpu: Julia
    {

        public override string Name => "Julia (CPU)";

        public override Guid Id => new Guid("c6d79046fed34ac9a800908b193218ad");

        public static byte[] Palette;

        static JuliaCpu()
        {
            Palette = new byte[768];
            var j = 255;
            for (var i = 0; i < Palette.Length; i += 3, j--)
            {
                Palette[i] = (byte)j;
                Palette[i+1] = (byte)j;
                Palette[i+2] = (byte)j;
            }
        }

        public override RegionData Generate(RegionDefinition definition, byte[] palette)
        {
            palette = Palette;
            var stopwatch = new Stopwatch();
            stopwatch.Start();

            var data = new RegionData(definition);
            var w = definition.Width;
            var h = definition.Height;
            var sx = definition.SetWidth / w;
            var sy = definition.SetHeight / h;
            var y = definition.SetTop;
            var index = 0;
            var cr = Parameters.Cr;
            var ci = Parameters.Ci;
            for (var j = 0; j < h; j++)
            {
                var x = definition.SetLeft;
                for (var i = 0; i < w; i++)
                {
                    var level = JSetLevel(x, y, cr, ci, definition.MaxLevels);
                    data.SetLevel(index, level);

                    var colors = data.Colors;
                    var colorOffset = index * 4;

                    var paletteOffset = (int)((level / (double)definition.MaxLevels) * 256.0);
                    if (paletteOffset > 255) paletteOffset = 255;
                    paletteOffset = (paletteOffset*3);
                    if (level <= definition.MaxLevels)
                    {
                        colors[colorOffset] = palette[paletteOffset + 2];
                        colors[colorOffset + 1] = palette[paletteOffset + 1];
                        colors[colorOffset + 2] = palette[paletteOffset];
                        colors[colorOffset + 3] = 255;
                    }

                    x += sx;
                    index++;
                }
                y += sy;
            }

            stopwatch.Stop();
            Debug.WriteLine("Milliseconds: {0}", stopwatch.ElapsedMilliseconds);

            return data;
        }

        public int JSetLevel(double zr, double zi, double cr, double ci, int max)
        {
            const double bailout = 4.0;
            var zrl = zr;
            var zil = zi;
            var zrs = zrl * zrl;
            var zis = zil * zil;
            var i = 0;

            do
            {
                zil = 2*(zrl*zil) + ci;
                zrl = (zrs - zis) + cr;
                zrs = zrl*zrl;
                zis = zil*zil;
                i++;
            } while (i < max && (zrs + zis) <= bailout);

            return i;
        }

    }
}
