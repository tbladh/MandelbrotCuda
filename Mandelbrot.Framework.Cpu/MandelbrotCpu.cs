using System;
using System.Threading.Tasks;

namespace Mandelbrot.Framework.Cpu
{
    public class MandelbrotCpu: Mandelbrot
    {

        public override string Name => "Mandelbrot (CPU)";

        public override Guid Id => new Guid("9fe96fcd474649c6a6be3472ec794336");

        public override Guid? LinkedId => new Guid("c6d79046fed34ac9a800908b193218ad");

        public override RegionData Generate(RegionDefinition definition, byte[] palette)
        {
            var data = new RegionData(definition);
            var w = definition.Width;
            var h = definition.Height;
            var pixels = w * h;

            var sx = definition.SetWidth / w;
            var sy = definition.SetHeight / h;

            var degree = Environment.ProcessorCount * 2;

            Parallel.For(0, pixels, new ParallelOptions { MaxDegreeOfParallelism = degree }, index =>
            {

                var i = index % w;
                var j = (index - i) / w;

                var x = definition.SetLeft + i * sx;
                var y = definition.SetTop + j * sy;

                var level = MSetLevel(x, y, definition.MaxLevels, data.Colors, index * 4, palette);
                data.SetLevel(index, level);
                var colors = data.Colors;
                var colorOffset = index * 4;
                if (level < definition.MaxLevels)
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

            });

            return data;
        }

        //public override RegionData Generate(RegionDefinition definition, byte[] palette)
        //{
        //    var data = new RegionData(definition);
        //    var w = definition.Width;
        //    var h = definition.Height;
        //    var sx = definition.SetWidth/w;
        //    var sy = definition.SetHeight/h;
        //    var y = definition.SetTop;
        //    var index = 0;

        //    for (var j = 0; j < h; j++)
        //    {
        //        var x = definition.SetLeft;
        //        for (var i = 0; i < w; i++)
        //        {
        //            var level = MSetLevel(x, y, definition.MaxLevels, data.Colors, index*4, palette);
        //            data.SetLevel(index, level);
        //            var colors = data.Colors;
        //            var colorOffset = index * 4;
        //            if (level < definition.MaxLevels)
        //            {
        //                var paletteOffset = level * 3 % palette.Length;
        //                colors[colorOffset] = palette[paletteOffset + 2];
        //                colors[colorOffset + 1] = palette[paletteOffset + 1];
        //                colors[colorOffset + 2] = palette[paletteOffset];
        //                colors[colorOffset + 3] = 255;
        //            }
        //            else
        //            {
        //                colors[colorOffset] = 0;
        //                colors[colorOffset + 1] = 0;
        //                colors[colorOffset + 2] = 0;
        //                colors[colorOffset + 3] = 255;
        //            }

        //            x += sx;
        //            index++;
        //        }
        //        y += sy;
        //    }
        //    return data;
        //}

        private static int MSetLevel(double cr, double ci, int max, byte[] colors, int colorIndex, byte[] palette)
        {
            const double bailout = 4.0;
            var zr = 0.0;
            var zi = 0.0;
            var zrs = 0.0;
            var zis = 0.0;
            var i = 0;
           
            while(i < max && (zis + zrs) < bailout)
            {
                zi = 2.0*(zr*zi) + ci;
                zr = (zrs - zis) + cr;
                zis = zi * zi;
                zrs = zr * zr;
                i++;
            }

            if (i < max)
            { 
                var paletteIndex = i*3%palette.Length;
                colors[colorIndex] = palette[paletteIndex+2];
                colors[colorIndex+1] = palette[paletteIndex+1];
                colors[colorIndex+2] = palette[paletteIndex];
                colors[colorIndex + 3] = 255;
            }
            else
            {
                colors[colorIndex] = 0;
                colors[colorIndex + 1] = 0;
                colors[colorIndex + 2] = 0;
                colors[colorIndex + 3] = 255;
            }

            return i;
        }

    }
}
