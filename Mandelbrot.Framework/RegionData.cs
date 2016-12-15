using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mandelbrot.Framework
{
    public class RegionData
    {

        public RegionData(RegionDefinition definition): this(new int[definition.Width * definition.Height], new byte[definition.Width * definition.Height*4], definition)
        {
        }

        public RegionData(int[] levels, byte[] colors, RegionDefinition definition)
        {
            Levels = levels;
            Definition = definition;
            Colors = colors;
            //Palette = palette;
        }

        public int[] Levels { get; }
        
        public byte[] Colors { get; }

        //public byte[] Palette { get; }

        public RegionDefinition Definition { get; }

        public void SetLevel(int x, int y, int value)
        {
            Levels[y*Definition.Width + x] = value;
        }

        public void SetLevel(int index, int value)
        {
            Levels[index] = value;
        }

    }
}
