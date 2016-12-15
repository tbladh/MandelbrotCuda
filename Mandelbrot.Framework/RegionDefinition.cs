using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Mandelbrot.Framework.Extensions;

namespace Mandelbrot.Framework
{
    public struct  RegionDefinition
    {

        public int MaxLevels { get; set; }

        //public Dictionary<string, double> Parameters { get; set; }
        public double SetLeft;

        public double SetTop;

        public double SetWidth;

        public double SetHeight;

        public int Width;

        public int Height;

        public RegionDefinition(double setLeft, double setTop, double setWidth, 
            double setHeight, int maxLevels, int width, int height)
        {
            SetLeft = setLeft;
            SetTop = setTop;
            SetWidth = setWidth;
            SetHeight = setHeight;
            MaxLevels = maxLevels;
            Width = width;
            Height = height;
        }

        public RegionDefinition Zoom(RegionDefinition initial, double x, double y, double fraction)
        {
            var aspect = SetHeight/SetWidth;
            var nw = SetWidth*fraction;
            var nh = nw*aspect;
            var wh = nw / 2;
            var hh = nh / 2;
            var sl = x - wh;
            var st = y - hh;

            var scale = nw/initial.SetWidth;

            const int levelMax = 2048;
            var max = MaxLevels;
            if (max < levelMax)
            { 
                max = (int)(512 + Math.Pow(1.0 / scale, 1.0/1.5));
                if (max > levelMax || max < 0) max = levelMax;
            }
            Debug.WriteLine("Max: {0}", max);
            return new RegionDefinition(sl, st, nw, nh, max, Width, Height);
        }

        public RegionDefinition Clone()
        {
            return new RegionDefinition(SetLeft,SetTop,SetWidth,SetHeight, MaxLevels, Width, Height);
        }

    }
}
