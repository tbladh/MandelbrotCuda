using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Mandelbrot.Framework.Extensions
{ 
    public static class ObjectExtensions
    {

        public static double[] ToDoubleArray(this object obj)
        {
            return obj?.ToDoubleEnum().ToArray() ?? new double[1];
        }

        public static IEnumerable<double> ToDoubleEnum(this object obj)
        {
            if (obj == null) yield break;
            var type = obj.GetType();
            var properties = type.GetProperties(BindingFlags.Instance | BindingFlags.Public);
            foreach (var property in properties)
            {
                if (!property.PropertyType.IsPrimitive) continue;
                var value = property.GetValue(obj);
                yield return Convert.ToDouble(value);
            }
        }
    }
}
