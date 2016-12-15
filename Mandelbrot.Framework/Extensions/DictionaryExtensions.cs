using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mandelbrot.Framework.Extensions
{
    public static class DictionaryExtensions
    {
        public static Dictionary<TKey, TValue> CloneDictionary<TKey, TValue>(this Dictionary<TKey, TValue> original)// where TValue : ICloneable
        {
            var ret = new Dictionary<TKey, TValue>(original.Count, original.Comparer);
            foreach (var entry in original)
            {
                var cloneable = entry.Value as ICloneable;
                if (cloneable != null)
                {
                    ret.Add(entry.Key, (TValue)cloneable.Clone());
                }
                else
                {
                    ret.Add(entry.Key, entry.Value);
                }
                
            }
            return ret;
        }

    }
}
