using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mandelbrot.Framework
{
    public abstract class Julia: IFractal
    {
        
        public abstract string Name { get; }

        public abstract Guid Id { get; }
        public virtual Guid? LinkedId => null;

        public virtual dynamic Parameters
        {
            get; set;
        }

        protected Julia()
        {
            Parameters = new JuliaParameters { Cr = -0.74543, Ci = 0.11301 };
        }

        public virtual RegionDefinition InitialRegion => new RegionDefinition(-2.4, -1.5, 4.8, 3.0, 64,
            2048, 1280);

        public abstract RegionData Generate(RegionDefinition definition, byte[] palette);

        public override bool Equals(object obj)
        {
            var item = obj as Julia;
            return item != null && Id == item.Id;
        }

        protected bool Equals(Julia other)
        {
            return Equals((object)other);
        }

        public override int GetHashCode()
        {
            // Not ideal but there is no need for anything else right now.
            return Id.GetHashCode();
        }


    }
}
