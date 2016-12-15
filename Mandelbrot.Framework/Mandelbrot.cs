using System;

namespace Mandelbrot.Framework
{
    public abstract class Mandelbrot: IFractal
    {

        public abstract string Name { get; }

        public abstract Guid Id { get; }
        public abstract Guid? LinkedId { get; }

        public virtual RegionDefinition InitialRegion => new RegionDefinition(-3.0, -1.5, 4.8, 3.0, 384, 2048, 1280);

        public virtual dynamic Parameters
        {
            get { return null; }
        }

        public abstract RegionData Generate(RegionDefinition definition, byte[] palette);

        public override bool Equals(object obj)
        {
            var item = obj as Julia;
            return item != null && Id == item.Id;
        }

        protected bool Equals(Mandelbrot other)
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
