using System;
using System.Security.Cryptography.X509Certificates;

namespace Mandelbrot.Framework
{
    public interface IFractal
    {

        string Name { get; }

        Guid Id { get; }

        Guid? LinkedId { get; }

        RegionDefinition InitialRegion { get; }

        dynamic Parameters { get; }

        RegionData Generate(RegionDefinition definition, byte[] palette);
        
    }
}
