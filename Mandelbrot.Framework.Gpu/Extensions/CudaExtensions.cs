using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using Mandelbrot.Framework.Extensions;

namespace Mandelbrot.Framework.Gpu.Extensions
{
    public static class CudaExtensions
    {
        
        private static void LoadTypeModule(GPGPU gpu, Type typeToCudafy)
        {
            var appFolder = AppDomain.CurrentDomain.BaseDirectory;
            var typeModulePath = Path.Combine(appFolder, typeToCudafy.Name + ".cdfy");
            var cudaModule = CudafyModule.TryDeserialize(typeModulePath);
            if (cudaModule == null || !cudaModule.TryVerifyChecksums())
            {
                cudaModule = CudafyTranslator.Cudafy(new[] { typeToCudafy });
                cudaModule.Serialize();
            }
            gpu.LoadModule(cudaModule, false);
        }

        public static void Execute<T>(this T instance, string kernel, int[] levels, byte[] colors, byte[] palette, RegionDefinition definition) where T: IFractal
        {
            CudafyModes.Target = eGPUType.Cuda;
            CudafyModes.DeviceId = 0;
            CudafyTranslator.Language = CudafyModes.Target == eGPUType.OpenCL ? eLanguage.OpenCL : eLanguage.Cuda;
            var deviceCount = CudafyHost.GetDeviceCount(CudafyModes.Target);
            if (deviceCount == 0) return;
            var gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            LoadTypeModule(gpu, instance.GetType());
            double[] parameters = null;
            parameters = (instance.Parameters as object).ToDoubleArray();
            var devLevels = gpu.Allocate<int>(levels.Length);
            var devColors = gpu.Allocate<byte>(colors.Length);
            var devPalette = gpu.Allocate<byte>(palette.Length);
            var devParameters = gpu.Allocate<double>(parameters.Length);
            //var stopwatch = new Stopwatch();
            //stopwatch.Start();
            gpu.CopyToDevice(palette, devPalette);
            gpu.CopyToDevice(parameters, devParameters);
            const int gridSide = 128;

            if (definition.Width % gridSide != 0 || definition.Height % gridSide != 0)
            {
                throw new ArgumentException(string.Format("Width and height must be a multiple of {0}", gridSide));
            }

            var blockWidth = definition.Width / gridSide;
            var blockHeight = definition.Height / gridSide;

            gpu.Launch(new dim3(gridSide, gridSide), new dim3(blockWidth, blockHeight), kernel, devLevels, devColors, devPalette, definition.Width, definition.Height, definition.SetLeft,
                definition.SetTop, definition.SetWidth, definition.SetHeight,
                definition.MaxLevels, devParameters);

            gpu.CopyFromDevice(devLevels, levels);
            gpu.CopyFromDevice(devColors, colors);

            //stopwatch.Stop();
            //Debug.WriteLine("Milliseconds: {0}", stopwatch.ElapsedMilliseconds);
            gpu.FreeAll();
            gpu.UnloadModules();
        }
    }
}
