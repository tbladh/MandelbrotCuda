using System.IO;
using System.Text;

namespace Mandelbrot
{

    public static class Utilities
    {
        public static byte[] LoadPalette(string fileName)
        {
            return File.ReadAllBytes(fileName);
        }

        public static string DumpBinaryPalette(byte[] palette, string name)
        {
            var sb = new StringBuilder();
            sb.AppendLine(string.Format("public static readonly byte[] {0} = {{", name));
            sb.Append("\t\t");
            var first = true;
            var col = 0;
            foreach (var b in palette)
            {
                if (!first) sb.Append(",");
                if (col > 2)
                {
                    sb.AppendLine();
                    sb.Append("\t\t");
                    col = 0;
                }
                sb.Append(b);
                first = false;
                col++;
            }
            sb.AppendLine();
            sb.AppendLine("};");

            return sb.ToString();
        }

    }
}
