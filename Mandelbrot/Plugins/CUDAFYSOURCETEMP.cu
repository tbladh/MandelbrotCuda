
// Mandelbrot.Framework.Gpu.JuliaGpu
extern "C" __global__  void JuliaKernel( int* levels, int levelsLen0,  unsigned char* colors, int colorsLen0,  unsigned char* palette, int paletteLen0, int w, int h, double sx, double sy, double sw, double sh, int maxLevels,  double* parameters, int parametersLen0);
// Mandelbrot.Framework.Gpu.JuliaGpu
__device__  int JSetLevel(double zr, double zi, double cr, double ci, int max);
// Mandelbrot.Framework.Gpu.MandelbrotGpu
extern "C" __global__  void MandelbrotKernel( int* levels, int levelsLen0,  unsigned char* colors, int colorsLen0,  unsigned char* palette, int paletteLen0, int w, int h, double sx, double sy, double sw, double sh, int maxLevels,  double* parameters, int parametersLen0);
// Mandelbrot.Framework.Gpu.MandelbrotGpu
__device__  int MSetLevel(double cr, double ci, int max);

// Mandelbrot.Framework.Gpu.JuliaGpu
extern "C" __global__  void JuliaKernel( int* levels, int levelsLen0,  unsigned char* colors, int colorsLen0,  unsigned char* palette, int paletteLen0, int w, int h, double sx, double sy, double sw, double sh, int maxLevels,  double* parameters, int parametersLen0)
{
	int num = blockDim.x * blockIdx.x + threadIdx.x;
	int num2 = blockDim.y * blockIdx.y + threadIdx.y;
	int num3 = num + num2 * w;
	double num4 = sw / (double)w;
	double num5 = sh / (double)h;
	double zr = sx + (double)num * num4;
	double zi = sy + (double)num2 * num5;
	int num6 = num3 * 4;
	int num7 = JSetLevel(zr, zi, parameters[(0)], parameters[(1)], maxLevels);
	levels[(num3)] = num7;
	int num8 = (int)((double)num7 / (double)maxLevels * 256.0);
	bool flag = num8 > 255;
	if (flag)
	{
		num8 = 255;
	}
	num8 *= 3;
	bool flag2 = num7 <= maxLevels;
	if (flag2)
	{
		colors[(num6)] = palette[(num8 + 2)];
		colors[(num6 + 1)] = palette[(num8 + 1)];
		colors[(num6 + 2)] = palette[(num8)];
		colors[(num6 + 3)] = 255;
	}
}
// Mandelbrot.Framework.Gpu.JuliaGpu
__device__  int JSetLevel(double zr, double zi, double cr, double ci, int max)
{
	double num = zr;
	double num2 = zi;
	double num3 = num * num;
	double num4 = num2 * num2;
	int num5 = 0;
	do
	{
		num2 = 2.0 * (num * num2) + ci;
		num = num3 - num4 + cr;
		num3 = num * num;
		num4 = num2 * num2;
		num5++;
	}
	while (num5 < max && num3 + num4 <= 4.0);
	return num5;
}
// Mandelbrot.Framework.Gpu.MandelbrotGpu
extern "C" __global__  void MandelbrotKernel( int* levels, int levelsLen0,  unsigned char* colors, int colorsLen0,  unsigned char* palette, int paletteLen0, int w, int h, double sx, double sy, double sw, double sh, int maxLevels,  double* parameters, int parametersLen0)
{
	int num = blockDim.x * blockIdx.x + threadIdx.x;
	int num2 = blockDim.y * blockIdx.y + threadIdx.y;
	int num3 = num + num2 * w;
	double num4 = sw / (double)w;
	double num5 = sh / (double)h;
	double cr = sx + (double)num * num4;
	double ci = sy + (double)num2 * num5;
	int num6 = num3 * 4;
	int num7 = MSetLevel(cr, ci, maxLevels);
	levels[(num3)] = num7;
	bool flag = num7 < maxLevels;
	if (flag)
	{
		int num8 = num7 * 3 % paletteLen0;
		colors[(num6)] = palette[(num8 + 2)];
		colors[(num6 + 1)] = palette[(num8 + 1)];
		colors[(num6 + 2)] = palette[(num8)];
		colors[(num6 + 3)] = 255;
	}
	else
	{
		colors[(num6)] = 0;
		colors[(num6 + 1)] = 0;
		colors[(num6 + 2)] = 0;
		colors[(num6 + 3)] = 255;
	}
}
// Mandelbrot.Framework.Gpu.MandelbrotGpu
__device__  int MSetLevel(double cr, double ci, int max)
{
	double num = 0.0;
	double num2 = 0.0;
	double num3 = 0.0;
	double num4 = 0.0;
	int num5 = 0;
	while (num5 < max && num4 + num3 < 4.0)
	{
		num2 = 2.0 * (num * num2) + ci;
		num = num3 - num4 + cr;
		num4 = num2 * num2;
		num3 = num * num;
		num5++;
	}
	return num5;
}
