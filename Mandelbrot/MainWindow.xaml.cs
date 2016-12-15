using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using Mandelbrot.Framework;
using Mandelbrot.Presentation;
using Microsoft.Win32;


namespace Mandelbrot
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {

        private RegionDefinition _region;

        public IFractal Fractal { get; set; }

        private IFractal _linkedFractal;

        private readonly Stack<RegionDefinition> _regionStack;

        private readonly byte[] _palette;

        public List<IFractal> Implementations { get; }
        
        private Point _currentPosition;

        private RegionDefinition _juliaRegion;


        public MainWindow()
        {
            PreviewKeyDown += MainWindow_PreviewKeyDown;

            
            _regionStack = new Stack<RegionDefinition>();
            
            Implementations = GetImplementations().ToList();

            Fractal = Implementations.First(p => p.Id == new Guid("ed87ad6e2c984ef0aba5cf00f63b85a2"));
            Loaded += MainWindow_Loaded;

            InitializeComponent();
            ImplementationCombo.Loaded += ImplementationCombo_Loaded;
            //ImplementationCombo.DataContext = this;

            if (Fractal.LinkedId != null)
            {
                _linkedFractal = Implementations.FirstOrDefault(p => p.Id == Fractal.LinkedId);
                if (_linkedFractal != null)
                { 
                    _juliaRegion = _linkedFractal.InitialRegion.Clone();
                }
            }

            _region = Fractal.InitialRegion;
            _palette = Palettes.Standard;

            
            ResetButton.Click += ResetButton_Click;
            BackButton.Click += BackButton_Click;
            SaveButton.Click += SaveButton_Click;

            MandelbrotImage.Cursor = Cursors.Cross;
            MandelbrotImage.MouseDown += MandelbrotImage_MouseDown;
            MandelbrotImage.MouseMove += MandelbrotImage_MouseMove;
            
        }

        private bool _implementationComboUserInteracted = false;
        private void ImplementationCombo_DropDownOpened(object sender, EventArgs e)
        {
            _implementationComboUserInteracted = true;
        }

        private void MainWindow_PreviewKeyDown(object sender, KeyEventArgs e)
        {

            if (e.Key != Key.J) return;
            if (Fractal.LinkedId != null)
            {
                Fractal = _linkedFractal;
                _linkedFractal = null;

                if (Fractal is Julia)
                {
                    Fractal.Parameters.Cr = _currentPosition.X;
                    Fractal.Parameters.Ci = _currentPosition.Y;
                    _region = _juliaRegion.Clone();
                    _region.MaxLevels = 256;
                }

                ImplementationCombo.SelectedIndex = ImplementationCombo.Items.IndexOf(Fractal);

                UpdateFractal();
            }
            e.Handled = true;
        }

        private static IEnumerable<IFractal> GetImplementations()
        {
            var target = typeof(IFractal);
            var loadedAssemblies = AppDomain.CurrentDomain.GetAssemblies().ToList();
            var files = Directory.GetFiles(AppDomain.CurrentDomain.BaseDirectory)
                .Where(p => Path.GetExtension(p) == ".dll" && 
                loadedAssemblies.All(e => e.GetName().Name != Path.GetFileNameWithoutExtension(p))).ToArray();
            foreach (var file in files)
            {
                var assembly = AppDomain.CurrentDomain.Load(File.ReadAllBytes(file));
                var types = assembly.GetTypes().Where(p => target.IsAssignableFrom(p) && !p.IsInterface && !p.IsAbstract);
                foreach (var type in types)
                {
                    var fileName = Path.GetFileNameWithoutExtension(file);
                    if (fileName == null) continue;
                    var instance = (IFractal)AppDomain.CurrentDomain.CreateInstanceAndUnwrap(fileName, type.FullName);
                    yield return instance;
                }
            }
        }

        private void SaveButton_Click(object sender, RoutedEventArgs e)
        {
            var fileSave = new SaveFileDialog
            {
                FileName = "FractalHighRes",
                DefaultExt = ".png",
                Filter = "Bitmaps (.png)|*.png"
            };
            var result = fileSave.ShowDialog();
            if (result != true) return;
            var filePath = fileSave.FileName;

            var clone = _region.Clone();
            clone.Width *= 2;
            clone.Height *= 2;
            var data = Fractal.Generate(clone, _palette);
            var bitmap = data.ToBitmap();
            using (var fileStream = new FileStream(filePath, FileMode.Create))
            {
                var encoder = new PngBitmapEncoder();
                {
                    encoder.Frames.Add(BitmapFrame.Create(bitmap));
                    encoder.Save(fileStream);
                }
            }
        }

        private void BackButton_Click(object sender, RoutedEventArgs e)
        {
            if (_regionStack.Count == 0) return;
            _region = _regionStack.Pop();
            UpdateFractal();
        }

        private void ResetButton_Click(object sender, RoutedEventArgs e)
        {
            _regionStack.Clear();
            _region = Fractal.InitialRegion;
            UpdateFractal();
        }

        private Point GetPixelPosition(double x, double y)
        {
            var p = HostGrid.TranslatePoint(new Point(x, y), MandelbrotImage);
            var unitPosition = new Point(p.X / MandelbrotImage.ActualWidth, p.Y / MandelbrotImage.ActualHeight);
            var pixelPosition = new Point(unitPosition.X * _region.Width, unitPosition.Y * _region.Height);
            return pixelPosition;
        }

        private Point GetRegionPosition(object sender, MouseEventArgs e)
        {
            var clickedPoint = e.GetPosition((Image)sender);
            var unitPosition = new Point(clickedPoint.X / MandelbrotImage.ActualWidth, clickedPoint.Y / MandelbrotImage.ActualHeight);
            var mandelPosition = new Point(_region.SetLeft + unitPosition.X * _region.SetWidth, _region.SetTop + unitPosition.Y * _region.SetHeight);
            return mandelPosition;
        }

        private void MandelbrotImage_MouseMove(object sender, MouseEventArgs e)
        {
            if (_linkedFractal == null || Fractal is Julia) return;
            var mandelPosition = GetRegionPosition(sender, e);
            _currentPosition = mandelPosition;
            UpdateJulia(mandelPosition.X, mandelPosition.Y);
        }
       
        private void MandelbrotImage_MouseDown(object sender, MouseButtonEventArgs e)
        {
            var mandelPosition = GetRegionPosition(sender, e);
            var newRegion = _region.Zoom(Fractal.InitialRegion, mandelPosition.X, mandelPosition.Y, 0.5);
            _regionStack.Push(_region);
            _region = newRegion;
            UpdateFractal();
            MandelbrotImage_MouseMove(sender, e);
        }

        private void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            UpdateFractal();
        }

        private void ImplementationCombo_Loaded(object sender, RoutedEventArgs e)
        {
            ImplementationCombo.DropDownOpened += ImplementationCombo_DropDownOpened;
            ImplementationCombo.SelectionChanged += Selector_OnSelectionChanged;
            foreach (var implementation in Implementations)
            {
                ImplementationCombo.Items.Add(implementation);
            }
            ImplementationCombo.SelectedIndex = Implementations.IndexOf(Fractal);
            ImplementationCombo.Text = Fractal.Name;
        }

        private void UpdateFractal()
        {
            var stopWatch = new Stopwatch();
            stopWatch.Start();
            var data = Fractal.Generate(_region, _palette);
            stopWatch.Stop();

            StatsLabel.Content = string.Format("{0} ms", stopWatch.ElapsedMilliseconds);

            MandelbrotImage.Source = data.ToBitmap();
        }

        private void UpdateJulia(double cr, double ci)
        {
            var region = _linkedFractal.InitialRegion.Clone();
            region.Width = 512;
            region.Height = 256;
            _linkedFractal.Parameters.Cr = cr;
            _linkedFractal.Parameters.Ci = ci;
            var data = _linkedFractal.Generate(region, new byte[] { });
            var p = GetPixelPosition(0, 0);
            ((WriteableBitmap)MandelbrotImage.Source).Overlay(data, (int)p.X, (int)p.Y);
        }

        private void Selector_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            var combo = sender as ComboBox;
            if (combo != null)
            {

                if (!_implementationComboUserInteracted)
                {
                    //e.Handled = true;
                    return;
                }
                _implementationComboUserInteracted = false;

                var item = (IFractal)combo.SelectedItem;
                if (item == null) return;
                Fractal = item;

                _regionStack.Clear();
                if (Fractal.LinkedId != null)
                {
                    _linkedFractal = Implementations.FirstOrDefault(p => p.Id == Fractal.LinkedId);
                }

                if (Fractal is Julia)
                {
                    _region = _juliaRegion;
                }
                else
                {
                    _region = Fractal.InitialRegion;
                }

                
                UpdateFractal();
            }
         
        }
    }
}

