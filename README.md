# PlotFDM - FDM Seismic Data Visualization Tool

A Python tool for reading and visualizing FDM (Finite Difference Modeling) seismic data files with intelligent plotting capabilities.

## Features

- **FDM File Format Support**: Read FDM binary files with 512-byte headers
- **Intelligent Plotting**: Automatic colormap selection (TgsBanded for velocity, gray_r for seismic)
- **Smart Scaling**: Automatic RMS -4dB scaling with seismic power dB convention
- **Interactive Controls**: Real-time gain/dim adjustments with keyboard shortcuts
- **Custom Colormaps**: BigTiger and TgsBanded colormaps for seismic visualization
- **Multiple Views**: Single slice or summary plots with all three axes
- **Flexible Coordinates**: Support for both logical (grid) and physical coordinates
- **Manual Scaling**: Override automatic scaling with custom min/max values

## Sample Data

The repository includes sample FDM files in the `examples/` directory:
- `examples/shot_topo.fdm` - Seismic shot data with topography
- `examples/vel_topo.fdm` - Velocity model with topography

Try them out:
```bash
python plotfdm.py examples/shot_topo.fdm --summary
python plotfdm.py examples/vel_topo.fdm --summary
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/plotfdm.git
cd plotfdm
pip install -r requirements.txt
```

## Dependencies

- Python 3.6+
- NumPy
- Matplotlib
- Custom colormaps module

## Usage

### Basic Usage

```bash
# Plot middle Y-slice with automatic settings
python plotfdm.py data.fdm

# Show summary with all three axis slices
python plotfdm.py data.fdm --summary

# Plot specific slice
python plotfdm.py data.fdm --slice z --index 10

# Use physical coordinates instead of logical
python plotfdm.py data.fdm --physical
```

### Advanced Options

```bash
# Manual colormap scaling
python plotfdm.py data.fdm --vmin -100 --vmax 100

# Custom colormap
python plotfdm.py data.fdm --cmap viridis

# Save to file instead of displaying
python plotfdm.py data.fdm --save output.png

# Show file info only
python plotfdm.py data.fdm --info
```

### Interactive Controls

When displaying plots interactively, you can use keyboard shortcuts:

- `g` - Gain +1.5dB
- `gg` - Gain +3.0dB (press g twice quickly)
- `d` - Dim -1.5dB
- `dd` - Dim -3.0dB (press d twice quickly)

## File Structure

- `plotfdm.py` - Main plotting tool with CLI interface
- `fdm.py` - FDM file format reader class
- `custom_colormaps.py` - Custom colormap definitions

## FDM File Format

The tool supports FDM files with:
- 512-byte binary headers
- 3D data arrays with (ny, nx, nz) memory layout
- Physical coordinate conversion
- Automatic data type detection

## Automatic Scaling

The tool uses intelligent defaults:
- **Colormap Selection**: TgsBanded for velocity data (all positive), gray_r for seismic data (pos/neg)
- **Scaling**: RMS -4dB using seismic power dB convention (10^(±dB/10))
- **Coordinates**: Physical coordinates for Z-axis (depth), logical coordinates for X/Y by default

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Author

Created for seismic data visualization and analysis workflows.
