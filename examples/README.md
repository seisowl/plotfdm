# Example FDM Files

This directory contains sample FDM files for testing and demonstration:

## Files

### shot_topo.fdm
- **Type**: Seismic shot data
- **Description**: Contains seismic data with topographic information
- **Data characteristics**: Mixed positive/negative values (seismic waveforms)
- **Best viewed with**: Default gray_r colormap

### vel_topo.fdm  
- **Type**: Velocity model
- **Description**: Velocity model with topographic surface
- **Data characteristics**: All positive values (velocity)
- **Best viewed with**: TgsBanded colormap

## Quick Start

```bash
# View summary plots
python plotfdm.py examples/shot_topo.fdm --summary
python plotfdm.py examples/vel_topo.fdm --summary

# Interactive mode
python plotfdm.py examples/shot_topo.fdm

# Specific slices
python plotfdm.py examples/vel_topo.fdm --slice z --index 10

# File information
python plotfdm.py examples/shot_topo.fdm --info
```

## Expected Results

- **shot_topo.fdm**: Should automatically use gray_r colormap with RMS -4dB scaling
- **vel_topo.fdm**: Should automatically use TgsBanded colormap with appropriate scaling

Both files demonstrate the automatic colormap selection and scaling features of the plotfdm tool.
