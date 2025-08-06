#!/usr/bin/env python

"""
FDM File Plotting Tool

Usage:
    python plotfdm.py <fdm_file> [options]

This script reads FDM files and provides plotting capabilities for        print("Example usage:")
        print("  python plotfdm.py data.fdm --summary")
        print("  python plotfdm.py data.fdm --slice y --index 5")
        print("  python plotfdm.py data.fdm --info")
        print("  python plotfdm.py data.fdm  # plots middle y-slice w    parser.add_argument('--vmax', type=float, default=None,
                       help='Maximum value for colormap scaling (overrides automatic scaling)')
    parser.add_argument('--horizon', type=str, default=None,
                       help='Overlay horizon file (X Y Z format) on plots')
    
    if len(sys.argv) == 1:logical coords (default)")
        print("  python plotfdm.py data.fdm --physical  # use physical coordinates")
        print("  python plotfdm.py data.fdm --no-display  # save to file instead of displaying")
        print("  python plotfdm.py data.fdm --save plot.png  # save to specific file")ismic data.
"""

import sys
import argparse
import numpy as np
import matplotlib
# Don't set backend here - let it be determined later based on mode
import matplotlib.pyplot as plt
from fdm import FDM
import custom_colormaps  # Import custom colormaps
import time  # For double-key detection


class InteractiveGainHandler:
    """Handles keyboard shortcuts for interactive gain/dim control"""
    
    def __init__(self, im_objects, initial_vmin, initial_vmax):
        self.im_objects = im_objects if isinstance(im_objects, list) else [im_objects]
        self.current_vmin = initial_vmin
        self.current_vmax = initial_vmax
        self.last_key_time = 0
        self.last_key = None
        self.double_key_threshold = 0.5  # seconds
        
    def on_key_press(self, event):
        """Handle key press events for gain/dim control"""
        if event.key not in ['g', 'd']:
            return
            
        current_time = time.time()
        is_double_key = (current_time - self.last_key_time < self.double_key_threshold and 
                        event.key == self.last_key)
        
        # Determine gain/dim amount
        if is_double_key:
            # 3dB change
            db_change = 3.0
            print(f"{'Gain' if event.key == 'g' else 'Dim'} {db_change}dB")
        else:
            # 1.5dB change
            db_change = 1.5
            print(f"{'Gain' if event.key == 'g' else 'Dim'} {db_change}dB")
        
        # Convert dB to linear scale factor (seismic power dB convention)
        # Gain: multiply range by 10^(-dB/10) (smaller range = higher gain)
        # Dim: multiply range by 10^(+dB/10) (larger range = lower gain)
        if event.key == 'g':  # Gain
            scale_factor = 10**(-db_change/10)
        else:  # Dim
            scale_factor = 10**(db_change/10)
        
        # Calculate new scaling
        current_range = self.current_vmax - self.current_vmin
        new_range = current_range * scale_factor
        range_center = (self.current_vmin + self.current_vmax) / 2
        
        self.current_vmin = range_center - new_range / 2
        self.current_vmax = range_center + new_range / 2
        
        # Update all image objects
        for im in self.im_objects:
            im.set_clim(self.current_vmin, self.current_vmax)
        
        # Redraw the plot
        if event.canvas:
            event.canvas.draw_idle()
        else:
            plt.draw()
        
        # Update timing for double-key detection
        self.last_key_time = current_time
        self.last_key = event.key
        
        print(f"Scale range: [{self.current_vmin:.3f}, {self.current_vmax:.3f}]")


def get_smart_colormap_and_scaling(data, cmap=None, vmin=None, vmax=None):
    """
    Automatically determine appropriate colormap and scaling based on data characteristics
    
    Args:
        data: NumPy array of the data
        cmap: User-specified colormap (overrides automatic selection)
        vmin: User-specified minimum value (overrides automatic scaling)
        vmax: User-specified maximum value (overrides automatic scaling)
    
    Returns:
        tuple: (colormap_name, vmin, vmax)
    """
    data_min = np.min(data)
    data_max = np.max(data)
    data_rms = np.sqrt(np.mean(data**2))
    
    # Automatic colormap selection (unless user specified one)
    if cmap is None:
        # Check if data has both positive and negative values (seismic-like)
        if data_min < 0 and data_max > 0:
            # Seismic data with positive and negative values
            colormap = 'gray_r'  # Grayscale reversed
        else:
            # All positive data (velocity-like)
            colormap = 'TgsBanded'
    else:
        colormap = cmap
    
    # Automatic scaling: +/- RMS - 4dB (unless user specified min/max)
    if vmin is not None and vmax is not None:
        # Use user-specified scaling
        return colormap, vmin, vmax
    
    # -4dB means the display range is 4dB below RMS (requiring +4dB gain to reach RMS)
    # For seismic data, use power/energy dB: 10^(+4/10) ≈ 2.512 (larger range = dimmer display)
    scale_factor = 10**(4/10)  # ≈ 2.512 (seismic power dB convention)
    scale_limit = data_rms * scale_factor
    
    # For data with both positive and negative values
    if data_min < 0 and data_max > 0:
        auto_vmin = -scale_limit
        auto_vmax = scale_limit
    else:
        # For all-positive data, center around the data range
        data_center = (data_min + data_max) / 2
        auto_vmin = max(data_min, data_center - scale_limit)
        auto_vmax = min(data_max, data_center + scale_limit)
    
    # Use user-specified values if provided, otherwise use automatic
    final_vmin = vmin if vmin is not None else auto_vmin
    final_vmax = vmax if vmax is not None else auto_vmax
    
    return colormap, final_vmin, final_vmax


def load_horizon(horizon_file, dz_grid=None, z_min=None, z_max=None):
    """
    Load horizon data from a text file
    
    Args:
        horizon_file: Path to horizon file with X Y Z format
        dz_grid: If specified, replicate horizon at depth intervals to create grid
        z_min: Minimum Z value for grid replication (auto-detected if None)
        z_max: Maximum Z value for grid replication (auto-detected if None)
    
    Returns:
        tuple: (x_coords, y_coords, z_coords, level_info) as numpy arrays
               level_info contains the level number for each point (for grid separation)
    """
    try:
        data = np.loadtxt(horizon_file)
        if data.ndim == 1:
            # Single point
            hx, hy, hz = data[0], data[1], data[2]
        else:
            # Multiple points
            hx, hy, hz = data[:, 0], data[:, 1], data[:, 2]
        
        # If grid replication is requested
        if dz_grid is not None and dz_grid > 0:
            # Determine Z range for grid - use FDM range, not horizon range
            if z_min is None:
                z_min = 0  # Start from FDM minimum
            if z_max is None:
                z_max = 1000  # Use reasonable default
            
            # Original horizon data
            hx_orig = hx if hasattr(hx, '__len__') else np.array([hx])
            hy_orig = hy if hasattr(hy, '__len__') else np.array([hy])
            hz_orig = hz if hasattr(hz, '__len__') else np.array([hz])
            
            # Create horizon grid by replicating only in Z direction (depth levels)
            hx_grid = []
            hy_grid = []
            hz_grid = []
            level_grid = []
            
            # Generate Z levels: original + n*dz_grid until we exceed z_max
            level = 0
            while True:
                # Shift horizon down by level * dz_grid from original position
                hz_shifted = hz_orig + level * dz_grid
                
                # Check if any part of this replica would be within the FDM Z range
                if np.min(hz_shifted) > z_max:
                    break  # All points are below the FDM range, stop here
                
                # Only include points that fall within the FDM Z range
                valid_mask = (hz_shifted >= z_min) & (hz_shifted <= z_max)
                
                # If any points are valid for this level, add them
                if np.any(valid_mask):
                    hx_grid.extend(hx_orig[valid_mask])
                    hy_grid.extend(hy_orig[valid_mask])
                    hz_grid.extend(hz_shifted[valid_mask])
                    level_grid.extend([level] * np.sum(valid_mask))  # Track level for each point
                
                level += 1
            
            return np.array(hx_grid), np.array(hy_grid), np.array(hz_grid), np.array(level_grid)
        
        # For single horizon, all points are level 0
        level_info = np.zeros(len(hx) if hasattr(hx, '__len__') else 1, dtype=int)
        return hx, hy, hz, level_info
        
    except Exception as e:
        print(f"Warning: Could not load horizon file {horizon_file}: {e}")
        return None, None, None, None


def draw_mesh_lines(ax, fdm, axis, use_logical, dz_mesh=None, dx_mesh=None, dz_horizon=None):
    """
    Draw mesh grid lines on a plot
    
    Args:
        ax: matplotlib axes object
        fdm: FDM object with loaded data
        axis: Which axis slice ('x', 'y', or 'z')  
        use_logical: Whether to use logical coordinates
        dz_mesh: Depth interval for horizontal lines (None to disable)
        dx_mesh: Horizontal interval for vertical lines (None for auto/disable)
        dz_horizon: Horizon depth interval (used for dx_mesh default)
    """
    # Debug output
    print(f"Drawing mesh lines for {axis}-slice: dz_mesh={dz_mesh}, dx_mesh={dx_mesh}, dz_horizon={dz_horizon}")
    
    # Determine default dx_mesh if not specified
    if dx_mesh is None and (dz_horizon is not None or dz_mesh is not None):
        # Default to 2x the relevant depth interval
        if dz_horizon is not None:
            dx_mesh = 2 * dz_horizon
        elif dz_mesh is not None:
            dx_mesh = 2 * dz_mesh
    
    # Get coordinate extents
    x_extent = ax.get_xlim()
    y_extent = ax.get_ylim()
    
    # Get FDM coordinates for scaling
    fdm_x, fdm_y, fdm_z = fdm.get_coordinates(logical=False)
    
    if axis.lower() in ['x', 'y']:
        # For X and Y slices (showing depth), add horizontal and vertical lines
        
        # Horizontal lines (depth levels) - only if dz_mesh is specified
        if dz_mesh is not None and dz_mesh > 0:
            z_min, z_max = fdm_z[0], fdm_z[-1]
            print(f"  Adding horizontal mesh lines: z_range=[{z_min:.1f}, {z_max:.1f}], y_extent={y_extent}")
            # Generate horizontal lines at dz_mesh intervals
            z_start = int(z_min / dz_mesh) * dz_mesh
            z_positions = np.arange(z_start, z_max + dz_mesh/2, dz_mesh)
            print(f"  Horizontal line positions: {z_positions}")
            
            lines_drawn = 0
            for z_pos in z_positions:
                # For depth slices, Y extent is reversed (z_max, z_min), so check both ways
                if min(y_extent) <= z_pos <= max(y_extent):
                    ax.axhline(y=z_pos, color='black', linewidth=0.5, alpha=0.7, zorder=3)
                    lines_drawn += 1
            print(f"  Drew {lines_drawn} horizontal mesh lines")
        
        # Vertical lines - only if dx_mesh is specified
        if dx_mesh is not None and dx_mesh > 0:
            if axis.lower() == 'y':
                # Y-slice: vertical lines in X direction
                if use_logical:
                    dx_grid_logical = dx_mesh / fdm.header.dx
                    x_start = int(x_extent[0] / dx_grid_logical) * dx_grid_logical
                    x_positions = np.arange(x_start, x_extent[1] + dx_grid_logical/2, dx_grid_logical)
                else:
                    x_start = int(x_extent[0] / dx_mesh) * dx_mesh
                    x_positions = np.arange(x_start, x_extent[1] + dx_mesh/2, dx_mesh)
                
                for x_pos in x_positions:
                    if x_extent[0] <= x_pos <= x_extent[1]:
                        ax.axvline(x=x_pos, color='black', linewidth=0.5, alpha=0.7, zorder=3)
            
            elif axis.lower() == 'x':
                # X-slice: vertical lines in Y direction  
                if use_logical:
                    dy_grid_logical = dx_mesh / fdm.header.dy
                    x_start = int(x_extent[0] / dy_grid_logical) * dy_grid_logical  # x_extent is Y for X-slice
                    y_positions = np.arange(x_start, x_extent[1] + dy_grid_logical/2, dy_grid_logical)
                else:
                    x_start = int(x_extent[0] / dx_mesh) * dx_mesh
                    y_positions = np.arange(x_start, x_extent[1] + dx_mesh/2, dx_mesh)
                
                for y_pos in y_positions:
                    if x_extent[0] <= y_pos <= x_extent[1]:
                        ax.axvline(x=y_pos, color='black', linewidth=0.5, alpha=0.7, zorder=3)


def plot_slice(fdm, axis='z', index=0, cmap=None, title=None, use_logical=True, interactive=False, vmin=None, vmax=None, horizon_file=None, dz_horizon=None, dz_mesh=None, dx_mesh=None):
    """
    Plot a 2D slice from the 3D FDM data
    
    Args:
        fdm: FDM object with loaded data
        axis: Which axis to slice ('x', 'y', or 'z')
        index: Index of the slice
        cmap: Colormap for the plot (None for automatic selection)
        title: Custom title for the plot
        use_logical: Use logical coordinates instead of physical (default: True)
                    Note: Z-axis always uses physical coordinates for depth visualization
        interactive: Enable interactive gain/dim controls (default: False)
        vmin: Minimum value for colormap scaling (None for automatic)
        vmax: Maximum value for colormap scaling (None for automatic)
        horizon_file: Path to horizon file for overlay (None for no overlay)
        dz_horizon: Depth interval for horizon grid replication (None for single horizon)
        dz_mesh: Depth interval for horizontal mesh lines (None to disable)
        dx_mesh: Horizontal interval for vertical mesh lines (None for auto/disable)
    """
    # Get coordinates - use logical for X,Y but physical for Z (depth visualization)
    if use_logical:
        x, y, _ = fdm.get_coordinates(logical=True)
        coord_label_xy = "Grid Index"
    else:
        x, y, _ = fdm.get_coordinates(logical=False)
        coord_label_xy = "Physical Coordinate"
    
    # Always use physical coordinates for Z-axis (depth visualization)
    _, _, z = fdm.get_coordinates(logical=False)
    coord_label_z = "Physical Coordinate (Depth)"
    
    if axis.lower() == 'z':
        if index >= fdm.header.nz:
            print(f"Warning: z-index {index} >= nz={fdm.header.nz}, using last slice")
            index = fdm.header.nz - 1
        
        # Check if we should show horizon data instead of FDM data
        if horizon_file is not None:
            # Create horizon-only image for Z-slice
            if fdm.header.ny == 1:
                y_extent = [y[0] - 5*fdm.header.dy, y[-1] + 5*fdm.header.dy]
                data_slice_shape = (10, fdm.header.nx)
            else:
                y_extent = [y[0], y[-1]]
                data_slice_shape = (fdm.header.ny, fdm.header.nx)
            
            # Initialize horizon image with NaN
            data_slice = np.full(data_slice_shape, np.nan)
            
            # Load horizon data
            fdm_z = fdm.get_coordinates(logical=False)[2]
            z_min, z_max = fdm_z[0], fdm_z[-1]
            hx, hy, hz, levels = load_horizon(horizon_file, dz_grid=dz_horizon, z_min=z_min, z_max=z_max)
            
            if hx is not None and hy is not None and hz is not None:
                # Convert horizon coordinates to grid indices
                hx_indices = np.round((hx - fdm.header.x0) / fdm.header.dx).astype(int)
                hy_indices = np.round((hy - fdm.header.y0) / fdm.header.dy).astype(int)
                
                # Fill horizon data into the image
                for i, (xi, yi, hzi) in enumerate(zip(hx_indices, hy_indices, hz)):
                    if 0 <= xi < fdm.header.nx and 0 <= yi < fdm.header.ny:
                        if fdm.header.ny == 1:
                            # Fill all Y rows for ny=1 case
                            for row in range(data_slice_shape[0]):
                                data_slice[row, xi] = hzi
                        else:
                            data_slice[yi, xi] = hzi
        else:
            # Original FDM data behavior
            data_slice = fdm.data[:, :, index]  # (ny, nx, z_index) -> (ny, nx)
            # If ny=1, replicate the data to make it visible
            if fdm.header.ny == 1:
                data_slice = np.tile(data_slice, (10, 1))  # Replicate 10 times along Y
                y_extent = [y[0] - 5*fdm.header.dy, y[-1] + 5*fdm.header.dy]
            else:
                y_extent = [y[0], y[-1]]
        
        extent = [x[0], x[-1], y_extent[0], y_extent[1]]
        xlabel, ylabel = 'X', 'Y'
        slice_coord = z[index]
        axis_name = 'Z'
    elif axis.lower() == 'y':
        if index >= fdm.header.ny:
            print(f"Warning: y-index {index} >= ny={fdm.header.ny}, using last slice")
            index = fdm.header.ny - 1
        data_slice = fdm.data[index, :, :]  # (y_index, nx, nz) -> (nx, nz)
        data_slice = data_slice.T  # Transpose to (nz, nx) for proper orientation
        data_slice = np.flipud(data_slice)  # Flip vertically so Z=0 data is at top
        extent = [x[0], x[-1], z[-1], z[0]]  # Reverse Z to put 0 at top
        xlabel, ylabel = 'X', 'Z'
        slice_coord = y[index]
        axis_name = 'Y'
    elif axis.lower() == 'x':
        if index >= fdm.header.nx:
            print(f"Warning: x-index {index} >= nx={fdm.header.nx}, using last slice")
            index = fdm.header.nx - 1
        data_slice = fdm.data[:, index, :]  # (ny, x_index, nz) -> (ny, nz)
        data_slice = data_slice.T  # Transpose to (nz, ny) for proper orientation
        # If ny=1, replicate the data to make it visible  
        if fdm.header.ny == 1:
            data_slice = np.tile(data_slice, (1, 10))  # Replicate 10 times along Y (now second dimension)
            y_extent = [y[0] - 5*fdm.header.dy, y[-1] + 5*fdm.header.dy]
        else:
            y_extent = [y[0], y[-1]]
        data_slice = np.flipud(data_slice)  # Flip vertically so Z=0 data is at top
        extent = [y_extent[0], y_extent[1], z[-1], z[0]]  # Reverse Z to put 0 at top
        xlabel, ylabel = 'Y', 'Z'
        slice_coord = x[index]
        axis_name = 'X'
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    
    # Get smart colormap and scaling
    if axis.lower() == 'z' and horizon_file is not None:
        # For Z-slice with horizon data, use horizon-specific scaling
        horizon_mask = ~np.isnan(data_slice)
        if np.any(horizon_mask):
            hz_min = np.nanmin(data_slice)
            hz_max = np.nanmax(data_slice)
            colormap = cmap if cmap is not None else 'TgsBanded'  # Use user's colormap or default for depth visualization
            vmin_final = hz_min if vmin is None else vmin
            vmax_final = hz_max if vmax is None else vmax
            scaling_type = "horizon" if (vmin is None and vmax is None) else "manual"
            print(f"Using colormap: {colormap}, {scaling_type} scaling: [{vmin_final:.3f}, {vmax_final:.3f}]")
        else:
            # Fallback if no horizon data
            colormap, vmin_final, vmax_final = 'gray', 0, 1
            print(f"Using colormap: {colormap}, no horizon data found, using fallback")
    else:
        # Original behavior for other slices or Z-slice without horizon
        colormap, vmin_final, vmax_final = get_smart_colormap_and_scaling(data_slice, cmap, vmin, vmax)
        if vmin is not None or vmax is not None:
            print(f"Using colormap: {colormap}, manual scaling: [{vmin_final:.3f}, {vmax_final:.3f}]")
        else:
            print(f"Using colormap: {colormap}, auto scaling: [{vmin_final:.3f}, {vmax_final:.3f}]")
    
    plt.figure(figsize=(10, 8))
    im = plt.imshow(data_slice, aspect='auto', origin='lower', 
                    extent=extent, cmap=colormap, 
                    vmin=vmin_final, vmax=vmax_final)
    plt.colorbar(im, label='Value')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if title is None:
        if axis.lower() == 'z' and horizon_file is not None:
            title = f'Horizon Depth Map at Z={slice_coord:.2f}'
        else:
            title = f'{axis_name}-slice at {axis_name}={slice_coord:.2f}'
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add horizon overlay if provided
    if horizon_file is not None:
        # Get FDM Z coordinates for grid range
        fdm_z = fdm.get_coordinates(logical=False)[2]
        z_min, z_max = fdm_z[0], fdm_z[-1]
        
        hx, hy, hz, levels = load_horizon(horizon_file, dz_grid=dz_horizon, z_min=z_min, z_max=z_max)
        if hx is not None and hy is not None and hz is not None:
            # Get FDM coordinates for horizon transformation
            fdm_x, fdm_y, fdm_z = fdm.get_coordinates(logical=False)  # Use physical coordinates
            
            # Determine which coordinates to use based on the slice axis
            if axis.lower() == 'z':
                # Z-slice with horizon file already shows horizon data directly
                # No additional overlay needed since we're showing horizon data instead of FDM data
                pass
                    
            elif axis.lower() == 'y':
                # Y-slice: show horizons in XZ plane
                # Check if horizon intersects this Y level
                fdm_y_phys = fdm.get_coordinates(logical=False)[1]
                y_tolerance = abs(fdm_y_phys[1] - fdm_y_phys[0]) if len(fdm_y_phys) > 1 else 12.5
                y_slice_phys = fdm_y_phys[index]  # Always use physical for comparison with horizon
                mask = np.abs(hy - y_slice_phys) <= y_tolerance
                if np.any(mask):
                    if use_logical:
                        hx_plot = (hx[mask] - fdm.header.x0) / fdm.header.dx
                    else:
                        hx_plot = hx[mask]
                    hz_plot = hz[mask]
                    
                    # Plot horizons - use level information for proper separation
                    if dz_horizon is not None:
                        # Get levels for the masked points
                        levels_plot = levels[mask]
                        
                        # Plot each level separately to avoid connecting different levels
                        unique_levels = np.unique(levels_plot)
                        for level in unique_levels:
                            level_mask = levels_plot == level
                            level_x = hx_plot[level_mask]
                            level_z = hz_plot[level_mask]
                            if len(level_x) > 0:
                                # Sort points within this level by X coordinate
                                sort_idx = np.argsort(level_x)
                                plt.plot(level_x[sort_idx], level_z[sort_idx], 'cyan', linewidth=1, zorder=10)
                        
                        # Add vertical grid lines for uniform spacing (now handled by mesh system)
                        if dx_mesh is None and dz_horizon is not None:
                            # Default behavior: use 2*dz_horizon for vertical lines when dz_horizon is active
                            effective_dx_mesh = 2 * dz_horizon
                            x_extent = plt.xlim()
                            
                            if use_logical:
                                dx_grid_logical = effective_dx_mesh / fdm.header.dx
                                x_start = int(x_extent[0] / dx_grid_logical) * dx_grid_logical
                                x_positions = np.arange(x_start, x_extent[1] + dx_grid_logical/2, dx_grid_logical)
                            else:
                                x_start = int(x_extent[0] / effective_dx_mesh) * effective_dx_mesh
                                x_positions = np.arange(x_start, x_extent[1] + effective_dx_mesh/2, effective_dx_mesh)
                            
                            for x_pos in x_positions:
                                if x_extent[0] <= x_pos <= x_extent[1]:
                                    plt.axvline(x=x_pos, color='cyan', linewidth=0.5, alpha=0.7, zorder=5)
                    else:
                        # Sort by X coordinate for single horizon
                        sort_idx = np.argsort(hx_plot)
                        plt.plot(hx_plot[sort_idx], hz_plot[sort_idx], 'cyan', linewidth=1, zorder=10)
                    # Note: Z coordinates are flipped in the display
                    
            elif axis.lower() == 'x':
                # X-slice: show horizons in YZ plane
                # Check if horizon intersects this X level
                x_tolerance = abs(fdm_x[1] - fdm_x[0]) if len(fdm_x) > 1 else 12.5
                x_slice_phys = fdm_x[index]  # Always use physical for comparison with horizon
                mask = np.abs(hx - x_slice_phys) <= x_tolerance
                if np.any(mask):
                    if use_logical:
                        hy_plot = (hy[mask] - fdm.header.y0) / fdm.header.dy
                    else:
                        hy_plot = hy[mask]
                    hz_plot = hz[mask]
                    
                    # Plot horizons - use level information for proper separation
                    if dz_horizon is not None:
                        # Get levels for the masked points
                        levels_plot = levels[mask]
                        
                        # Plot each level separately to avoid connecting different levels
                        unique_levels = np.unique(levels_plot)
                        for level in unique_levels:
                            level_mask = levels_plot == level
                            level_y = hy_plot[level_mask]
                            level_z = hz_plot[level_mask]
                            if len(level_y) > 0:
                                # Sort points within this level by Y coordinate
                                sort_idx = np.argsort(level_y)
                                plt.plot(level_y[sort_idx], level_z[sort_idx], 'cyan', linewidth=1, zorder=10)
                        
                        # Add vertical grid lines for uniform spacing (now handled by mesh system)
                        if dx_mesh is None and dz_horizon is not None:
                            # Default behavior: use 2*dz_horizon for vertical lines when dz_horizon is active
                            effective_dx_mesh = 2 * dz_horizon
                            y_extent = plt.xlim()  # For X-slice, xlim is Y coordinate
                            
                            if use_logical:
                                dy_grid_logical = effective_dx_mesh / fdm.header.dy
                                y_start = int(y_extent[0] / dy_grid_logical) * dy_grid_logical
                                y_positions = np.arange(y_start, y_extent[1] + dy_grid_logical/2, dy_grid_logical)
                            else:
                                y_start = int(y_extent[0] / effective_dx_mesh) * effective_dx_mesh
                                y_positions = np.arange(y_start, y_extent[1] + effective_dx_mesh/2, effective_dx_mesh)
                            
                            for y_pos in y_positions:
                                if y_extent[0] <= y_pos <= y_extent[1]:
                                    plt.axvline(x=y_pos, color='cyan', linewidth=0.5, alpha=0.7, zorder=5)
                    else:
                        # Sort by Y coordinate for single horizon
                        sort_idx = np.argsort(hy_plot)
                        plt.plot(hy_plot[sort_idx], hz_plot[sort_idx], 'cyan', linewidth=1, zorder=10)
    
    # Add mesh lines (independent of horizon)
    draw_mesh_lines(plt.gca(), fdm, axis, use_logical, dz_mesh, dx_mesh, dz_horizon)
    
    plt.tight_layout()
    
    # Add interactive controls if requested
    if interactive:
        gain_handler = InteractiveGainHandler(im, vmin_final, vmax_final)
        fig = plt.gcf()
        
        # Ensure the figure window can receive focus
        if hasattr(fig.canvas, 'manager') and hasattr(fig.canvas.manager, 'window'):
            try:
                # Try to activate and focus the window
                fig.canvas.manager.window.activateWindow()
                fig.canvas.manager.window.raise_()
                fig.canvas.setFocusPolicy(2)  # Qt.StrongFocus
                fig.canvas.setFocus()
                print("Figure window focused successfully")
            except Exception as e:
                print(f"Could not set window focus: {e}")
        
        # Connect keyboard events for interactive gain controls
        def unified_key_handler(event):
            gain_handler.on_key_press(event)
        
        cid = fig.canvas.mpl_connect('key_press_event', unified_key_handler)
        
        print(f"\nInteractive controls connected (connection id: {cid})")
        print("  g     - Gain +1.5dB")
        print("  gg    - Gain +3.0dB (press g twice quickly)")
        print("  d     - Dim -1.5dB") 
        print("  dd    - Dim -3.0dB (press d twice quickly)")
        print("  Focus the plot window and use keys to adjust gain")
        print(f"  Backend: {matplotlib.get_backend()}")


def plot_summary(fdm, use_logical=True, interactive=False, vmin=None, vmax=None, horizon_file=None, dz_horizon=None, dz_mesh=None, dx_mesh=None, cmap=None):
    """Plot summary showing slices along all three axes
    Note: Z-axis always uses physical coordinates for depth visualization
    
    Args:
        fdm: FDM object with loaded data
        use_logical: Use logical coordinates for X,Y axes (default: True)
        interactive: Enable interactive gain/dim controls (default: False)
        vmin: Minimum value for colormap scaling (None for automatic)
        vmax: Maximum value for colormap scaling (None for automatic)
        horizon_file: Path to horizon file for overlay (None for no overlay)
        dz_horizon: Depth interval for horizon grid replication (None for single horizon)
        dz_mesh: Depth interval for horizontal mesh lines (None to disable)
        dx_mesh: Horizontal interval for vertical mesh lines (None for auto/disable)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Middle slices
    z_mid = fdm.header.nz // 2
    y_mid = fdm.header.ny // 2
    x_mid = fdm.header.nx // 2
    
    # If horizon is provided, try to find a Z-slice that intersects with horizon data
    if horizon_file is not None:
        fdm_z = fdm.get_coordinates(logical=False)[2]
        z_min, z_max = fdm_z[0], fdm_z[-1]
        
        # Load horizon to check available Z levels
        hx_temp, hy_temp, hz_temp, levels_temp = load_horizon(horizon_file, dz_grid=dz_horizon, z_min=z_min, z_max=z_max)
        if hx_temp is not None and hz_temp is not None:
            # Find Z-slice indices that have nearby horizon data
            z_tolerance = abs(fdm_z[1] - fdm_z[0]) if len(fdm_z) > 1 else 12.5
            
            for z_idx in range(fdm.header.nz):
                z_slice_phys = fdm_z[z_idx] 
                mask = np.abs(hz_temp - z_slice_phys) <= z_tolerance
                if np.any(mask):
                    z_mid = z_idx  # Use first Z-slice that has horizon data
                    print(f"Found horizon data at Z-slice index {z_idx} (Z={z_slice_phys:.2f}), using for summary")
                    break
    
    # Get coordinates - use logical for X,Y but physical for Z (depth visualization)
    if use_logical:
        x, y, _ = fdm.get_coordinates(logical=True)
        coord_label_xy = "Grid Index"
    else:
        x, y, _ = fdm.get_coordinates(logical=False)
        coord_label_xy = "Physical Coordinate"
    
    # Always use physical coordinates for Z-axis (depth visualization)
    _, _, z = fdm.get_coordinates(logical=False)
    coord_label_z = "Physical Coordinate (Depth)"
    
    # Z-slice (XY plane)
    if horizon_file is not None:
        # When horizon is provided, show horizon data instead of FDM data
        # Create a horizon-only image
        if fdm.header.ny == 1:
            y_extent = [y[0] - 5*fdm.header.dy, y[-1] + 5*fdm.header.dy]
            # Create expanded shape for ny=1 case
            data_z_shape = (10, fdm.header.nx)
        else:
            y_extent = [y[0], y[-1]]
            data_z_shape = (fdm.header.ny, fdm.header.nx)
        
        # Initialize horizon image with NaN (will show as transparent/white)
        data_z = np.full(data_z_shape, np.nan)
        
        # Load horizon data for this summary
        fdm_z = fdm.get_coordinates(logical=False)[2] 
        z_min, z_max = fdm_z[0], fdm_z[-1]
        hx, hy, hz, levels = load_horizon(horizon_file, dz_grid=dz_horizon, z_min=z_min, z_max=z_max)
        
        if hx is not None and hy is not None and hz is not None:
            # Convert horizon coordinates to grid indices
            if use_logical:
                hx_indices = np.round((hx - fdm.header.x0) / fdm.header.dx).astype(int)
                hy_indices = np.round((hy - fdm.header.y0) / fdm.header.dy).astype(int)
            else:
                hx_indices = np.round((hx - fdm.header.x0) / fdm.header.dx).astype(int)
                hy_indices = np.round((hy - fdm.header.y0) / fdm.header.dy).astype(int)
            
            # Fill horizon data into the image
            for i, (xi, yi, hzi) in enumerate(zip(hx_indices, hy_indices, hz)):
                if 0 <= xi < fdm.header.nx and 0 <= yi < fdm.header.ny:
                    if fdm.header.ny == 1:
                        # Fill all Y rows for ny=1 case
                        for row in range(data_z_shape[0]):
                            data_z[row, xi] = hzi
                    else:
                        data_z[yi, xi] = hzi
        
        # Use horizon data range for colormap scaling
        horizon_mask = ~np.isnan(data_z)
        if np.any(horizon_mask):
            hz_min = np.nanmin(data_z)
            hz_max = np.nanmax(data_z)
            colormap_z = cmap if cmap is not None else 'TgsBanded'  # Use user's colormap or default for depth visualization
            vmin_z = hz_min if vmin is None else vmin
            vmax_z = hz_max if vmax is None else vmax
            scaling_type = "horizon" if (vmin is None and vmax is None) else "manual"
            print(f"Z-slice (horizon): colormap={colormap_z}, {scaling_type} scaling=[{vmin_z:.3f}, {vmax_z:.3f}]")
        else:
            # Fallback if no horizon data
            colormap_z, vmin_z, vmax_z = 'gray', 0, 1
            print(f"Z-slice (horizon): no horizon data found, using fallback")
        
        im1 = axes[0, 0].imshow(data_z, aspect='auto', origin='lower',
                               extent=[x[0], x[-1], y_extent[0], y_extent[1]], 
                               cmap=colormap_z, vmin=vmin_z, vmax=vmax_z)
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        axes[0, 0].set_title(f'Horizon Depth Map')
        axes[0, 0].grid(True, alpha=0.3)
        
    else:
        # Original behavior: show FDM data when no horizon file
        data_z = fdm.data[:, :, z_mid]
        # If ny=1, replicate the data to make it visible
        if fdm.header.ny == 1:
            data_z = np.tile(data_z, (10, 1))  # Replicate 10 times along Y
            y_extent = [y[0] - 5*fdm.header.dy, y[-1] + 5*fdm.header.dy]
        else:
            y_extent = [y[0], y[-1]]
        
        # Get smart colormap and scaling for Z-slice
        colormap_z, vmin_z, vmax_z = get_smart_colormap_and_scaling(data_z, None, vmin, vmax)
        scaling_type = "manual" if (vmin is not None or vmax is not None) else "auto"
        print(f"Z-slice: colormap={colormap_z}, {scaling_type} scaling=[{vmin_z:.3f}, {vmax_z:.3f}]")
        
        im1 = axes[0, 0].imshow(data_z, aspect='auto', origin='lower',
                               extent=[x[0], x[-1], y_extent[0], y_extent[1]], 
                               cmap=colormap_z, vmin=vmin_z, vmax=vmax_z)
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        axes[0, 0].set_title(f'Z-slice at Z={z[z_mid]:.2f}')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Y-slice (XZ plane) - flip data and reverse Z extent for proper orientation
    data_y = fdm.data[y_mid, :, :].T  # Transpose to (nz, nx)
    data_y = np.flipud(data_y)  # Flip vertically so Z=0 data is at top
    
    # Get smart colormap and scaling for Y-slice
    colormap_y, vmin_y, vmax_y = get_smart_colormap_and_scaling(data_y, None, vmin, vmax)
    print(f"Y-slice: colormap={colormap_y}, {scaling_type} scaling=[{vmin_y:.3f}, {vmax_y:.3f}]")
    
    im2 = axes[0, 1].imshow(data_y, aspect='auto', origin='lower',
                           extent=[x[0], x[-1], z[-1], z[0]], 
                           cmap=colormap_y, vmin=vmin_y, vmax=vmax_y)
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Z')
    axes[0, 1].set_title(f'Y-slice at Y={y[y_mid]:.2f}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add horizon overlay for Y-slice
    if horizon_file is not None:
        # Get FDM Z coordinates for grid range
        fdm_z = fdm.get_coordinates(logical=False)[2]
        z_min, z_max = fdm_z[0], fdm_z[-1]
        
        hx, hy, hz, levels = load_horizon(horizon_file, dz_grid=dz_horizon, z_min=z_min, z_max=z_max)
        if hx is not None and hy is not None and hz is not None:
            # Check if horizon intersects this Y level
            fdm_y_phys = fdm.get_coordinates(logical=False)[1]
            y_tolerance = abs(fdm_y_phys[1] - fdm_y_phys[0]) if len(fdm_y_phys) > 1 else 12.5
            y_slice_phys = fdm_y_phys[y_mid]  # Always use physical for comparison with horizon
            mask = np.abs(hy - y_slice_phys) <= y_tolerance
            if np.any(mask):
                if use_logical:
                    hx_plot = (hx[mask] - fdm.header.x0) / fdm.header.dx
                else:
                    hx_plot = hx[mask]
                hz_plot = hz[mask]
                
                # Plot horizons - use level information for proper separation
                if dz_horizon is not None:
                    # Get levels for the masked points
                    levels_plot = levels[mask]
                    
                    # Plot each level separately to avoid connecting different levels
                    unique_levels = np.unique(levels_plot)
                    for level in unique_levels:
                        level_mask = levels_plot == level
                        level_x = hx_plot[level_mask]
                        level_z = hz_plot[level_mask]
                        if len(level_x) > 0:
                            # Sort points within this level by X coordinate
                            sort_idx = np.argsort(level_x)
                            axes[0, 1].plot(level_x[sort_idx], level_z[sort_idx], 'cyan', linewidth=1, zorder=10)
                    
                    # Add vertical grid lines for uniform spacing (now handled by mesh system)
                    if dx_mesh is None and dz_horizon is not None:
                        # Default behavior: use 2*dz_horizon for vertical lines when dz_horizon is active
                        effective_dx_mesh = 2 * dz_horizon
                        x_extent = axes[0, 1].get_xlim()
                        
                        if use_logical:
                            dx_grid_logical = effective_dx_mesh / fdm.header.dx
                            x_start = int(x_extent[0] / dx_grid_logical) * dx_grid_logical
                            x_positions = np.arange(x_start, x_extent[1] + dx_grid_logical/2, dx_grid_logical)
                        else:
                            x_start = int(x_extent[0] / effective_dx_mesh) * effective_dx_mesh
                            x_positions = np.arange(x_start, x_extent[1] + effective_dx_mesh/2, effective_dx_mesh)
                        
                        for x_pos in x_positions:
                            if x_extent[0] <= x_pos <= x_extent[1]:
                                axes[0, 1].axvline(x=x_pos, color='cyan', linewidth=0.5, alpha=0.7, zorder=5)
                else:
                    # Sort by X coordinate for single horizon
                    sort_idx = np.argsort(hx_plot)
                    axes[0, 1].plot(hx_plot[sort_idx], hz_plot[sort_idx], 'cyan', linewidth=1, zorder=10)
    
    # X-slice (YZ plane) - flip data and reverse Z extent for proper orientation
    data_x = fdm.data[:, x_mid, :]  # (ny, nz)
    data_x = data_x.T  # Transpose to (nz, ny) for proper orientation
    # If ny=1, replicate the data to make it visible
    if fdm.header.ny == 1:
        data_x = np.tile(data_x, (1, 10))  # Replicate 10 times along Y (now second dimension)
        y_extent_x = [y[0] - 5*fdm.header.dy, y[-1] + 5*fdm.header.dy]
    else:
        y_extent_x = [y[0], y[-1]]
    data_x = np.flipud(data_x)  # Flip vertically so Z=0 data is at top
    
    # Get smart colormap and scaling for X-slice
    colormap_x, vmin_x, vmax_x = get_smart_colormap_and_scaling(data_x, None, vmin, vmax)
    print(f"X-slice: colormap={colormap_x}, {scaling_type} scaling=[{vmin_x:.3f}, {vmax_x:.3f}]")
    
    im3 = axes[1, 0].imshow(data_x, aspect='auto', origin='lower',
                           extent=[y_extent_x[0], y_extent_x[1], z[-1], z[0]], 
                           cmap=colormap_x, vmin=vmin_x, vmax=vmax_x)
    axes[1, 0].set_xlabel('Y')
    axes[1, 0].set_ylabel('Z')
    axes[1, 0].set_title(f'X-slice at X={x[x_mid]:.2f}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add horizon overlay for X-slice
    if horizon_file is not None:
        # Get FDM Z coordinates for grid range
        fdm_z = fdm.get_coordinates(logical=False)[2]
        z_min, z_max = fdm_z[0], fdm_z[-1]
        
        hx, hy, hz, levels = load_horizon(horizon_file, dz_grid=dz_horizon, z_min=z_min, z_max=z_max)
        if hx is not None and hy is not None and hz is not None:
            # Check if horizon intersects this X level
            fdm_x_phys = fdm.get_coordinates(logical=False)[0]
            x_tolerance = abs(fdm_x_phys[1] - fdm_x_phys[0]) if len(fdm_x_phys) > 1 else 12.5
            x_slice_phys = fdm_x_phys[x_mid]  # Always use physical for comparison with horizon
            mask = np.abs(hx - x_slice_phys) <= x_tolerance
            if np.any(mask):
                if use_logical:
                    hy_plot = (hy[mask] - fdm.header.y0) / fdm.header.dy
                else:
                    hy_plot = hy[mask]
                hz_plot = hz[mask]
                
                # Plot horizons - use level information for proper separation
                if dz_horizon is not None:
                    # Get levels for the masked points
                    levels_plot = levels[mask]
                    
                    # Plot each level separately to avoid connecting different levels
                    unique_levels = np.unique(levels_plot)
                    for level in unique_levels:
                        level_mask = levels_plot == level
                        level_y = hy_plot[level_mask]
                        level_z = hz_plot[level_mask]
                        if len(level_y) > 0:
                            # Sort points within this level by Y coordinate
                            sort_idx = np.argsort(level_y)
                            axes[1, 0].plot(level_y[sort_idx], level_z[sort_idx], 'cyan', linewidth=1, zorder=10)
                    
                    # Add vertical grid lines for uniform spacing (now handled by mesh system)
                    if dx_mesh is None and dz_horizon is not None:
                        # Default behavior: use 2*dz_horizon for vertical lines when dz_horizon is active
                        effective_dx_mesh = 2 * dz_horizon
                        y_extent = axes[1, 0].get_xlim()  # For X-slice, xlim is Y coordinate
                        
                        if use_logical:
                            dy_grid_logical = effective_dx_mesh / fdm.header.dy
                            y_start = int(y_extent[0] / dy_grid_logical) * dy_grid_logical
                            y_positions = np.arange(y_start, y_extent[1] + dy_grid_logical/2, dy_grid_logical)
                        else:
                            y_start = int(y_extent[0] / effective_dx_mesh) * effective_dx_mesh
                            y_positions = np.arange(y_start, y_extent[1] + effective_dx_mesh/2, effective_dx_mesh)
                        
                        for y_pos in y_positions:
                            if y_extent[0] <= y_pos <= y_extent[1]:
                                axes[1, 0].axvline(x=y_pos, color='cyan', linewidth=0.5, alpha=0.7, zorder=5)
                else:
                    # Sort by Y coordinate for single horizon
                    sort_idx = np.argsort(hy_plot)
                    axes[1, 0].plot(hy_plot[sort_idx], hz_plot[sort_idx], 'cyan', linewidth=1, zorder=10)
    
    # Histogram
    axes[1, 1].hist(fdm.data.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Data Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add mesh lines to all slice plots (independent of horizon)
    draw_mesh_lines(axes[0, 1], fdm, 'y', use_logical, dz_mesh, dx_mesh, dz_horizon)  # Y-slice
    draw_mesh_lines(axes[1, 0], fdm, 'x', use_logical, dz_mesh, dx_mesh, dz_horizon)  # X-slice
    # Note: Z-slice (axes[0, 0]) doesn't show depth, so no mesh lines needed
    
    # Add colorbars
    fig.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    fig.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    fig.colorbar(im3, ax=axes[1, 0], shrink=0.8)
    
    plt.tight_layout()
    
    # Add interactive controls if requested
    if interactive:
        # Use the average scaling for global control
        avg_vmin = (vmin_z + vmin_y + vmin_x) / 3
        avg_vmax = (vmax_z + vmax_y + vmax_x) / 3
        gain_handler = InteractiveGainHandler([im1, im2, im3], avg_vmin, avg_vmax)
        
        # Ensure the figure window can receive focus
        if hasattr(fig.canvas, 'manager') and hasattr(fig.canvas.manager, 'window'):
            try:
                # Try to activate and focus the window
                fig.canvas.manager.window.activateWindow()
                fig.canvas.manager.window.raise_()
                fig.canvas.setFocusPolicy(2)  # Qt.StrongFocus
                fig.canvas.setFocus()
                print("Figure window focused successfully")
            except Exception as e:
                print(f"Could not set window focus: {e}")
        
        # Connect keyboard events for interactive gain controls
        def unified_key_handler(event):
            gain_handler.on_key_press(event)
        
        cid = fig.canvas.mpl_connect('key_press_event', unified_key_handler)
        
        print(f"\nInteractive controls connected (connection id: {cid})")
        print("  g     - Gain +1.5dB")
        print("  gg    - Gain +3.0dB (press g twice quickly)")
        print("  d     - Dim -1.5dB") 
        print("  dd    - Dim -3.0dB (press d twice quickly)")
        print("  Focus the plot window and use keys to adjust gain")
        print(f"  Backend: {matplotlib.get_backend()}")


def main():
    parser = argparse.ArgumentParser(description='Plot FDM seismic data files')
    parser.add_argument('filename', nargs='?', help='FDM file to plot')
    parser.add_argument('--slice', choices=['x', 'y', 'z'], default='y',
                       help='Axis for slice plot (default: y)')
    parser.add_argument('--index', type=int, default=None,
                       help='Index of slice to plot (default: middle slice)')
    parser.add_argument('--summary', action='store_true',
                       help='Show summary plot with multiple slices')
    parser.add_argument('--cmap', default=None,
                       help='Colormap for plotting (default: auto-select TgsBanded for velocity, gray_r for seismic)')
    parser.add_argument('--list-cmaps', action='store_true',
                       help='List all available colormaps and exit')
    parser.add_argument('--preview-cmaps', nargs='*',
                       help='Preview colormaps visually (optionally specify names)')
    parser.add_argument('--info', action='store_true',
                       help='Show header information only')
    parser.add_argument('--header-only', action='store_true',
                       help='Show header info without reading data (like fdminfo binary)')
    parser.add_argument('--save', help='Save plot to file instead of displaying')
    parser.add_argument('--logical', action='store_true', default=True,
                       help='Use logical coordinates (grid indices) - DEFAULT')
    parser.add_argument('--physical', action='store_true',
                       help='Use physical coordinates instead of logical')
    parser.add_argument('--test-interactive', action='store_true',
                       help='Test interactive gain controls without GUI (for debugging)')
    parser.add_argument('--no-display', action='store_true',
                       help='Save to file instead of displaying interactively')
    parser.add_argument('--vmin', type=float, default=None,
                       help='Minimum value for colormap scaling (overrides automatic scaling)')
    parser.add_argument('--vmax', type=float, default=None,
                       help='Maximum value for colormap scaling (overrides automatic scaling)')
    parser.add_argument('--horizon', type=str, default=None,
                       help='Overlay horizon file (X Y Z format) on plots')
    parser.add_argument('--dz-horizon', type=float, default=None,
                       help='Shift and replicate horizon at depth intervals (grid pattern)')
    parser.add_argument('--dz-mesh', type=float, default=None,
                       help='Add horizontal grid lines at specified depth intervals')
    parser.add_argument('--dx-mesh', type=float, default=None,
                       help='Add vertical grid lines at specified intervals (defaults to 2*dz_horizon or 2*dz_mesh)')
    
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nExample usage:")
        print("  python plotfdm.py examples/vel_topo.fdm --summary")
        print("  python plotfdm.py examples/vel_topo.fdm --slice y --index 5")
        print("  python plotfdm.py examples/vel_topo.fdm --info")
        print("  python plotfdm.py examples/vel_topo.fdm  # plots middle y-slice with logical coords (default)")
        print("  python plotfdm.py examples/vel_topo.fdm --physical  # use physical coordinates")
        print("  python plotfdm.py examples/vel_topo.fdm --no-display  # save to file instead of displaying")
        print("  python plotfdm.py examples/vel_topo.fdm --save plot.png  # save to specific file")
        print("  python plotfdm.py examples/vel_topo.fdm --cmap viridis  # override automatic colormap selection")
        print("  python plotfdm.py --list-cmaps  # list all available colormaps")
        print("  python plotfdm.py --preview-cmaps  # preview default colormaps")
        print("  python plotfdm.py --preview-cmaps seismic BigTiger  # preview specific colormaps")
        print("  python plotfdm.py examples/vel_topo.fdm --vmin -100 --vmax 100  # manual colormap scaling")
        print("  python plotfdm.py examples/vel_topo.fdm --horizon examples/horizon.txt  # overlay horizon data")
        print("  python plotfdm.py examples/vel_topo.fdm --horizon examples/horizon.txt --dz-horizon 50  # horizon grid every 50 units")
        print("  python plotfdm.py examples/vel_topo.fdm --dz-mesh 25 --dx-mesh 100  # add mesh lines: horizontal every 25, vertical every 100")
        print("  python plotfdm.py examples/vel_topo.fdm --dz-mesh 25  # horizontal mesh every 25, vertical auto (2*dz_mesh = 50)")
        print()
        print("Automatic colormap selection:")
        print("  - Velocity data (all positive): TgsBanded")
        print("  - Seismic data (pos/neg): gray_r")
        print("  - Automatic scaling: +/- RMS -4dB")
        print()
        print("Interactive controls (when displaying):")
        print("  g     - Gain +1.5dB")
        print("  gg    - Gain +3.0dB (press g twice quickly)")
        print("  d     - Dim -1.5dB")
        print("  dd    - Dim -3.0dB (press d twice quickly)")
        return
    
    args = parser.parse_args()
    
    # Handle colormap listing/preview options (no FDM file needed)
    if args.list_cmaps:
        custom_colormaps.list_all_colormaps()
        return
    
    if args.preview_cmaps is not None:
        if args.preview_cmaps:
            custom_colormaps.preview_colormaps(args.preview_cmaps)
        else:
            custom_colormaps.preview_colormaps()
        return
    
    # Handle test interactive mode
    if args.test_interactive:
        print("Testing interactive gain controls...")
        # Create a simple mock event class
        class MockEvent:
            def __init__(self, key):
                self.key = key
        
        # Test data
        import numpy as np
        test_data = np.random.randn(100, 100)
        vmin, vmax = -2.0, 2.0
        
        # Create a mock image object
        class MockImage:
            def __init__(self):
                self.vmin = vmin
                self.vmax = vmax
            def set_clim(self, vmin, vmax):
                self.vmin = vmin
                self.vmax = vmax
                print(f"  Mock image clim set to: [{vmin:.3f}, {vmax:.3f}]")
        
        mock_im = MockImage()
        handler = InteractiveGainHandler(mock_im, vmin, vmax)
        
        print("Testing gain controls:")
        print("  Initial scaling: [%.3f, %.3f]" % (vmin, vmax))
        
        # Test single gain
        print("\n1. Testing single 'g' (gain +1.5dB):")
        handler.on_key_press(MockEvent('g'))
        
        # Test single dim
        print("\n2. Testing single 'd' (dim -1.5dB):")
        handler.on_key_press(MockEvent('d'))
        
        # Test double gain (simulate quick succession)
        print("\n3. Testing double 'g' (gain +3.0dB):")
        handler.on_key_press(MockEvent('g'))
        import time; time.sleep(0.1)  # Small delay
        handler.on_key_press(MockEvent('g'))  # Should be detected as double
        
        print("\nInteractive gain control test completed successfully!")
        return
    
    try:
        # Load FDM file
        print(f"Loading FDM file: {args.filename}")
        fdm = FDM(args.filename)
        
        if args.header_only:
            fdm.read_header_only()
            fdm.print_header()
            return
        else:
            fdm.read()
            # Always show header info
            fdm.print_header()
            print(f"Data shape: {fdm.data.shape}")
            print(f"Data range: {fdm.data.min():.6f} to {fdm.data.max():.6f}")
        
        if args.info:
            return
        
        # Set up matplotlib backend based on mode
        if args.save or args.no_display:
            # Use non-interactive backend for file output
            matplotlib.use('Agg')
            print("Non-interactive backend (Agg) activated for file output")
        else:
            # Set interactive backend before creating any plots
            backend_set = False
            try:
                plt.close('all')  # Close any existing figures first
                matplotlib.use('TkAgg')  # Try interactive backend
                print("Interactive backend (TkAgg) activated")
                backend_set = True
            except Exception as e:
                try:
                    matplotlib.use('Qt5Agg')  # Try Qt backend
                    print("Interactive backend (Qt5Agg) activated")
                    backend_set = True
                except Exception as e2:
                    print(f"Warning: Could not set interactive backend: {e}, {e2}")
                    print("Interactive controls may not work")
                    # Fall back to Agg and disable interactive
                    matplotlib.use('Agg')
                    backend_set = False
            
            if not backend_set:
                print("Interactive mode disabled - falling back to file output mode")
                args.no_display = True  # Force file output
        
        # Determine default index if not specified
        if args.index is None:
            if args.slice == 'x':
                args.index = fdm.header.nx // 2
            elif args.slice == 'y':
                args.index = fdm.header.ny // 2
            else:  # z
                args.index = fdm.header.nz // 2
        
        # Create plots
        if args.summary:
            # Default to logical coordinates, use physical if --physical flag is set
            use_logical = not args.physical
            is_interactive = not args.save and not args.no_display
            plot_summary(fdm, use_logical=use_logical, interactive=is_interactive, 
                        vmin=args.vmin, vmax=args.vmax, horizon_file=args.horizon, dz_horizon=args.dz_horizon,
                        dz_mesh=args.dz_mesh, dx_mesh=args.dx_mesh, cmap=args.cmap)
        else:
            # Default to logical coordinates, use physical if --physical flag is set
            use_logical = not args.physical
            is_interactive = not args.save and not args.no_display
            plot_slice(fdm, axis=args.slice, index=args.index, cmap=args.cmap, 
                      use_logical=use_logical, interactive=is_interactive,
                      vmin=args.vmin, vmax=args.vmax, horizon_file=args.horizon, dz_horizon=args.dz_horizon,
                      dz_mesh=args.dz_mesh, dx_mesh=args.dx_mesh)
        
        # Save or show plot
        if args.save:
            plt.savefig(args.save, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {args.save}")
        elif args.no_display:
            # Save to file instead of displaying
            default_filename = f"{args.filename.replace('.fdm', '')}_plot.png"
            plt.savefig(default_filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {default_filename}")
        else:
            # Default: display interactively and keep figures open
            try:
                plt.show(block=False)  # Non-blocking show to keep program running
                print("Plot displayed. Close the plot window or press Ctrl+C to exit.")
                # Keep the program running so figures stay open
                try:
                    plt.show(block=True)  # Block to keep figures open
                except KeyboardInterrupt:
                    print("\nExiting...")
            except Exception as e:
                print(f"Could not display plot interactively: {e}")
                print("Falling back to saving plot to file...")
                default_filename = f"{args.filename.replace('.fdm', '')}_plot.png"
                plt.savefig(default_filename, dpi=300, bbox_inches='tight')
                print(f"Plot saved to: {default_filename}")
            
    except FileNotFoundError:
        print(f"Error: File '{args.filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()