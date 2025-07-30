#!/usr/bin/env python

"""
FDM File Plotting Tool

Usage:
    python plotfdm.py <fdm_file> [options]

This script reads FDM files and provides plotting capabilities for        print("Example usage:")
        print("  python plotfdm.py data.fdm --summary")
        print("  python plotfdm.py data.fdm --slice y --index 5")
        print("  python plotfdm.py data.fdm --info")
        print("  python plotfdm.py data.fdm  # plots middle y-slice with logical coords (default)")
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


def plot_slice(fdm, axis='z', index=0, cmap=None, title=None, use_logical=True, interactive=False, vmin=None, vmax=None):
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
        title = f'{axis_name}-slice at {axis_name}={slice_coord:.2f}'
    plt.title(title)
    plt.grid(True, alpha=0.3)
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


def plot_summary(fdm, use_logical=True, interactive=False, vmin=None, vmax=None):
    """Plot summary showing slices along all three axes
    Note: Z-axis always uses physical coordinates for depth visualization
    
    Args:
        fdm: FDM object with loaded data
        use_logical: Use logical coordinates for X,Y axes (default: True)
        interactive: Enable interactive gain/dim controls (default: False)
        vmin: Minimum value for colormap scaling (None for automatic)
        vmax: Maximum value for colormap scaling (None for automatic)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Middle slices
    z_mid = fdm.header.nz // 2
    y_mid = fdm.header.ny // 2
    x_mid = fdm.header.nx // 2
    
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
    
    # Histogram
    axes[1, 1].hist(fdm.data.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Data Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
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
    
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nExample usage:")
        print("  python plotfdm.py examples/shot_topo.fdm --summary")
        print("  python plotfdm.py examples/vel_topo.fdm --slice y --index 5")
        print("  python plotfdm.py examples/shot_topo.fdm --info")
        print("  python plotfdm.py examples/vel_topo.fdm  # plots middle y-slice with logical coords (default)")
        print("  python plotfdm.py examples/shot_topo.fdm --physical  # use physical coordinates")
        print("  python plotfdm.py examples/vel_topo.fdm --no-display  # save to file instead of displaying")
        print("  python plotfdm.py examples/shot_topo.fdm --save plot.png  # save to specific file")
        print("  python plotfdm.py examples/shot_topo.fdm --cmap viridis  # override automatic colormap selection")
        print("  python plotfdm.py --list-cmaps  # list all available colormaps")
        print("  python plotfdm.py --preview-cmaps  # preview default colormaps")
        print("  python plotfdm.py --preview-cmaps seismic BigTiger  # preview specific colormaps")
        print("  python plotfdm.py examples/vel_topo.fdm --vmin -100 --vmax 100  # manual colormap scaling")
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
                        vmin=args.vmin, vmax=args.vmax)
        else:
            # Default to logical coordinates, use physical if --physical flag is set
            use_logical = not args.physical
            is_interactive = not args.save and not args.no_display
            plot_slice(fdm, axis=args.slice, index=args.index, cmap=args.cmap, 
                      use_logical=use_logical, interactive=is_interactive,
                      vmin=args.vmin, vmax=args.vmax)
        
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