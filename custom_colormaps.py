#!/usr/bin/env python
"""
Custom colormaps for FDM plotting
Extracted from SeisVu project
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm


def register_custom_colormaps():
    """Register custom colormaps with matplotlib"""
    
    # BigTiger colormap
    cm_bigtiger = colors.LinearSegmentedColormap(name='BigTiger', segmentdata={
        'red': ((0.0, 0.0, 0.0),
            (0.584, 1.0, 1.0),
            (0.588, 1.0, 1.0),
            (1.0, 1.0, 1.0)),
        'green': ((0.0, 0.0, 0.0),
            (0.584, 1.0, 1.0),
            (0.588, 1.0, 1.0),
            (1.0, 0.0, 0.0)),
        'blue': ((0.0, 0.0, 0.0),
            (0.584, 1.0, 1.0),
            (0.588, 0.0, 0.0),
            (1.0, 0.0, 0.0))
    })
    cm._colormaps.register(name='BigTiger', cmap=cm_bigtiger)
    cm._colormaps.register(name='BigTiger_r', cmap=cm.get_cmap(cm_bigtiger).reversed())
    
    # TGS banded rainbow colormap
    cm_tgs_banded = [
        [100, 100, 100],
        [94, 92, 96],
        [88, 84, 92],
        [82, 76, 88],
        [77, 69, 85],
        [71, 61, 81],
        [65, 53, 77],
        [60, 46, 74],
        [54, 38, 70],
        [48, 30, 66],
        [43, 23, 63],
        [37, 15, 59],
        [26, 0, 52],
        [27, 0, 56],
        [29, 0, 62],
        [31, 0, 71],
        [33, 0, 79],
        [35, 0, 87],
        [36, 0, 94],
        [35, 0, 98],
        [34, 0, 100],
        [32, 0, 98],
        [28, 0, 94],
        [25, 0, 88],
        [21, 0, 79],
        [17, 0, 71],
        [14, 0, 63],
        [11, 0, 56],
        [9, 0, 52],
        [8, 0, 50],
        [7, 0, 52],
        [7, 0, 56],
        [6, 0, 62],
        [6, 0, 71],
        [5, 0, 79],
        [3, 0, 87],
        [2, 0, 94],
        [0, 0, 98],
        [0, 2, 100],
        [0, 4, 98],
        [0, 6, 94],
        [0, 7, 88],
        [0, 8, 79],
        [0, 8, 71],
        [0, 9, 63],
        [0, 9, 56],
        [0, 9, 52],
        [0, 10, 50],
        [0, 11, 52],
        [0, 13, 56],
        [0, 16, 62],
        [0, 20, 71],
        [0, 24, 79],
        [0, 28, 87],
        [0, 32, 94],
        [0, 35, 98],
        [0, 38, 100],
        [0, 39, 98],
        [0, 40, 94],
        [0, 39, 88],
        [0, 36, 79],
        [0, 34, 71],
        [0, 31, 63],
        [0, 29, 56],
        [0, 28, 52],
        [0, 28, 50],
        [0, 30, 52],
        [0, 34, 56],
        [0, 39, 62],
        [0, 45, 71],
        [0, 52, 79],
        [0, 59, 87],
        [0, 66, 94],
        [0, 71, 98],
        [0, 74, 100],
        [0, 75, 98],
        [0, 73, 94],
        [0, 70, 88],
        [0, 65, 79],
        [0, 59, 71],
        [0, 54, 63],
        [0, 49, 56],
        [0, 46, 52],
        [0, 46, 50],
        [0, 48, 52],
        [0, 54, 56],
        [0, 61, 62],
        [0, 71, 71],
        [0, 79, 78],
        [0, 87, 84],
        [0, 94, 89],
        [0, 98, 91],
        [0, 100, 90],
        [0, 98, 87],
        [0, 94, 81],
        [0, 88, 74],
        [0, 79, 65],
        [0, 71, 57],
        [0, 63, 49],
        [0, 56, 42],
        [0, 52, 38],
        [0, 50, 36],
        [0, 52, 36],
        [0, 56, 38],
        [0, 62, 41],
        [0, 71, 45],
        [0, 79, 49],
        [0, 87, 52],
        [0, 94, 55],
        [0, 98, 55],
        [0, 100, 54],
        [0, 98, 51],
        [0, 94, 47],
        [0, 88, 42],
        [0, 79, 36],
        [0, 71, 31],
        [0, 63, 26],
        [0, 56, 22],
        [0, 52, 20],
        [0, 50, 18],
        [0, 52, 18],
        [0, 56, 18],
        [0, 62, 19],
        [0, 71, 20],
        [0, 79, 21],
        [0, 87, 21],
        [0, 94, 21],
        [0, 98, 20],
        [0, 100, 18],
        [0, 98, 16],
        [0, 94, 13],
        [0, 88, 11],
        [0, 79, 8],
        [0, 71, 6],
        [0, 63, 4],
        [0, 56, 2],
        [0, 52, 1],
        [0, 50, 0],
        [1, 52, 0],
        [2, 56, 0],
        [4, 62, 0],
        [6, 71, 0],
        [8, 79, 0],
        [10, 87, 0],
        [13, 94, 0],
        [16, 98, 0],
        [18, 100, 0],
        [20, 98, 0],
        [21, 94, 0],
        [21, 88, 0],
        [21, 79, 0],
        [20, 71, 0],
        [19, 63, 0],
        [18, 56, 0],
        [18, 52, 0],
        [18, 50, 0],
        [20, 52, 0],
        [22, 56, 0],
        [26, 62, 0],
        [31, 71, 0],
        [36, 79, 0],
        [42, 87, 0],
        [47, 94, 0],
        [51, 98, 0],
        [54, 100, 0],
        [55, 98, 0],
        [55, 94, 0],
        [53, 88, 0],
        [49, 79, 0],
        [45, 71, 0],
        [41, 63, 0],
        [38, 56, 0],
        [36, 52, 0],
        [36, 50, 0],
        [38, 52, 0],
        [42, 56, 0],
        [49, 62, 0],
        [57, 71, 0],
        [65, 79, 0],
        [73, 87, 0],
        [81, 94, 0],
        [87, 98, 0],
        [90, 100, 0],
        [91, 98, 0],
        [89, 94, 0],
        [84, 88, 0],
        [78, 79, 0],
        [71, 71, 0],
        [63, 61, 0],
        [56, 54, 0],
        [52, 48, 0],
        [50, 46, 0],
        [52, 46, 0],
        [56, 49, 0],
        [62, 54, 0],
        [71, 59, 0],
        [79, 65, 0],
        [87, 70, 0],
        [94, 73, 0],
        [98, 75, 0],
        [100, 74, 0],
        [98, 71, 0],
        [94, 66, 0],
        [88, 60, 0],
        [79, 52, 0],
        [71, 45, 0],
        [63, 39, 0],
        [56, 34, 0],
        [52, 30, 0],
        [50, 28, 0],
        [52, 28, 0],
        [56, 29, 0],
        [62, 31, 0],
        [71, 34, 0],
        [79, 36, 0],
        [87, 38, 0],
        [94, 40, 0],
        [98, 39, 0],
        [100, 38, 0],
        [98, 35, 0],
        [94, 32, 0],
        [88, 28, 0],
        [79, 24, 0],
        [71, 20, 0],
        [63, 16, 0],
        [56, 13, 0],
        [52, 11, 0],
        [50, 10, 0],
        [52, 9, 0],
        [56, 9, 0],
        [62, 9, 0],
        [71, 8, 0],
        [79, 8, 0],
        [87, 7, 0],
        [94, 6, 0],
        [98, 4, 0],
        [100, 2, 0],
    ]
    
    # Register tgs_banded colormap (ListedColormap)
    cm_tgs_banded_arr = np.array(cm_tgs_banded, dtype=float) / 100.0
    try:
        cmap_tgs_banded = colors.ListedColormap(cm_tgs_banded_arr, name='TgsBanded')
        cm._colormaps.register(name='TgsBanded', cmap=cmap_tgs_banded)
        cmap_tgs_banded_r = colors.ListedColormap(cm_tgs_banded_arr[::-1], name='TgsBanded_r')
        cm._colormaps.register(name='TgsBanded_r', cmap=cmap_tgs_banded_r)
    except Exception:
        pass

    print("Custom colormaps registered: BigTiger, BigTiger_r, TgsBanded, TgsBanded_r")


def list_all_colormaps():
    """List all available colormaps in matplotlib"""
    print("Available colormaps:")
    print("=" * 50)
    
    # Get all colormap names
    all_cmaps = sorted(plt.colormaps())
    
    # Group by category
    sequential = []
    diverging = []
    qualitative = []
    cyclic = []
    custom = []
    
    for cmap_name in all_cmaps:
        if cmap_name in ['BigTiger', 'BigTiger_r', 'TgsBanded', 'TgsBanded_r']:
            custom.append(cmap_name)
        elif cmap_name.endswith('_r'):
            continue  # Skip reversed versions for now
        elif cmap_name in ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'hot', 'copper', 'Blues', 'BuGn', 'BuPu', 'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']:
            sequential.append(cmap_name)
        elif cmap_name in ['RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'RdBu', 'RdGy', 'PiYG', 'PRGn', 'BrBG', 'PuOr']:
            diverging.append(cmap_name)
        elif cmap_name in ['twilight', 'twilight_shifted', 'hsv']:
            cyclic.append(cmap_name)
        elif cmap_name in ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']:
            qualitative.append(cmap_name)
        else:
            sequential.append(cmap_name)  # Default to sequential
    
    if custom:
        print(f"Custom ({len(custom)}):")
        for cmap in custom:
            print(f"  {cmap}")
        print()
    
    if sequential:
        print(f"Sequential ({len(sequential)}):")
        for cmap in sequential:
            print(f"  {cmap}")
        print()
    
    if diverging:
        print(f"Diverging ({len(diverging)}):")
        for cmap in diverging:
            print(f"  {cmap}")
        print()
    
    if qualitative:
        print(f"Qualitative ({len(qualitative)}):")
        for cmap in qualitative:
            print(f"  {cmap}")
        print()
    
    if cyclic:
        print(f"Cyclic ({len(cyclic)}):")
        for cmap in cyclic:
            print(f"  {cmap}")
        print()
    
    print(f"Total colormaps available: {len(all_cmaps)}")
    print("\nNote: Most colormaps also have a reversed version (ending with '_r')")


def preview_colormaps(names=None):
    """Preview colormaps visually"""
    if names is None:
        names = ['seismic', 'viridis', 'BigTiger', 'TgsBanded']
    
    # Create sample data
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    
    fig, axes = plt.subplots(len(names), 1, figsize=(8, len(names) * 0.8))
    if len(names) == 1:
        axes = [axes]
    
    for ax, name in zip(axes, names):
        try:
            ax.imshow(gradient, aspect='auto', cmap=name)
            ax.set_xlim(0, 256)
            ax.set_yticks([])
            ax.set_ylabel(name, rotation=0, ha='right')
        except ValueError:
            ax.text(0.5, 0.5, f"Colormap '{name}' not found", 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()


# Auto-register custom colormaps when module is imported
register_custom_colormaps()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--list':
            list_all_colormaps()
        elif sys.argv[1] == '--preview':
            if len(sys.argv) > 2:
                preview_colormaps(sys.argv[2:])
            else:
                preview_colormaps()
        else:
            print("Usage:")
            print("  python custom_colormaps.py --list      # List all available colormaps")
            print("  python custom_colormaps.py --preview   # Preview default colormaps")
            print("  python custom_colormaps.py --preview seismic viridis BigTiger  # Preview specific colormaps")
    else:
        print("Custom colormaps module loaded.")
        print("Available functions:")
        print("  list_all_colormaps()  - List all available colormaps")
        print("  preview_colormaps()   - Preview colormaps visually")
        print("  register_custom_colormaps() - Register custom colormaps")
        print()
        print("Run with --help for command line options")
