#!/usr/bin/env python

import struct
import numpy as np
import os
from typing import Optional, Tuple, Union


class FDMHeader:
    """FDM file header structure"""
    def __init__(self):
        # Grid origin coordinates
        self.x0: float = 0.0
        self.y0: float = 0.0
        self.z0: float = 0.0
        
        # Grid dimensions
        self.nx: int = 0
        self.ny: int = 0
        self.nz: int = 0
        
        # Grid spacing
        self.dx: float = 0.0
        self.dy: float = 0.0
        self.dz: float = 0.0
        
        # First line coordinates
        self.fx: int = 0
        self.fy: int = 0
        self.fz: int = 0
        
        # Line increments
        self.xi: int = 0
        self.yi: int = 0
        self.zi: int = 1
        
        # Last line coordinates (computed)
        self.lx: int = 0
        self.ly: int = 0
        self.lz: int = 0
        
        # Data value range
        self.vmin: float = 0.0
        self.vmax: float = 0.0


class FDM:
    """Python class to handle FDM file format"""
    
    HEADER_SIZE = 512  # 512 bytes = 128 floats
    HEADER_FLOAT_COUNT = 128  # 512 bytes / 4 bytes per float
    MAGIC_BYTES = b'SFDM'
    
    def __init__(self, filename: Optional[str] = None):
        self.filename = filename
        self.header = FDMHeader()
        self.data: Optional[np.ndarray] = None
        self._swap_endian = False
    
    def _detect_endian_swap(self, header_array: np.ndarray) -> bool:
        """Detect if endian swapping is needed based on dx value"""
        dx = header_array[2]
        # If dx is unreasonable, assume we need to swap
        return dx < 1e-10 or dx > 100000
    
    def _swap_float_endian(self, data: np.ndarray) -> np.ndarray:
        """Swap endianness of float array"""
        return data.byteswap().newbyteorder()
    
    def _array_to_header(self, array: np.ndarray) -> None:
        """Convert float array to header structure (equivalent to array2Header)"""
        self.header.x0 = float(array[0])
        self.header.y0 = float(array[3])
        self.header.z0 = float(array[6])
        
        self.header.nx = int(array[1])
        self.header.ny = int(array[4])
        self.header.nz = int(array[7])
        
        self.header.dx = float(array[2])
        self.header.dy = float(array[5])
        self.header.dz = float(array[8])
        
        self.header.fx = int(array[14])
        self.header.fy = int(array[15])
        self.header.fz = 1  # Based on fdminfo output, should be 1, not 0
        
        self.header.xi = int(array[16])
        self.header.yi = int(array[17])
        self.header.zi = 1
        
        # Compute last line coordinates
        self.header.lx = self.header.fx + (self.header.nx - 1) * self.header.xi
        self.header.ly = self.header.fy + (self.header.ny - 1) * self.header.yi
        self.header.lz = self.header.fz + (self.header.nz - 1) * self.header.zi
        
        self.header.vmin = float(array[12])
        self.header.vmax = float(array[13])
    
    def _header_to_array(self) -> np.ndarray:
        """Convert header structure to float array (equivalent to header2Array)"""
        array = np.zeros(self.HEADER_FLOAT_COUNT, dtype=np.float32)
        
        # Grid parameters
        array[0] = self.header.x0
        array[1] = float(self.header.nx)
        array[2] = self.header.dx
        
        array[3] = self.header.y0
        array[4] = float(self.header.ny)
        array[5] = self.header.dy
        
        array[6] = self.header.z0
        array[7] = float(self.header.nz)
        array[8] = self.header.dz
        
        # Unused format fields (9, 10, 11)
        
        # Value range
        array[12] = self.header.vmin
        array[13] = self.header.vmax
        
        # Line coordinates and increments
        array[14] = float(self.header.fx)
        array[15] = float(self.header.fy)
        array[16] = float(self.header.xi)
        array[17] = float(self.header.yi)
        
        # Magic bytes at position 18 (as float representation)
        magic_int = struct.unpack('<I', self.MAGIC_BYTES)[0]
        array[18] = struct.unpack('<f', struct.pack('<I', magic_int))[0]
        
        # Version
        array[19] = struct.unpack('<f', struct.pack('<I', 2))[0]
        
        # Distance unit
        array[20] = struct.unpack('<f', struct.pack('<I', 1))[0]
        
        # Angle unit (radians)
        array[21] = struct.unpack('<f', struct.pack('<I', 2))[0]
        
        # North angle
        array[22] = 0.0
        
        # Rotation angle
        array[23] = 0.0
        
        return array
    
    def read_header(self, file_handle) -> None:
        """Read and parse the FDM header"""
        file_handle.seek(0)
        header_bytes = file_handle.read(self.HEADER_SIZE)
        
        if len(header_bytes) != self.HEADER_SIZE:
            raise ValueError(f"Expected {self.HEADER_SIZE} bytes for header, got {len(header_bytes)}")
        
        # Convert to float array
        header_array = np.frombuffer(header_bytes, dtype=np.float32)
        
        # Check if endian swapping is needed
        if self._detect_endian_swap(header_array):
            self._swap_endian = True
            header_array = self._swap_float_endian(header_array)
        
        # Parse header
        self._array_to_header(header_array)
    
    def read_data(self, file_handle) -> None:
        """Read the FDM data portion"""
        if self.header.nx <= 0 or self.header.ny <= 0 or self.header.nz <= 0:
            raise ValueError("Invalid grid dimensions in header")
        
        total_size = self.header.nx * self.header.ny * self.header.nz
        expected_bytes = total_size * 4  # 4 bytes per float
        
        file_handle.seek(self.HEADER_SIZE)
        data_bytes = file_handle.read(expected_bytes)
        
        if len(data_bytes) != expected_bytes:
            raise ValueError(f"Expected {expected_bytes} bytes for data, got {len(data_bytes)}")
        
        # Convert to float array
        self.data = np.frombuffer(data_bytes, dtype=np.float32)
        
        if self._swap_endian:
            self.data = self._swap_float_endian(self.data)
        
        # Reshape to 3D array - memory layout is nz innermost, nx, then ny outermost
        # So the array should be shaped as (ny, nx, nz)
        self.data = self.data.reshape((self.header.ny, self.header.nx, self.header.nz))
        
        # Update header with actual min/max values
        self.header.vmin = float(np.min(self.data))
        self.header.vmax = float(np.max(self.data))
    
    def read_header_only(self, filename: Optional[str] = None) -> None:
        """Read only FDM header without data (like fdminfo binary)"""
        if filename:
            self.filename = filename
        
        if not self.filename:
            raise ValueError("No filename specified")
        
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"File not found: {self.filename}")
        
        with open(self.filename, 'rb') as f:
            self.read_header(f)
            # Don't read data, keep vmin/vmax from header
    
    def read(self, filename: Optional[str] = None) -> None:
        """Read complete FDM file"""
        if filename:
            self.filename = filename
        
        if not self.filename:
            raise ValueError("No filename specified")
        
        if not os.path.exists(self.filename):
            raise FileNotFoundError(f"File not found: {self.filename}")
        
        with open(self.filename, 'rb') as f:
            self.read_header(f)
            self.read_data(f)
    
    def write_header(self, file_handle) -> None:
        """Write FDM header to file"""
        header_array = self._header_to_array()
        header_bytes = header_array.tobytes()
        
        if len(header_bytes) != self.HEADER_SIZE:
            # Pad or truncate to exact header size
            if len(header_bytes) < self.HEADER_SIZE:
                header_bytes += b'\x00' * (self.HEADER_SIZE - len(header_bytes))
            else:
                header_bytes = header_bytes[:self.HEADER_SIZE]
        
        file_handle.seek(0)
        file_handle.write(header_bytes)
    
    def write_data(self, file_handle, data: Optional[np.ndarray] = None) -> None:
        """Write FDM data to file"""
        output_data = data if data is not None else self.data
        
        if output_data is None:
            raise ValueError("No data to write")
        
        # Ensure data is float32 and flattened
        if output_data.dtype != np.float32:
            output_data = output_data.astype(np.float32)
        
        file_handle.seek(self.HEADER_SIZE)
        file_handle.write(output_data.flatten().tobytes())
    
    def write(self, filename: Optional[str] = None, data: Optional[np.ndarray] = None) -> None:
        """Write complete FDM file"""
        if filename:
            self.filename = filename
        
        if not self.filename:
            raise ValueError("No filename specified")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.filename)), exist_ok=True)
        
        # Update header with data min/max if data is provided
        if data is not None:
            self.data = data
            self.header.vmin = float(np.min(data))
            self.header.vmax = float(np.max(data))
        
        with open(self.filename, 'wb') as f:
            self.write_header(f)
            self.write_data(f, data)
    
    def set_header(self, x0: float, y0: float, z0: float, 
                   nx: int, ny: int, nz: int,
                   dx: float, dy: float, dz: float,
                   fx: int = 0, fy: int = 0, fz: int = 1,
                   xi: int = 1, yi: int = 1, zi: int = 1) -> None:
        """Set header parameters"""
        self.header.x0, self.header.y0, self.header.z0 = x0, y0, z0
        self.header.nx, self.header.ny, self.header.nz = nx, ny, nz
        self.header.dx, self.header.dy, self.header.dz = dx, dy, dz
        self.header.fx, self.header.fy, self.header.fz = fx, fy, fz
        self.header.xi, self.header.yi, self.header.zi = xi, yi, zi
        
        # Compute last line coordinates - note: uses 1-based indexing for z
        self.header.lx = self.header.fx + (self.header.nx - 1) * self.header.xi
        self.header.ly = self.header.fy + (self.header.ny - 1) * self.header.yi
        self.header.lz = self.header.fz + (self.header.nz - 1) * self.header.zi
    
    def print_header(self) -> None:
        """Print header information matching fdminfo binary format"""
        print("FDM info:")
        print(f"          orig: (x0,y0,z0) = {self.header.x0:.6f},{self.header.y0:.6f},{self.header.z0:.6f}")
        print(f"          step: (dx,dy,dz) = {self.header.dx:.6f},{self.header.dy:.6f},{self.header.dz:.6f}")
        print(f"          size: (nx,ny,nz) = {self.header.nx},{self.header.ny},{self.header.nz}")
        print(f"      line inc: (xi,yi,zi) = {self.header.xi},{self.header.yi},{self.header.zi}")
        print(f"    first line:              {self.header.fx},{self.header.fy},{self.header.fz}")
        print(f"     last line:              {self.header.lx},{self.header.ly},{self.header.lz}")
        print(f"     v min/max: {self.header.vmin:.6f} / {self.header.vmax:.6f}")
    
    def get_coordinates(self, logical: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get coordinate arrays for the grid
        
        Args:
            logical: If True (default), returns logical coordinates (grid indices).
                    If False, returns physical coordinates.
        """
        if logical:
            return self.get_logical_coordinates()
        else:
            return self.get_physical_coordinates()
    
    def get_logical_coordinates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get logical/grid coordinate arrays (indices)"""
        xi = np.arange(self.header.nx)
        yi = np.arange(self.header.ny) 
        zi = np.arange(self.header.nz)
        return xi, yi, zi
    
    def get_physical_coordinates(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get physical coordinate arrays"""
        x = np.arange(self.header.nx) * self.header.dx + self.header.x0
        y = np.arange(self.header.ny) * self.header.dy + self.header.y0
        z = np.arange(self.header.nz) * self.header.dz + self.header.z0
        return x, y, z
    
    def logical_to_physical(self, xi: Union[int, np.ndarray], yi: Union[int, np.ndarray], zi: Union[int, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Convert logical coordinates to physical coordinates"""
        x = xi * self.header.dx + self.header.x0
        y = yi * self.header.dy + self.header.y0
        z = zi * self.header.dz + self.header.z0
        return x, y, z
    
    def physical_to_logical(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray], z: Union[float, np.ndarray]) -> Tuple[Union[int, np.ndarray], Union[int, np.ndarray], Union[int, np.ndarray]]:
        """Convert physical coordinates to logical coordinates (grid indices)"""
        xi = np.round((x - self.header.x0) / self.header.dx).astype(int)
        yi = np.round((y - self.header.y0) / self.header.dy).astype(int)
        zi = np.round((z - self.header.z0) / self.header.dz).astype(int)
        return xi, yi, zi
    
    def get_value_at_logical(self, xi: int, yi: int, zi: int) -> float:
        """Get data value at logical coordinates (grid indices)"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        if not (0 <= xi < self.header.nx and 0 <= yi < self.header.ny and 0 <= zi < self.header.nz):
            raise IndexError(f"Logical coordinates ({xi},{yi},{zi}) out of bounds")
        
        return float(self.data[zi, yi, xi])
    
    def get_value_at_physical(self, x: float, y: float, z: float) -> float:
        """Get data value at physical coordinates (with interpolation to nearest grid point)"""
        xi, yi, zi = self.physical_to_logical(x, y, z)
        return self.get_value_at_logical(xi, yi, zi)
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get data shape as (nz, ny, nx)"""
        return (self.header.nz, self.header.ny, self.header.nx)
    
    @property
    def size(self) -> int:
        """Get total number of data points"""
        return self.header.nx * self.header.ny * self.header.nz


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test with provided filename
        filename = sys.argv[1]
        print(f"Reading FDM file: {filename}")
        fdm = FDM(filename)
        fdm.read()
        fdm.print_header()
        print(f"Data shape: {fdm.data.shape}")
        print(f"Data range: {fdm.data.min()} to {fdm.data.max()}")
    else:
        # Create a simple example
        print("Creating example FDM file...")
        fdm = FDM()
        
        # Set up a simple 3D grid
        fdm.set_header(x0=0.0, y0=0.0, z0=0.0,
                       nx=10, ny=10, nz=5,
                       dx=1.0, dy=1.0, dz=1.0)
        
        # Create some sample data
        import numpy as np
        data = np.random.rand(5, 10, 10).astype(np.float32)
        
        # Write to file
        fdm.write("example.fdm", data)
        
        # Read it back
        fdm2 = FDM("example.fdm")
        fdm2.read()
        fdm2.print_header()
        
        print(f"Data shape: {fdm2.data.shape}")
        print(f"Data range: {fdm2.data.min()} to {fdm2.data.max()}")
