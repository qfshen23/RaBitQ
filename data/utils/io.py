import numpy as np
import struct
from tqdm import tqdm

# for reading '.xbin' format files
def read_vector_data(filepath, percent=100):
    if filepath.endswith('.fbin'):
        dtype = np.float32
    elif filepath.endswith('.u8bin'):
        dtype = np.uint8
    elif filepath.endswith('.i8bin'):
        dtype = np.int8
    elif filepath.endswith('.ibin'):
        dtype = np.int32
    else:
        raise ValueError("Unsupported file type")
    
    with open(filepath, 'rb') as file:
        header = np.frombuffer(file.read(8), dtype=np.uint32)
        num_points = np.int64(header[0])
        num_dimensions = np.int64(header[1])

        print(f"Reading No. of Points: {num_points}, No. of Dimensions: {num_dimensions}")
        total_elements = int((percent / 100) * num_points * num_dimensions)
        
        # Check total_elements does not exceed limits before attempting to read
        try:
            data = np.frombuffer(file.read(total_elements * dtype().itemsize), dtype=dtype)
            num_points_to_reshape = total_elements // num_dimensions
            data = data.reshape(num_points_to_reshape, num_dimensions)
        except MemoryError:
            raise MemoryError("Not enough memory to reshape the data. Consider processing in chunks.")
        
    return data

# for reading '.fvecs' format files
def read_fvecs(filename, c_contiguous=True):
    print(f"Reading from {filename}.")
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def read_ivecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.int32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def to_fvecs(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in tqdm(data):
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('f', x)
                fp.write(a)

def to_Ivecs(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in tqdm(data):
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('Q', x)
                fp.write(a)

def to_ivecs(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('I', x)
                fp.write(a)