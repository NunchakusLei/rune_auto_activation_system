import numpy as np

def normalize(src):
    max_value = src.max()
    min_value = src.min()

    src = src - min_value
    src = src.astype(np.float64)
    src /= (max_value - min_value)
    src *= 255.0
    src =  src.astype(np.uint8)
    return src

def normalize_cutoff_lower(src):
    size = src.shape
    return (src > np.zeros(size)) * np.ones(size) * src

def normalize_cutoff_higher(src):
    size = src.shape
    mask = np.ones(size)*255
    not_higher_255 = (src < mask) * np.ones(size) * src
    return not_higher_255 + (src >= mask) * np.ones(size) * 255

def normalize_cutoff_higher_and_lower(src):
    src = normalize_cutoff_higher(src)
    src = normalize_cutoff_lower(src)
    return src

def compute_MSE(src, rec):
    return np.sum((src.astype("float64") - rec.astype("float64"))**2) / float(src.shape[0] * src.shape[1])

def compute_MDE(src, rec):
    return np.max(abs(src.astype("float64") - rec.astype("float64")))
