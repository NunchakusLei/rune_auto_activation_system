import numpy as np


# """ comment used variables """
draw_number_box_color = (255, 0, 255)
draw_number_color = (100, 230, 0)



def sort_box_points(box):
    """
    This function will sort box points in TL->BL->BR->TR order
    params box: 4 points of the box
        type: 2d np.array
    return out: 4 sorted points of the box
        tyep: 2d np.array
    """
    out = box.copy()
    xs, ys = [], []
    for i in range(4):
        xs.append(box[i][0])
        ys.append(box[i][1])
    xs.sort()
    ys.sort()

    for i in range(4):
        if (box[i][0] in xs[:2]) and (box[i][1] in ys[:2]):
            out[0] = box[i]
        elif (box[i][0] in xs[:2]) and (box[i][1] in ys[2:]):
            out[1] = box[i]
        elif (box[i][0] in xs[2:]) and (box[i][1] in ys[2:]):
            out[2] = box[i]
        elif (box[i][0] in xs[2:]) and (box[i][1] in ys[:2]):
            out[3] = box[i]
    return out



# """ comment used functions """
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
