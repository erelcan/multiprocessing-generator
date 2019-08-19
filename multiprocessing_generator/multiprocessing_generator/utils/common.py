import numpy as np
from PIL import Image
import glob
import os


def get_class_instance(module_name, class_name, parameters):
    module = __import__(module_name, fromlist=[''])
    class_instance = getattr(module, class_name)(**parameters)
    return class_instance


def read_image_to_np_array(im_path):
    img = Image.open(im_path)
    img_array = np.array(img)
    return img_array


def get_file_list(dir_path, pattern, recursive=True):
    return [f for f in glob.glob(os.path.join(dir_path, pattern), recursive=recursive)]
