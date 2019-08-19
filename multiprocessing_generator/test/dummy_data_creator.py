import numpy as np
import os
from PIL import Image


def create(sample_size, num_of_classes, image_size, dest_path):
    arr_data = np.random.randint(0, 256, (sample_size, image_size[0], image_size[1], image_size[2]))
    arr_labels = np.random.randint(0, num_of_classes, sample_size)
    for i in range(sample_size):
        image = Image.fromarray(arr_data[i].astype('uint8'), 'RGB')
        image.save(os.path.join(dest_path, str(arr_labels[i]), str(i) + ".jpg"))
