import os
import numpy as np
from keras.utils import np_utils

from multiprocessing_generator.utils.common import read_image_to_np_array

# Have your own utils functions, do not use utils of external libraries but wrap them (So in case they change, you do
# not have to look for all usages and changed them one by one, but instead change from single place).


class ExampleDataReader(object):
    def __init__(self, num_of_classes):
        self.__num_of_classes = num_of_classes

    def read_data(self, data_info):
        # Assuming absolute paths!!!
        data_batch = []
        label_batch = []
        for cur_path in data_info["data_locations"]:
            data_batch.append(read_image_to_np_array(cur_path))
            label_batch.append(np_utils.to_categorical(int(self.__get_label(cur_path)), num_classes=self.__num_of_classes))

        return np.asarray(data_batch), np.asarray(label_batch)

    def __get_label(self, fpath):
        return fpath.split(os.path.sep)[-2]
