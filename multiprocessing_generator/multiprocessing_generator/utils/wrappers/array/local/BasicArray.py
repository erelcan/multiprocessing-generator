import random

from multiprocessing_generator.utils.wrappers.array.local.ReadOnlyArray import ReadOnlyArray


class BasicArray(ReadOnlyArray):
    def __init__(self, array):
        super().__init__(array)

    def shuffle(self):
        random.shuffle(self._array)
