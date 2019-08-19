import random

from multiprocessing_generator.utils.wrappers.array.shared.ReadOnlyNonLockArray import ReadOnlyNonLockArray


class NonLockArray(ReadOnlyNonLockArray):
    def __init__(self, shared_array):
        super().__init__(shared_array)

    def shuffle(self):
        random.shuffle(self._array)
