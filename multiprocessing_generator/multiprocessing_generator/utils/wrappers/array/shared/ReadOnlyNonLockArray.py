class ReadOnlyNonLockArray(object):
    def __init__(self, shared_array):
        self._array = shared_array

    def get_value_at(self, index):
        return self._array[index]

    def get_values_for_interval(self, start_index, end_index):
        # [start_index, end_index)
        return self._array[start_index: end_index]

    def get_values_for_index_list(self, index_list, ignore_nones=True):
        # Maybe assume the index_list is appropriate and remove the additional computation.
        # Also may add a feature to remove Nones if requested...
        result = [self._array[index] if index < len(self._array) else None for index in index_list]
        if not ignore_nones:
            check = any([elem is None for elem in result])
            if check:
                raise Exception("Some indices does not exist...")
        return result

    def get_length(self):
        return len(self._array)
