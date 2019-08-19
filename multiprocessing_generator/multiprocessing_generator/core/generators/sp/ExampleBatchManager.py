from multiprocessing_generator.core.generators.sp.BatchManagerABC import BatchManagerABC


class ExampleBatchManager(BatchManagerABC):
    def __init__(self, sample_ids, data_locations, batch_size):
        super(ExampleBatchManager, self).__init__()
        self._sample_ids = sample_ids
        self._data_locations = data_locations
        self._batch_size = batch_size

    def get_iteration_info(self):
        sample_size = self._sample_ids.get_length()
        full_batch_count = sample_size // self._batch_size
        remainder = sample_size % self._batch_size
        if remainder == 0:
            has_remainder = False
        else:
            has_remainder = True

        return {"full_batch_count": full_batch_count, "has_remainder": has_remainder}

    def get_batch_info(self, batch_num):
        batch_ids = self._sample_ids.get_values_for_interval(batch_num * self._batch_size, (batch_num + 1) * self._batch_size)
        return {"data_locations": self._data_locations.get_values_for_index_list(batch_ids)}

    def get_remainder_batch_info(self):
        sample_size = self._sample_ids.get_length()
        full_batch_count = sample_size // self._batch_size
        batch_ids = self._sample_ids.get_values_for_interval(full_batch_count * self._batch_size, sample_size)
        return {"data_locations": self._data_locations.get_values_for_index_list(batch_ids)}
