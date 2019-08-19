from multiprocessing_generator.core.generators.mp.BatchManagerABC import BatchManagerABC

# There might be several basic batch managers and the design is flexible to create custom ones as well.
# For example, this batch manager can be used for datasets such that we have a list of data locations (local-fs/distributed-fs paths, database ids, matrix ids etc.).
# This batch manager assumes that labels can be figured out from the location info.
# Other main alternatives might be:
# - Location info + corresponding label list
# - Location info for both data and labels (e.g. images with masks)
# - Hierarchic/structured location info + ...
# For such alternatives and others, we may create corresponding batch managers, data readers, preprocessors, sample-weighters etc.
# We may determine such basic categories and create corresponding classes & abstract classes and have a better directory structure.
# For now, let's aim for PoC implementation.

# data_locations is a wrapper for shared array (i.e. mp.Array)
# We may wrap shared variables if we would like to control the access pattern (read-only, synchronized read/write etc.).
# For example, we do not want this class to modify the shared array; so that  we passed a read-only array wrapper.
# (We do not need a synchronized array in this case, so using ReadOnlyNonLockArray as the wrapper)

# chunk_info: [start, end)

# sample_ids will be used to keep reference to the actual data in data_locations. It's purpose to avoid shuffling
# data_locations. In other words, play with sample_ids rather than data_locations for any need. This separation allows
# handling data_locations when it has a different structure from a simple array (also allows shufflers to be
# data agnostic). Note that an element of sample_ids does not have to be a primitive but any object.

# Notice that sample_ids and data_locations should be compatible (e.g. get_values_in_index_list method of
# data_locations assumes a specific structure for (a subset of) sample_ids; and has an appropriate lookup functionality).
# For this class, sample_ids is an instance of NonLockArray...

# To decide the format/structure of batch_info to be returned is not the responsibility of data_locations, it is
# the responsibility of batch manager. Hence, write get_batch_info and get_remainder_batch_info accordingly.


class ExampleBatchManager(BatchManagerABC):
    def __init__(self, batch_queue, sample_ids, data_locations, chunk_info, batch_size):
        super(ExampleBatchManager, self).__init__()
        self._batch_queue = batch_queue
        self._sample_ids = sample_ids
        self._data_locations = data_locations
        self._chunk_info = chunk_info
        self._batch_size = batch_size

    def get_iteration_info(self):
        chunk_size = self._chunk_info["end"] - self._chunk_info["start"]
        full_batch_count = chunk_size // self._batch_size
        remainder = chunk_size % self._batch_size
        if remainder == 0:
            has_remainder = False
        else:
            has_remainder = True

        return {"full_batch_count": full_batch_count, "has_remainder": has_remainder}

    def get_batch_info(self, batch_num):
        batch_ids = self._sample_ids.get_values_for_interval(self._chunk_info["start"] + batch_num * self._batch_size, self._chunk_info["start"] + (batch_num + 1) * self._batch_size)
        return {"data_locations": self._data_locations.get_values_for_index_list(batch_ids)}

    def get_remainder_batch_info(self):
        full_batch_count = (self._chunk_info["end"] - self._chunk_info["start"]) // self._batch_size
        batch_ids = self._sample_ids.get_values_for_interval(self._chunk_info["start"] + full_batch_count * self._batch_size, self._chunk_info["end"])
        return {"data_locations": self._data_locations.get_values_for_index_list(batch_ids)}

    def send_batch(self, batch):
        self._batch_queue.put(batch)
