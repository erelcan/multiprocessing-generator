from multiprocessing import Process
import os

from multiprocessing_generator.core.data_readers.DataReaderFactory import create_data_reader
from multiprocessing_generator.core.preprocessors.PreprocessorFactory import create_preprocessor
from multiprocessing_generator.core.sample_weighters.SampleWeighterFactory import create_sample_weighter


# Preferred creating data_readers in sub-processors since some data_readers might break while copying from main process.
# (e.g. passing objects with network connections might fail~)

# Note that preprocessor_info and sample_weighter_info might need to be adjusted. (e.g. you may not provide
# sample weighter's parameters without computing class ratios. Hence, given configuration json should be updated
# when such values are computed.)

# batch_manager will return batch_info compatible with the data reader. Hence, a check to validate their
# compatibility might be useful. (For many cases, an array of data paths would be sufficient, but for some datasets,
# there might be additional requirements)

class GenericDataProducer(Process):
    def __init__(self, batch_manager, data_reader_info, preprocessor_info=None, sample_weighter_info=None, **kwargs):
        super(GenericDataProducer, self).__init__()

        self._batch_manager = batch_manager
        self._data_reader = create_data_reader(data_reader_info["id"], data_reader_info["parameters"])
        self._preprocessor = None
        self._sample_weighter = None
        self.end_epoch_condition = kwargs["end_epoch_condition"]

        if preprocessor_info is not None:
            self._preprocessor = create_preprocessor(preprocessor_info["id"], preprocessor_info["parameters"])
        if sample_weighter_info is not None:
            self._sample_weighter = create_sample_weighter(sample_weighter_info["id"], sample_weighter_info["parameters"])

    def run(self):
        while True:
            # Getting iteration_info (e.g. num_of_chunks, remainder etc.) for each iteration. Although it will be mostly
            # constant, there may be some training processes that the number of samples changes throughout the process.
            iteration_info = self._batch_manager.get_iteration_info()
            for batch_num in range(iteration_info["full_batch_count"]):
                batch_info = self._batch_manager.get_batch_info(batch_num)
                self._batch_manager.send_batch(self.prepare_batch(batch_info))

            if iteration_info["has_remainder"]:
                batch_info = self._batch_manager.get_remainder_batch_info()
                self._batch_manager.send_batch(self.prepare_batch(batch_info))

            print("\n Entering wait: " + str(os.getpid()))
            with self.end_epoch_condition:
                self.end_epoch_condition.wait()
                print("\n Notified: " + str(os.getpid()))

    def prepare_batch(self, batch_info):
        feature_batch, label_batch = self._data_reader.read_data(batch_info)

        if self._preprocessor is not None:
            feature_batch, label_batch = self._preprocessor.process(feature_batch, label_batch)

        if self._sample_weighter is None:
            return feature_batch, label_batch
        else:
            # sample_weighter might not need some of these parameters, but it is to assure that we pass for needy ones, in case.
            weighter_info = {"batch_info": batch_info, "feature_batch": feature_batch, "label_batch": label_batch}
            weight_batch = self._sample_weighter(weighter_info)
            return feature_batch, label_batch, weight_batch
