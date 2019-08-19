from multiprocessing_generator.core.data_readers.DataReaderFactory import create_data_reader
from multiprocessing_generator.core.preprocessors.PreprocessorFactory import create_preprocessor
from multiprocessing_generator.core.sample_weighters.SampleWeighterFactory import create_sample_weighter
from multiprocessing_generator.core.generators.sp.ExampleBatchManager import ExampleBatchManager


# In the future, dynamically load batch_manager. Hence, the generator will become much more generic.

def generator(sample_ids, data_locations, data_reader_info, preprocessor_info, sample_weighter_info, **kwargs):
    batch_manager = ExampleBatchManager(sample_ids, data_locations, kwargs["batch_size"])
    data_reader = create_data_reader(data_reader_info["id"], data_reader_info["parameters"])
    preprocessor = None
    sample_weighter = None

    if preprocessor_info is not None:
        preprocessor = create_preprocessor(preprocessor_info["id"], preprocessor_info["parameters"])
    if sample_weighter_info is not None:
        sample_weighter = create_sample_weighter(sample_weighter_info["id"], sample_weighter_info["parameters"])

    completed_epoch = kwargs["epoch_info"]["last_completed_epoch"]
    target_epoch = kwargs["epoch_info"]["target_epoch"]
    iteration_info = batch_manager.get_iteration_info()
    while completed_epoch <= target_epoch:
        for batch_num in range(iteration_info["full_batch_count"]):
            batch_info = batch_manager.get_batch_info(batch_num)
            yield prepare_batch(batch_info, data_reader, preprocessor, sample_weighter)

        if iteration_info["has_remainder"]:
            batch_info = batch_manager.get_remainder_batch_info()
            yield prepare_batch(batch_info, data_reader, preprocessor, sample_weighter)

        if kwargs["epoch_info"]["epoch_end_handler_info"]["shuffle_on"]:
            sample_ids.shuffle()


def prepare_batch(batch_info, data_reader, preprocessor, sample_weighter):
    feature_batch, label_batch = data_reader.read_data(batch_info)

    if preprocessor is not None:
        feature_batch, label_batch = preprocessor.process(feature_batch, label_batch)

    if sample_weighter is None:
        return feature_batch, label_batch
    else:
        # sample_weighter might not need some of these parameters, but it is to assure that we pass for needy ones, in case.
        weighter_info = {"batch_info": batch_info, "feature_batch": feature_batch, "label_batch": label_batch}
        weight_batch = sample_weighter(weighter_info)
        return feature_batch, label_batch, weight_batch
