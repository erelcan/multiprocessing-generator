from test import dummy_data_creator
import os
import time

import multiprocessing
from ctypes import c_wchar_p

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from multiprocessing_generator.core.chunkers.example_chunker import create_chunks, compute_steps_per_epoch
from multiprocessing_generator.core.generators.mp import multiprocessing_generator
from multiprocessing_generator.core.generators.sp import singleprocessing_generator

from multiprocessing_generator.utils.wrappers.array.shared.NonLockArray import NonLockArray
from multiprocessing_generator.utils.wrappers.array.shared.ReadOnlyNonLockArray import ReadOnlyNonLockArray
from multiprocessing_generator.utils.wrappers.array.local.BasicArray import BasicArray
from multiprocessing_generator.utils.wrappers.array.local.ReadOnlyArray import ReadOnlyArray
from multiprocessing_generator.utils.common import get_file_list


def get_dummy_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


# In the future, there will be classes/modules that are responsible for such shared variable creation according to the
# given "style". (For example, this function handles samples that is list of integers and data_locations that are list
# of strings)
# ExampleBatchManager handles a variety of "styles", but we may develop new ones if required. However, ensure that
# wrappers have appropriate interfaces and compatible with each other.

def create_shared_wrappers(sample_ids, data_locations):
    shared_sample_ids = NonLockArray(multiprocessing.Array("i", sample_ids))
    shared_data_locations = ReadOnlyNonLockArray(multiprocessing.Array(c_wchar_p, data_locations))
    return shared_sample_ids, shared_data_locations


def create_wrappers(sample_ids, data_locations):
    sample_ids_wrapper = BasicArray(sample_ids)
    data_locations_wrapper = ReadOnlyArray(data_locations)
    return sample_ids_wrapper, data_locations_wrapper


def do_experiment(experiment_info):
    data_locations = get_file_list(experiment_info["data_path"], "**/*.jpg", True)
    sample_ids = list(range(len(data_locations)))

    kwargs = {"batch_size": experiment_info["batch_size"], "epoch_info": experiment_info["epoch_info"]}

    if experiment_info["mp_info"]["is_active"]:
        sample_ids_wrapper, data_locations_wrapper = create_shared_wrappers(sample_ids, data_locations)
        chunks, samples_per_epoch = create_chunks(len(sample_ids), experiment_info["mp_info"]["num_of_workers"])
        steps_per_epoch = compute_steps_per_epoch(chunks, experiment_info["batch_size"])
        kwargs["mp_info"] = experiment_info["mp_info"]
        kwargs["chunk_info"] = {"chunks": chunks, "samples_per_epoch": samples_per_epoch}
        current_generator = multiprocessing_generator.generator(sample_ids_wrapper, data_locations_wrapper, experiment_info["data_reader_info"], experiment_info["preprocessor_info"], experiment_info["sample_weighter_info"], **kwargs)
    else:
        sample_ids_wrapper, data_locations_wrapper = create_wrappers(sample_ids, data_locations)
        steps_per_epoch = sample_ids_wrapper.get_length() // experiment_info["batch_size"]
        if sample_ids_wrapper.get_length() % experiment_info["batch_size"] > 0:
            steps_per_epoch += 1
        current_generator = singleprocessing_generator.generator(sample_ids_wrapper, data_locations_wrapper, experiment_info["data_reader_info"], experiment_info["preprocessor_info"], experiment_info["sample_weighter_info"], **kwargs)

    model = get_dummy_model()

    start = time.time()
    model.fit_generator(current_generator, verbose=1, max_queue_size=10, steps_per_epoch=steps_per_epoch, use_multiprocessing=False, epochs=kwargs["epoch_info"]["target_epoch"], initial_epoch=kwargs["epoch_info"]["last_completed_epoch"])
    print("Time elapsed: %s" % (time.time() - start))


def create_dummy_data(dest_path, num_of_classes):
    sample_size = 1000
    image_size = (224, 224, 3)

    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
        for cl in range(num_of_classes):
            os.mkdir(os.path.join(dest_path, str(cl)))
        dummy_data_creator.create(sample_size, num_of_classes, image_size, dest_path)


if __name__ == '__main__':
    # Note that when training procedure is designed generic, the structure of experiment_info will be better..
    exp_info = {
        "mp_info": {
            "is_active": True,
            "num_of_workers": 4,
            "max_queue_size": 10
        },
        "epoch_info": {
            "last_completed_epoch": 0,
            "target_epoch": 2,
            "epoch_end_handler_info": {
                "shuffle_on": True
            }
        },
        "batch_size": 25,
        "num_of_classes": 2,
        "data_reader_info": {
            "id": "example_data_reader",
            "parameters": {"num_of_classes": 2}
        },
        "preprocessor_info": {
            "id": "example_preprocessor",
            "parameters": {"wait_time": 1}
        },
        "sample_weighter_info": None,
        "data_path": "/home/matrixengineer/ALL/Temporary/dummy_images"
    }

    create_dummy_data(exp_info["data_path"], exp_info["num_of_classes"])
    do_experiment(exp_info)

