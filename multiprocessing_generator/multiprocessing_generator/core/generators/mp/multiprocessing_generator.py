import multiprocessing
import os
from copy import deepcopy

from multiprocessing_generator.core.generators.mp.ExampleBatchManager import ExampleBatchManager
from multiprocessing_generator.core.generators.mp.GenericDataProducer import GenericDataProducer


# In the future, dynamically load batch_managers. Hence, the generator will become much more generic.

# Note that sample_ids and data_locations are wrappers with shared variables.

def generator(sample_ids, data_locations, data_reader_info, preprocessor_info, sample_weighter_info, **kwargs):
    batch_queue = multiprocessing.Queue(maxsize=kwargs["mp_info"]["max_queue_size"])
    end_epoch_condition = multiprocessing.Condition()
    data_producer_kwargs = deepcopy(kwargs)
    data_producer_kwargs["end_epoch_condition"] = end_epoch_condition

    workers = []
    for worker_id in range(kwargs["mp_info"]["num_of_workers"]):
        workers.append(GenericDataProducer(ExampleBatchManager(batch_queue, sample_ids, data_locations, kwargs["chunk_info"]["chunks"][worker_id], kwargs["batch_size"]), data_reader_info, preprocessor_info=preprocessor_info, sample_weighter_info=sample_weighter_info, **data_producer_kwargs))
        workers[-1].start()

    samples_read_in_epoch = 0
    completed_epoch = kwargs["epoch_info"]["last_completed_epoch"]
    while True:
        current_batch = batch_queue.get()
        samples_read_in_epoch += len(current_batch[0])
        print("\n" + str(samples_read_in_epoch) + "\n")
        if samples_read_in_epoch == kwargs["chunk_info"]["samples_per_epoch"]:
            completed_epoch += 1
            samples_read_in_epoch=0
            if completed_epoch == kwargs["epoch_info"]["target_epoch"]:
                for worker in workers:
                    worker.terminate()
                batch_queue.close()
                yield current_batch
                break
            else:
                handle_epoch_end(sample_ids, end_epoch_condition, kwargs["epoch_info"]["epoch_end_handler_info"])
                yield current_batch
        else:
            yield current_batch


def handle_epoch_end(sample_ids, end_epoch_condition, epoch_end_handler_info):
    if epoch_end_handler_info["shuffle_on"]:
        sample_ids.shuffle()
    print("\n Will notify all: " + str(os.getpid()))
    with end_epoch_condition:
        end_epoch_condition.notify_all()
        print("\n Notified all: " + str(os.getpid()))
