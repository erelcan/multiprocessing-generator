from multiprocessing_generator.utils.common import get_class_instance

__data_reader_info = {
    "example_data_reader": {"module_name": "multiprocessing_generator.core.data_readers.ExampleDataReader", "class_name": "ExampleDataReader"}
}


def create_data_reader(data_reader_id, parameters):
    return get_class_instance(__data_reader_info[data_reader_id]["module_name"], __data_reader_info[data_reader_id]["class_name"], parameters)
