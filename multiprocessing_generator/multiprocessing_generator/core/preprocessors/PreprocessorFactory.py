from multiprocessing_generator.utils.common import get_class_instance

__preprocessor_info = {
    "example_preprocessor": {"module_name": "multiprocessing_generator.core.preprocessors.ExamplePreprocessor", "class_name": "ExamplePreprocessor"}
}


def create_preprocessor(preprocessor_id, parameters):
    return get_class_instance(__preprocessor_info[preprocessor_id]["module_name"], __preprocessor_info[preprocessor_id]["class_name"], parameters)
