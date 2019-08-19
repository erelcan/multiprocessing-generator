from multiprocessing_generator.utils.common import get_class_instance

__sample_weighter_info = {

}


def create_sample_weighter(sample_weighter_id, parameters):
    return get_class_instance(__sample_weighter_info[sample_weighter_id]["module_name"], __sample_weighter_info[sample_weighter_id]["class_name"], parameters)
