import time


class ExamplePreprocessor(object):
    def __init__(self, wait_time):
        self.__wait_time = wait_time

    def process(self, feature_batch, label_batch):
        time.sleep(self.__wait_time)
        return feature_batch, label_batch
