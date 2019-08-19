from abc import ABC, abstractmethod


class BatchManagerABC(ABC):
    def __init__(self):
        super(BatchManagerABC, self).__init__()

    @abstractmethod
    def get_iteration_info(self):
        pass

    @abstractmethod
    def get_batch_info(self, batch_num):
        pass

    @abstractmethod
    def get_remainder_batch_info(self):
        pass
