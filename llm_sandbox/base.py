from abc import ABC, abstractmethod
from typing import Optional, List


class Session(ABC):
    def __init__(self, lang: str, verbose: bool = True, *args, **kwargs):
        self.lang = lang
        self.verbose = verbose
        super().__init__(*args, **kwargs)

    @abstractmethod
    def open(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError

    @abstractmethod
    def run(self, code: str, libraries: Optional[List] = None):
        raise NotImplementedError

    @abstractmethod
    def copy_to_runtime(self, src: str, dest: str):
        raise NotImplementedError

    @abstractmethod
    def copy_from_runtime(self, src: str, dest: str):
        raise NotImplementedError

    @abstractmethod
    def execute_command(self, command: str):
        raise NotImplementedError

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()
