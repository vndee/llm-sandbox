from typing import List

from .const import SupportedLanguage


class SandboxSession:
    def __init__(self, image: str = None, dockerfile: str = None, lang: str = SupportedLanguage.PYTHON):
        # raise error if both the image and dockerfile exist since we only need one
        if image and dockerfile:
            raise ValueError("Only one of image or dockerfile should be provided")

        # raise error if neither the image nor dockerfile exist since we need one
        if not image and not dockerfile:
            raise ValueError("Either image or dockerfile should be provided")

        self.lang = lang

    def open(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def run(self, code: str, libraries: List = []):
        raise NotImplementedError

    def copy_from_runtime(self, path: str):
        raise NotImplementedError

    def copy_to_runtime(self, path: str):
        raise NotImplementedError

    def execute_command(self, command: str, shell: str = "/bin/sh"):
        raise NotImplementedError

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()