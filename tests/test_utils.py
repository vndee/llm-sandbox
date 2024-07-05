import docker
import unittest
from llm_sandbox.utils import image_exists, pull_image, build_image


class TestUtilities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = docker.from_env()
        cls.image = 'alpine'
        cls.tag = 'latest'
        cls.container_name = 'test_container'
        cls.container_id = 'test_container_id'

    def test_pull_image(self):
        self.assertIsNotNone(pull_image(self.client, self.image, self.tag))
        self.assertTrue(image_exists(self.client, f"{self.image}:{self.tag}"))

        self.client.images.remove(f"{self.image}:{self.tag}")
        self.assertFalse(image_exists(self.client, f"{self.image}:{self.tag}"))

    def test_build_image(self):
        self.assertIsNotNone(build_image(self.client, "tests", tag="llm-sandbox"))
        self.assertTrue(image_exists(self.client, "llm-sandbox:latest"))

        self.client.images.remove("llm-sandbox:latest")
        self.assertFalse(image_exists(self.client, "llm-sandbox:latest"))

        self.assertIsNotNone(build_image(self.client, path="tests", dockerfile="busybox.Dockerfile", tag="llm-sandbox"))
        self.assertTrue(image_exists(self.client, "llm-sandbox:latest"))

        self.client.images.remove("llm-sandbox:latest")
        self.assertFalse(image_exists(self.client, "llm-sandbox:latest"))


if __name__ == '__main__':
    unittest.main()
