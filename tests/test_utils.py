import docker
import unittest
from unittest.mock import patch, MagicMock
from llm_sandbox.utils import image_exists, pull_image, build_image, create_container, start_container, stop_container, remove_container, remove_image


class TestUtilities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = docker.from_env()
        cls.image = 'alpine'
        cls.tag = 'latest'
        cls.container_name = 'test_container'
        cls.container_id = 'test_container_id'

    def test_pull_image(self):
        result = pull_image(self.client, self.image)
        self.assertEqual(result, [f"{self.image}:{self.tag}"])

    def test_image_exists(self):
        self.assertTrue(image_exists(self.client, self.image))


if __name__ == '__main__':
    unittest.main()
