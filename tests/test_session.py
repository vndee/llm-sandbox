import os
import tarfile
import unittest
from io import BytesIO
from unittest.mock import patch, MagicMock
from llm_sandbox import SandboxSession


class TestSandboxSession(unittest.TestCase):
    @patch("docker.from_env")
    def setUp(self, mock_docker_from_env):
        self.mock_docker_client = MagicMock()
        mock_docker_from_env.return_value = self.mock_docker_client

        self.image = "python:3.9.19-bullseye"
        self.dockerfile = None
        self.lang = "python"
        self.keep_template = False
        self.verbose = False

        self.session = SandboxSession(
            image=self.image,
            dockerfile=self.dockerfile,
            lang=self.lang,
            keep_template=self.keep_template,
            verbose=self.verbose,
        )

    def test_init_with_invalid_lang(self):
        with self.assertRaises(ValueError):
            SandboxSession(lang="invalid_language")

    def test_init_with_both_image_and_dockerfile(self):
        with self.assertRaises(ValueError):
            SandboxSession(image="some_image", dockerfile="some_dockerfile")

    def test_open_with_image(self):
        self.mock_docker_client.images.get.return_value = MagicMock(
            tags=["python:3.9.19-bullseye"]
        )
        self.mock_docker_client.containers.run.return_value = MagicMock()

        self.session.open()
        self.mock_docker_client.containers.run.assert_called_once()
        self.assertIsNotNone(self.session.container)

    def test_close(self):
        mock_container = MagicMock()
        self.session.container = mock_container

        self.session.close()
        mock_container.remove.assert_called_once()
        self.assertIsNone(self.session.container)

    def test_run_without_open(self):
        with self.assertRaises(RuntimeError):
            self.session.run("print('Hello')")

    def test_run_with_code(self):
        self.session.container = MagicMock()
        self.session.execute_command = MagicMock(return_value=(0, "Output"))

        result = self.session.run("print('Hello')")
        self.session.execute_command.assert_called()
        self.assertEqual(result, (0, "Output"))

    def test_copy_to_runtime(self):
        self.session.container = MagicMock()
        src = "test.txt"
        dest = "/tmp/test.txt"
        with open(src, "w") as f:
            f.write("test content")

        self.session.copy_to_runtime(src, dest)
        self.session.container.put_archive.assert_called()

        os.remove(src)

    @patch("tarfile.open")
    def test_copy_from_runtime(self, mock_tarfile_open):
        self.session.container = MagicMock()
        src = "/tmp/test.txt"
        dest = "test.txt"

        # Create a mock tarfile
        tarstream = BytesIO()
        with tarfile.open(fileobj=tarstream, mode="w") as tar:
            tarinfo = tarfile.TarInfo(name=os.path.basename(dest))
            tarinfo.size = len(b"test content")
            tar.addfile(tarinfo, BytesIO(b"test content"))

        tarstream.seek(0)
        self.session.container.get_archive.return_value = (
            [tarstream.read()],
            {"size": tarstream.__sizeof__()},
        )

        def mock_extractall(path):
            with open(dest, "wb") as f:
                f.write(b"test content")

        mock_tarfile = MagicMock()
        mock_tarfile.extractall.side_effect = mock_extractall
        mock_tarfile_open.return_value.__enter__.return_value = mock_tarfile

        self.session.copy_from_runtime(src, dest)
        self.assertTrue(os.path.exists(dest))

        os.remove(dest)

    def test_execute_command(self):
        mock_container = MagicMock()
        self.session.container = mock_container

        command = "echo 'Hello'"
        mock_container.exec_run.return_value = (0, b"Hello\n")

        exit_code, output = self.session.execute_command(command)
        mock_container.exec_run.assert_called_with(command)
        self.assertEqual(exit_code, 0)
        self.assertEqual(output, "Hello\n")


if __name__ == "__main__":
    unittest.main()
