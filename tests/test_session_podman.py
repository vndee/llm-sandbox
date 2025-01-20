import os
import tarfile
import unittest
from io import BytesIO
from unittest.mock import patch, MagicMock
from llm_sandbox.podman import SandboxPodmanSession


class TestSandboxSession(unittest.TestCase):
    @patch("podman.from_env")
    def setUp(self, mock_podman_from_env):
        self.mock_podman_client = MagicMock()
        mock_podman_from_env.return_value = self.mock_podman_client

        self.image = "python:3.9.19-bullseye"
        self.dockerfile = None
        self.lang = "python"
        self.keep_template = False
        self.verbose = False

        self.session = SandboxPodmanSession(
            client=self.mock_podman_client,
            image=self.image,
            dockerfile=self.dockerfile,
            lang=self.lang,
            keep_template=self.keep_template,
            verbose=self.verbose,
        )

    def test_init_with_invalid_lang(self):
        with self.assertRaises(ValueError):
            SandboxPodmanSession(lang="invalid_language")

    def test_init_with_both_image_and_dockerfile(self):
        with self.assertRaises(ValueError):
            SandboxPodmanSession(image="some_image", dockerfile="some_dockerfile")

    def test_open_with_image(self):
        # Mock the image retrieval
        mock_image = MagicMock(tags=["python:3.9.19-bullseye"])
        self.mock_podman_client.images.get.return_value = mock_image

        # Mock the container creation
        self.mock_podman_client.containers = MagicMock()  # Ensure containers is mocked
        mock_container = MagicMock()
        self.mock_podman_client.containers.create.return_value = mock_container

        # Call the open method
        self.session.open()

        # Assert that `images.get` was called to check if the image exists
        self.mock_podman_client.images.get.assert_called_once_with(
            "python:3.9.19-bullseye"
        )

        # Assert that `containers.create` was called with the correct parameters
        self.mock_podman_client.containers.create.assert_called_once_with(
            image=mock_image,  # Use the mock returned by `images.get`
            tty=True,
            mounts=[],  # Default value when `self.mounts` is None
        )

        # Assert that `start` was called on the container
        mock_container.start.assert_called_once()

        # Assert that the `self.container` attribute was set
        self.assertEqual(self.session.container, mock_container)

    def test_close(self):
        mock_container = MagicMock()
        self.session.container = mock_container
        mock_container.commit.return_values = MagicMock(tags=["python:3.9.19-bullseye"])

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
        mock_container.exec_run.return_value = (0, iter([b"Hello\n"]))

        output = self.session.execute_command(command)

        # Assert that exec_run was called with the correct arguments
        mock_container.exec_run.assert_called_with(command, stream=True, tty=True)
        # Assert that the returned output is as expected
        self.assertEqual(output.text, "Hello\n")  # Match the `output` attribute

    def test_execute_empty_command(self):
        with self.assertRaises(ValueError):
            self.session.execute_command("")

    def test_execute_failing_command(self):
        mock_container = MagicMock()
        self.session.container = mock_container

        command = "exit 1"
        mock_container.exec_run.return_value = (1, iter([]))

        output = self.session.execute_command(command)

        mock_container.exec_run.assert_called_with(command, stream=True, tty=True)
        self.assertEqual(output.exit_code, 1)
        self.assertEqual(output.text, "")


if __name__ == "__main__":
    unittest.main()
