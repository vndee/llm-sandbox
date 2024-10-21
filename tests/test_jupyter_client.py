import docker
import time
from jupyter_client import KernelManager, BlockingKernelClient
import json


class InteractivePythonDockerSession:
    def __init__(self, image="ipython-kernel"):
        self.docker_client = docker.from_env()
        self.image = image
        self.container = None
        self.kernel_connection_file = "/root/.ipython/profile_default/security/kernel.json"
        self.km = None
        self.kc = None

    def open(self):
        # try:
        # Start the Docker container with IPython kernel
        print("Starting Docker container with IPython kernel...")
        self.container = self.docker_client.containers.run(
            self.image,
            command=f"python3 -m ipykernel -f {self.kernel_connection_file}",
            detach=True,
            tty=True,
            stdin_open=True
        )
        print(f"Container ID: {self.container.id}")

        # Wait for the container to start the kernel and expose the port
        time.sleep(3)

        # Locate and parse the kernel connection file inside the Docker container
        self._fetch_kernel_connection_info()

        # Initialize the kernel manager and client
        self.km = KernelManager(kernel_name='python3')
        self.km.load_connection_info(self.kernel_connection_file)
        # self.km.start_kernel()
        self.kc = self.km.client()
        # self.kc.start_channels()
        # self.kc.wait_for_ready()

        # except docker.errors.DockerException as e:
        #     print(f"Docker error: {e}")
        # except Exception as e:
        #     print(f"Error initializing session: {e}")

    def _fetch_kernel_connection_info(self):
        # Example of fetching kernel connection details
        # In a real scenario, you may have to copy the file from the container
        exec_command = f"cat {self.kernel_connection_file}"
        exec_result = self.container.exec_run(exec_command)

        # Parse kernel connection file (adjust path if necessary)
        connection_info = json.loads(exec_result.output.decode())
        print(f"Kernel connection info: {connection_info}")
        self.kernel_connection_file = connection_info

    def execute_cell(self, code):
        if not self.kc:
            raise RuntimeError("Kernel client not initialized. Call open() first.")

        print(f"Executing code: {code}")
        msg_id = self.kc.execute(code)

        # Wait for the result of the execution
        output = ""
        while True:
            try:
                # Get a message from the kernel
                msg = self.kc.get_iopub_msg(timeout=1)
                print(msg)
                content = msg["content"]

                print(msg)

                # Handle different message types
                if msg["msg_type"] == "stream" and content.get("name") == "stdout":
                    # Handle standard output
                    output += content["text"]
                elif msg["msg_type"] == "execute_result":
                    # Handle execution results (e.g., return values of expressions)
                    output += content["data"]["text/plain"]
                elif msg["msg_type"] == "error":
                    # Handle execution errors
                    output += "Error executing cell:\n"
                    output += "\n".join(content["traceback"])
                    break
                elif msg["msg_type"] == "status" and content.get("execution_state") == "idle":
                    # Execution is complete when we receive an "idle" status
                    break
                else:
                    print(f"Unhandled message: {msg}")
            except Exception as e:
                output += f"Exception occurred: {e}\n"
                break

        return output

    def close(self):
        # Stop the kernel and clean up the Docker container
        try:
            if self.kc:
                print("Stopping kernel channels...")
                self.kc.stop_channels()
            if self.km:
                print("Shutting down kernel...")
                self.km.shutdown_kernel()

            if self.container:
                print("Stopping and removing Docker container...")
                self.container.stop()
                self.container.remove()

        except docker.errors.DockerException as e:
            print(f"Docker error during cleanup: {e}")
        except Exception as e:
            print(f"Error during cleanup: {e}")

        print("Session closed.")


# Example usage:
if __name__ == "__main__":
    session = InteractivePythonDockerSession()

    # try:
    # Open a new session
    session.open()

    # Define code cells to execute
    cells = [
        "x = 10",  # Cell 1: Define variable x
        "y = x * 2",  # Cell 2: Use x to compute y
        "z = y + 5",  # Cell 3: Use y to compute z
        "print(f'x={x}, y={y}, z={z}')",  # Cell 4: Print the values of x, y, and z
    ]

    # Execute each cell and print the results
    for idx, code in enumerate(cells):
        print(f"\nCell {idx + 1} output:")
        result = session.execute_cell(code)
        print(result)

    # finally:
    #     Ensure the session is closed and resources are cleaned up
    session.close()
