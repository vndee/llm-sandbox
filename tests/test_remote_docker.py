import docker
from llm_sandbox import SandboxSession

docker_host="tcp://<remote-docker-ip>:2375"

docker_client = docker.DockerClient(base_url=docker_host)

with SandboxSession(
    image="python:3.9.19-bullseye",
    client=docker_client,
    keep_template=True,
    lang="python",
) as session:
    result = session.run("print('Hello, World!')")
    print(result)
    output1 = session.run("import time\ntime.sleep(60)")
    print(output1.text)