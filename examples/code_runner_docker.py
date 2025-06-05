import logging

import docker

from llm_sandbox import SandboxSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

client = docker.DockerClient(base_url="unix:///Users/vndee/.docker/run/docker.sock")


def run_python_code() -> None:
    """Run Python code in the sandbox."""
    with SandboxSession(lang="python", keep_template=True, verbose=True) as session:
        output = session.run("print('Hello, World!')")
        logger.info(output)

        output = session.run("import numpy as np\nprint(np.random.rand())", libraries=["numpy"])
        logger.info(output)

        session.install(["pandas"])
        output = session.run("import pandas as pd\nprint(pd.__version__)")
        logger.info(output)

        session.copy_to_runtime("README.md", "/sandbox/data.csv")


def run_java_code() -> None:
    """Run Java code in the sandbox."""
    with SandboxSession(lang="java", keep_template=True, verbose=True) as session:
        output = session.run(
            """
            public class Main {
                public static void main(String[] args) {
                    System.out.println("Hello, World!");
                }
            }
            """,
        )
        logger.info(output)


def run_javascript_code() -> None:
    """Run JavaScript code in the sandbox."""
    with SandboxSession(lang="javascript", keep_template=True, verbose=True) as session:
        output = session.run("console.log('Hello, World!')")
        logger.info(output)

        output = session.run(
            """
            const axios = require('axios');
            axios.get('https://jsonplaceholder.typicode.com/posts/1')
                .then(response => console.log(response.data));
            """,
            libraries=["axios"],
        )
        logger.info(output)


def run_cpp_code() -> None:
    """Run C++ code in the sandbox."""
    with SandboxSession(lang="cpp", keep_template=True, verbose=True) as session:
        output = session.run(
            """
            #include <iostream>
            int main() {
                std::cout << "Hello, World!" << std::endl;
                return 0;
            }
            """,
        )
        logger.info(output)

        output = session.run(
            """
            #include <iostream>
            #include <vector>
            int main() {
                std::vector<int> v = {1, 2, 3, 4, 5};
                for (int i : v) {
                    std::cout << i << " ";
                }
                std::cout << std::endl;
                return 0;
            }
            """,
        )
        logger.info(output)

        # run with libraries
        output = session.run(
            """
            #include <iostream>
            #include <vector>
            #include <algorithm>
            int main() {
                std::vector<int> v = {1, 2, 3, 4, 5};
                std::reverse(v.begin(), v.end());
                for (int i : v) {
                    std::cout << i << " ";
                }
                std::cout << std::endl;
                return 0;
            }
            """,
            libraries=["libstdc++"],
        )
        logger.info(output)


def run_go_code() -> None:
    """Run Go code in the sandbox."""
    with SandboxSession(lang="go", keep_template=True, verbose=True) as session:
        output = session.run(
            """
            package main
            import "fmt"
            func main() {
                fmt.Println("Hello, World!")
            }
            """,
        )
        logger.info(output)

        # run with libraries
        output = session.run(
            """
            package main
            import (
                "fmt"
                "github.com/spyzhov/ajson"
            )
            func main() {
                fmt.Println("Hello, World!")
                json := []byte(`{"price": 100}`)

                root, _ := ajson.Unmarshal(json)
                nodes, _ := root.JSONPath("$..price")
                for _, node := range nodes {
                    node.SetNumeric(node.MustNumeric() * 1.25)
                    node.Parent().AppendObject("currency", ajson.StringNode("", "EUR"))
                }
                result, _ := ajson.Marshal(root)

                fmt.Printf("%s", result)
            }
            """,
            libraries=["github.com/spyzhov/ajson"],
        )
        logger.info(output)


if __name__ == "__main__":
    run_python_code()
    run_java_code()
    run_javascript_code()
    run_cpp_code()
    run_go_code()
