from llm_sandbox import SandboxSession


def run_python_code():
    with SandboxSession(
        lang="python", keep_template=True, verbose=True, backend="kubernetes"
    ) as session:
        output = session.run("print('Hello, World!')")
        print(output.text)

        output = session.run(
            "import numpy as np\nprint(np.random.rand())", libraries=["numpy"]
        )
        print(output.text)

        session.execute_command("pip install pandas")
        output = session.run("import pandas as pd\nprint(pd.__version__)")
        print(output.text)

        session.copy_to_runtime("README.md", "/sandbox/data.csv")


def run_java_code():
    with SandboxSession(
        lang="java", keep_template=True, verbose=True, backend="kubernetes"
    ) as session:
        output = session.run(
            """
            public class Main {
                public static void main(String[] args) {
                    System.out.println("Hello, World!");
                }
            }
            """,
        )
        print(output.text)


def run_javascript_code():
    with SandboxSession(
        lang="javascript", keep_template=True, verbose=True, backend="kubernetes"
    ) as session:
        output = session.run("console.log('Hello, World!')")
        print(output.text)

        output = session.run(
            """
            const axios = require('axios');
            axios.get('https://jsonplaceholder.typicode.com/posts/1',
            {
                timeout: 5000,
                headers: {
                    Accept: 'application/json',
                },
            }).then(response => console.log(response.data));
            """,
            libraries=["axios"],
        )
        print(output.text)


def run_cpp_code():
    with SandboxSession(
        lang="cpp", keep_template=True, verbose=True, backend="kubernetes"
    ) as session:
        output = session.run(
            """
            #include <iostream>
            int main() {
                std::cout << "Hello, World!" << std::endl;
                return 0;
            }
            """,
        )
        print(output.text)

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
        print(output.text)

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
        print(output.text)


def run_go_code():
    with SandboxSession(
        lang="go", keep_template=True, verbose=True, backend="kubernetes"
    ) as session:
        output = session.run(
            """
            package main
            import "fmt"
            func main() {
                fmt.Println("Hello, World!")
            }
            """,
        )
        print(output.text)

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
        print(output.text)


if __name__ == "__main__":
    run_python_code()
    run_java_code()
    run_javascript_code()
    run_cpp_code()
    run_go_code()
