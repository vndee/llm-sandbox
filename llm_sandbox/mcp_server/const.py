from llm_sandbox import SupportedLanguage
from llm_sandbox.mcp_server.types import LanguageDetails

LANGUAGE_RESOURCES: dict[str, LanguageDetails] = {
    SupportedLanguage.PYTHON.value: {
        "version": "3.11",
        "package_manager": "pip",
        "preinstalled_libraries": [
            "numpy",
            "pandas",
            "matplotlib",
            "pillow",
            "seaborn",
            "scikit-learn",
            "scipy",
            "scikit-image",
            "plotly",
        ],
        "use_cases": [
            "Data science",
            "Web development",  # NOSONAR
            "Automation",
            "Machine learning",
        ],
        "visualization_support": True,
        "examples": [
            {
                "title": "Hello World",  # NOSONAR
                "description": "Simple Python hello world program",
                "code": "print('Hello, World!')",
            },
            {
                "title": "Data Analysis with Pandas",
                "description": "Basic data analysis using pandas",
                "code": """import pandas as pd
import numpy as np

# Create sample data
data = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'score': [85, 92, 78, 96]
})

print("Dataset:")
print(data)
print("\\nAverage score:", data['score'].mean())""",
            },
            {
                "title": "Data Visualization",
                "description": "Creating plots with matplotlib",
                "code": """import matplotlib.pyplot as plt
import numpy as np

# Generate data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)', linewidth=2)
plt.plot(x, y2, label='cos(x)', linewidth=2)
plt.title('Trigonometric Functions')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()""",
            },
        ],
    },
    SupportedLanguage.JAVASCRIPT.value: {
        "version": "Node.js 22",
        "package_manager": "npm",
        "preinstalled_libraries": [],
        "use_cases": [
            "Web development",
            "APIs",
            "Frontend apps",
            "Server-side scripting",
        ],
        "visualization_support": False,
        "examples": [
            {
                "title": "Hello World",
                "description": "Simple JavaScript hello world program",
                "code": "console.log('Hello, World!');",
            },
            {
                "title": "Async Operations",
                "description": "Working with promises and async/await",
                "code": """async function fetchData() {
    try {
        const response = await fetch('https://jsonplaceholder.typicode.com/posts/1');
        const data = await response.json();
        console.log('Post title:', data.title);
    } catch (error) {
        console.error('Error:', error);
    }
}

fetchData();""",
            },
            {
                "title": "Data Processing",
                "description": "Array manipulation and functional programming",
                "code": """const numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

// Filter even numbers and square them
const evenSquares = numbers
    .filter(n => n % 2 === 0)
    .map(n => n * n);

console.log('Even squares:', evenSquares);

// Calculate sum
const sum = numbers.reduce((acc, n) => acc + n, 0);
console.log('Sum:', sum);""",
            },
        ],
    },
    SupportedLanguage.JAVA.value: {
        "version": "OpenJDK 11",
        "package_manager": "Maven",
        "preinstalled_libraries": [],
        "use_cases": ["Enterprise applications", "Android development", "Web services"],
        "visualization_support": False,
        "examples": [
            {
                "title": "Hello World",
                "description": "Simple Java hello world program",
                "code": """public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}""",
            },
            {
                "title": "Object-Oriented Programming",
                "description": "Basic class and object example",
                "code": """class Person {
    private String name;
    private int age;

    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public void introduce() {
        System.out.println("Hi, I'm " + name + " and I'm " + age + " years old.");
    }
}

public class Main {
    public static void main(String[] args) {
        Person person = new Person("Alice", 25);
        person.introduce();
    }
}""",
            },
        ],
    },
    SupportedLanguage.CPP.value: {
        "version": "GCC 11.2.0",
        "package_manager": "System packages",
        "preinstalled_libraries": ["STL"],
        "use_cases": ["System programming", "Performance-critical applications", "Game development"],
        "visualization_support": False,
        "examples": [
            {
                "title": "Hello World",
                "description": "Simple C++ hello world program",
                "code": """#include <iostream>
int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}""",
            },
            {
                "title": "STL Vector Operations",
                "description": "Working with vectors and algorithms",
                "code": """#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> numbers = {5, 2, 8, 1, 9, 3};

    std::cout << "Original: ";
    for (int n : numbers) {
        std::cout << n << " ";
    }
    std::cout << std::endl;

    std::sort(numbers.begin(), numbers.end());

    std::cout << "Sorted: ";
    for (int n : numbers) {
        std::cout << n << " ";
    }
    std::cout << std::endl;

    return 0;
}""",
            },
        ],
    },
    SupportedLanguage.GO.value: {
        "version": "1.23.4",
        "package_manager": "go mod",
        "preinstalled_libraries": [],
        "use_cases": ["Microservices", "CLI tools", "Network programming", "Cloud native apps"],
        "visualization_support": False,
        "examples": [
            {
                "title": "Hello World",
                "description": "Simple Go hello world program",
                "code": """package main
import "fmt"

func main() {
    fmt.Println("Hello, World!")
}""",
            },
            {
                "title": "Goroutines and Channels",
                "description": "Concurrent programming with goroutines",
                "code": """package main
import (
    "fmt"
    "time"
)

func worker(id int, jobs <-chan int, results chan<- int) {
    for j := range jobs {
        fmt.Printf("Worker %d processing job %d\\n", id, j)
        time.Sleep(time.Second)
        results <- j * 2
    }
}

func main() {
    jobs := make(chan int, 100)
    results := make(chan int, 100)

    // Start 3 workers
    for w := 1; w <= 3; w++ {
        go worker(w, jobs, results)
    }

    // Send 5 jobs
    for j := 1; j <= 5; j++ {
        jobs <- j
    }
    close(jobs)

    // Collect results
    for a := 1; a <= 5; a++ {
        <-results
    }
}""",
            },
        ],
    },
    SupportedLanguage.R.value: {
        "version": "4.5.1",
        "package_manager": "CRAN",
        "preinstalled_libraries": [
            "tidyverse",
            "data.table",
            "plotly",
            "gridExtra",
            "RColorBrewer",
            "viridis",
            "caret",
            "randomForest",
            "glmnet",
            "broom",
            "forecast",
            "zoo",
            "xts",
            "lubridate",
            "tm",
            "stringr",
            "devtools",
            "here",
            "janitor",
            "BiocManager",
            "Biobase",
            "BiocGenerics",
            "S4Vectors",
            "IRanges",
            "GenomeInfoDb",
            "GenomicRanges",
            "GenomicFeatures",
            "AnnotationDbi",
            "org.Hs.eg.db",
            "DESeq2",
            "edgeR",
            "limma",
            "ComplexHeatmap",
            "EnhancedVolcano",
            "biomaRt",
            "GO.db",
            "KEGGREST",
            "clusterProfiler",
        ],
        "use_cases": [
            "Statistical analysis",
            "Data visualization",
            "Research",
            "Bioinformatics",
        ],
        "visualization_support": True,
        "examples": [
            {
                "title": "Hello World",
                "description": "Simple R hello world program",
                "code": """print("Hello, World!")

# Basic R operations
numbers <- c(1, 2, 3, 4, 5)
print(paste("Numbers:", paste(numbers, collapse=", ")))
print(paste("Mean:", mean(numbers)))""",
            },
            {
                "title": "Data Analysis with dplyr",
                "description": "Data manipulation using tidyverse",
                "code": """library(dplyr)

# Create sample data
employees <- data.frame(
    name = c("John", "Jane", "Bob", "Alice", "Charlie"),
    department = c("IT", "HR", "IT", "Finance", "HR"),
    salary = c(75000, 65000, 80000, 70000, 60000),
    experience = c(5, 3, 7, 4, 2)
)

print("Employee Data:")
print(employees)

# Calculate average salary by department
avg_by_dept <- employees %>%
    group_by(department) %>%
    summarise(
        avg_salary = mean(salary),
        avg_experience = mean(experience),
        count = n()
    )

print("Average by Department:")
print(avg_by_dept)""",
            },
            {
                "title": "Data Visualization with ggplot2",
                "description": "Creating plots with ggplot2",
                "code": """library(ggplot2)

# Create sample data
data <- data.frame(
    x = rnorm(100),
    y = rnorm(100),
    group = sample(c("A", "B", "C"), 100, replace = TRUE)
)

# Create scatter plot
p <- ggplot(data, aes(x = x, y = y, color = group)) +
    geom_point(alpha = 0.7, size = 2) +
    geom_smooth(method = "lm", se = FALSE) +
    labs(
        title = "Scatter Plot by Group",
        x = "X values",
        y = "Y values",
        color = "Group"
    ) +
    theme_minimal()

print(p)""",
            },
        ],
    },
    SupportedLanguage.RUBY.value: {
        "version": "Ruby 3.0.2",
        "package_manager": "gem",
        "preinstalled_libraries": [],
        "use_cases": ["Web development", "Scripting", "Automation", "APIs"],
        "visualization_support": False,
        "examples": [
            {
                "title": "Hello World",
                "description": "Simple Ruby hello world program",
                "code": """puts "Hello, World!" """,
            },
            {
                "title": "Class and Object",
                "description": "Object-oriented programming in Ruby",
                "code": """class Person
  attr_accessor :name, :age

  def initialize(name, age)
    @name = name
    @age = age
  end

  def introduce
    puts "Hi, I'm #{@name} and I'm #{@age} years old."
  end
end

person = Person.new("Alice", 25)
person.introduce""",
            },
            {
                "title": "Array Operations",
                "description": "Working with arrays and blocks",
                "code": """numbers = [1, 2, 3, 4, 5]

puts "Original array: #{numbers}"

# Double each number
doubled = numbers.map { |n| n * 2 }
puts "Doubled: #{doubled}"

# Filter even numbers
evens = numbers.select { |n| n.even? }
puts "Even numbers: #{evens}"

# Sum all numbers
sum = numbers.reduce(0) { |acc, n| acc + n }
puts "Sum: #{sum}" """,
            },
        ],
    },
}
