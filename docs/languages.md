# Supported Languages

LLM Sandbox supports multiple programming languages, each with specific features and configurations. This guide covers language-specific details, features, and best practices.

## Overview

| Language | Version | Package Manager | Plot Support | Default Image |
|----------|---------|----------------|--------------|---------------|
| **Python** | 3.11 | pip | ✅ Full | `ghcr.io/vndee/sandbox-python-311-bullseye` |
| **R** | 4.5.1 | CRAN | ✅ Full | `ghcr.io/vndee/sandbox-r-451-bullseye` |
| **JavaScript** | Node 22 | npm | ❌ | `ghcr.io/vndee/sandbox-node-22-bullseye` |
| **Java** | 11 | Maven | ❌ | `ghcr.io/vndee/sandbox-java-11-bullseye` |
| **C++** | GCC 11.2 | apt | ❌ | `ghcr.io/vndee/sandbox-cpp-11-bullseye` |
| **Go** | 1.23.4 | go get | ❌ | `ghcr.io/vndee/sandbox-go-123-bullseye` |

## Python

### Overview

Python is the most feature-rich language in LLM Sandbox, with full support for package management, plot extraction, and data science workflows.

### Basic Usage

```python
from llm_sandbox import SandboxSession

with SandboxSession(lang="python") as session:
    result = session.run("""
import sys
print(f"Python {sys.version}")
print("Hello from Python!")
    """)
    print(result.stdout)
```

### Package Management

```python
# Install packages during code execution
with SandboxSession(lang="python") as session:
    result = session.run("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
    """, libraries=["numpy", "pandas", "matplotlib"])

# Install packages separately
with SandboxSession(lang="python") as session:
    session.install(["scikit-learn", "seaborn"])
    result = session.run("""
from sklearn import __version__
print(f"Scikit-learn version: {__version__}")
    """)
```

### Plot Extraction

Python supports automatic extraction of plots from matplotlib, seaborn, and plotly:

```python
from llm_sandbox import ArtifactSandboxSession
import base64

with ArtifactSandboxSession(lang="python") as session:
    result = session.run("""
import matplotlib.pyplot as plt
import numpy as np

# Create multiple plots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Plot 1: Line plot
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Sine Wave')

# Plot 2: Scatter plot
axes[0, 1].scatter(np.random.rand(50), np.random.rand(50))
axes[0, 1].set_title('Random Scatter')

# Plot 3: Histogram
axes[1, 0].hist(np.random.normal(0, 1, 1000), bins=30)
axes[1, 0].set_title('Normal Distribution')

# Plot 4: Bar plot
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
axes[1, 1].bar(categories, values)
axes[1, 1].set_title('Bar Chart')

plt.tight_layout()
plt.show()
    """)

    # Save extracted plots
    for i, plot in enumerate(result.plots):
        with open(f"plot_{i}.png", "wb") as f:
            f.write(base64.b64decode(plot.content_base64))
```

### Data Science Workflows

```python
with SandboxSession(lang="python") as session:
    result = session.run("""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X + np.random.randn(100, 1) * 2

# Create DataFrame
df = pd.DataFrame({'X': X.flatten(), 'y': y.flatten()})
print("Data shape:", df.shape)
print("\nData summary:")
print(df.describe())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel coefficients: {model.coef_[0][0]:.4f}")
print(f"Model intercept: {model.intercept_[0]:.4f}")
print(f"Mean squared error: {mse:.4f}")
print(f"R² score: {r2:.4f}")
    """, libraries=["pandas", "numpy", "scikit-learn"])

    print(result.stdout)
```

### Python-Specific Features

#### Virtual Environment

LLM Sandbox automatically creates a virtual environment for Python:

```python
# Virtual environment is at /tmp/venv
with SandboxSession(lang="python") as session:
    result = session.run("""
import sys
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path[0]}")
    """)
```

#### Custom Python Images

```python
# Use specific Python version
with SandboxSession(
    lang="python",
    image="python:3.12-slim"
) as session:
    pass

# Use data science image
with SandboxSession(
    lang="python",
    image="jupyter/scipy-notebook:latest"
) as session:
    pass

# Use custom image with pre-installed packages
with SandboxSession(
    lang="python",
    dockerfile="./python-ds/Dockerfile"
) as session:
    pass
```

## R

### Overview

R support includes CRAN package management and plot extraction. Comprehensive documentation is available [here](https://cran.r-project.org/).

### Basic Usage

```python
with SandboxSession(lang="r") as session:
    result = session.run("""
print("Hello from R!")
    """)
    print(result.stdout)
```

### Package Management

```python
# Install CRAN packages
with SandboxSession(lang="r") as session:
    session.install(["dplyr", "ggplot2"])
    result = session.run("""
library(dplyr)
library(ggplot2)
    """)
```

### Plot Extraction

R supports automatic extraction of plots from ggplot2:

```python
from llm_sandbox import ArtifactSandboxSession
import base64

with ArtifactSandboxSession(lang="r") as session:
    result = session.run("""
library(ggplot2)

# Create a plot
p <- ggplot(mtcars, aes(x = hp, y = mpg)) +
    geom_point() +
    labs(title = "Miles per Gallon vs Horsepower", x = "Horsepower", y = "Miles per Gallon")

# Save the plot as a PNG
png(file = "plot.png")
print(p)
dev.off()
    """)

    # Save extracted plots
    for i, plot in enumerate(result.plots):
        with open(f"plot_{i}.png", "wb") as f:
            f.write(base64.b64decode(plot.content_base64))
```

### Data Science Workflows

#### Data Manipulation with Tidyverse

```python
with SandboxSession(lang="r") as session:
    result = session.run("""
library(tidyverse)
library(data.table)

# Generate comprehensive dataset
set.seed(42)
n <- 1000

data <- tibble(
    id = 1:n,
    age = rnorm(n, mean = 35, sd = 10),
    income = exp(rnorm(n, mean = 10.5, sd = 0.5)),
    education = sample(c("High School", "Bachelor", "Master", "PhD"), n, replace = TRUE),
    performance_score = 70 + 0.3 * rnorm(n, mean = 5, sd = 2) + rnorm(n, mean = 0, sd = 5)
)

print("Data Summary:")
print(glimpse(data))

# Advanced data manipulation
summary_stats <- data %>%
    group_by(education) %>%
    summarise(
        count = n(),
        avg_age = mean(age, na.rm = TRUE),
        avg_income = mean(income, na.rm = TRUE),
        avg_performance = mean(performance_score, na.rm = TRUE),
        .groups = 'drop'
    ) %>%
    arrange(desc(avg_performance))

print("Performance Analysis:")
print(summary_stats)
    """, libraries=["tidyverse", "data.table"])
```

#### Machine Learning

```python
with SandboxSession(lang="r") as session:
    result = session.run("""
library(caret)
library(randomForest)
library(broom)

# Create dataset for modeling
set.seed(123)
n <- 500
data <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    x3 = rnorm(n),
    x4 = runif(n, -2, 2)
)
data$y <- 2 * data$x1 - 1.5 * data$x2 + 0.5 * data$x3 + rnorm(n, 0, 0.5)

# Split data
train_index <- createDataPartition(data$y, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Linear Regression
lm_model <- lm(y ~ ., data = train_data)
print("Linear Regression Coefficients:")
print(tidy(lm_model))

# Random Forest
rf_model <- randomForest(y ~ ., data = train_data, ntree = 100)
print("Random Forest Variable Importance:")
print(importance(rf_model))

# Model evaluation
lm_pred <- predict(lm_model, test_data)
rf_pred <- predict(rf_model, test_data)

lm_rmse <- sqrt(mean((test_data$y - lm_pred)^2))
rf_rmse <- sqrt(mean((test_data$y - rf_pred)^2))

print(paste("Linear Regression RMSE:", round(lm_rmse, 4)))
print(paste("Random Forest RMSE:", round(rf_rmse, 4)))
    """, libraries=["caret", "randomForest", "broom"])
```

#### Time Series Analysis

```python
with SandboxSession(lang="r") as session:
    result = session.run("""
library(forecast)
library(zoo)

# Create synthetic time series
set.seed(42)
n <- 200
trend <- seq(100, 150, length.out = n)
seasonal <- 10 * sin(2 * pi * (1:n) / 12)
noise <- rnorm(n, 0, 3)
values <- trend + seasonal + noise

# Convert to time series object
ts_data <- ts(values, frequency = 12, start = c(2020, 1))

print("Time Series Summary:")
print(summary(ts_data))

# Fit ARIMA model
arima_model <- auto.arima(ts_data)
print("ARIMA Model:")
print(arima_model)

# Generate forecasts
forecast_result <- forecast(arima_model, h = 12)
print("12-Period Forecast:")
print(as.data.frame(forecast_result))
    """, libraries=["forecast", "zoo"])
```

### Advanced Plotting

#### ggplot2 Visualizations

```python
with ArtifactSandboxSession(lang="r") as session:
    result = session.run("""
library(ggplot2)
library(dplyr)
library(gridExtra)

# Generate sample data
set.seed(42)
data <- data.frame(
    x = rnorm(1000, 50, 15),
    y = rnorm(1000, 30, 10),
    category = sample(c("A", "B", "C", "D"), 1000, replace = TRUE),
    size_var = runif(1000, 1, 5)
)

# Advanced scatter plot
p1 <- ggplot(data, aes(x = x, y = y, color = category, size = size_var)) +
    geom_point(alpha = 0.7) +
    geom_smooth(method = "lm", se = FALSE, color = "black", linetype = "dashed") +
    scale_color_brewer(type = "qual", palette = "Set2") +
    scale_size_continuous(range = c(1, 4)) +
    labs(title = "Advanced Scatter Plot with Multiple Aesthetics",
         subtitle = "Color by category, size by continuous variable") +
    theme_minimal()

print(p1)

# Violin plot with statistics
p2 <- data %>%
    ggplot(aes(x = category, y = x)) +
    geom_violin(aes(fill = category), alpha = 0.7) +
    geom_boxplot(width = 0.2, fill = "white", outlier.shape = NA) +
    geom_jitter(width = 0.1, alpha = 0.3, size = 0.8) +
    stat_summary(fun = mean, geom = "point", shape = 23,
                 size = 3, fill = "red", color = "darkred") +
    labs(title = "Distribution Analysis: Violin + Box + Jitter") +
    theme_minimal()

print(p2)

# Base R plots for comparison
plot(data$x, data$y,
     col = rainbow(4)[as.factor(data$category)],
     pch = 19, cex = 0.8,
     main = "Base R: Scatter Plot with Colors",
     xlab = "X Variable", ylab = "Y Variable")
legend("topright", legend = levels(as.factor(data$category)),
       col = rainbow(4), pch = 19, title = "Category")
    """, libraries=["ggplot2", "dplyr", "gridExtra"])
```

### Statistical Analysis

```python
with SandboxSession(lang="r") as session:
    result = session.run("""
# Generate sample data
set.seed(42)
group1 <- rnorm(50, mean = 100, sd = 15)
group2 <- rnorm(50, mean = 110, sd = 12)

# Descriptive statistics
print("Group 1 Statistics:")
print(summary(group1))
print("Group 2 Statistics:")
print(summary(group2))

# Statistical tests
t_test_result <- t.test(group1, group2)
print("T-test Results:")
print(t_test_result)

# Correlation analysis
data <- data.frame(x = group1, y = group1 + rnorm(50, 0, 5))
correlation <- cor.test(data$x, data$y)
print("Correlation Test:")
print(correlation)

# ANOVA
data$group <- rep(c("A", "B"), each = 25)
anova_result <- aov(x ~ group, data = data)
print("ANOVA Results:")
print(summary(anova_result))
    """)
```

### R-Specific Features

#### CRAN Package Management

```python
# Install from specific CRAN mirror
with SandboxSession(lang="r") as session:
    result = session.run("""
# Install from CRAN
install.packages("lubridate", repos = "https://cran.rstudio.com/")
library(lubridate)

# Work with dates
today_date <- today()
print(paste("Today is:", today_date))
print(paste("Year:", year(today_date)))
print(paste("Month:", month(today_date, label = TRUE)))
    """)
```

## JavaScript (Node.js)

### Overview

JavaScript support includes Node.js runtime with npm package management.

### Basic Usage

```python
with SandboxSession(lang="javascript") as session:
    result = session.run("""
console.log(`Node.js ${process.version}`);
console.log('Hello from JavaScript!');

// Modern JavaScript features
const greeting = (name) => `Hello, ${name}!`;
console.log(greeting('World'));

// Async/await support
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

(async () => {
    console.log('Starting...');
    await delay(100);
    console.log('Finished!');
})();
    """)
    print(result.stdout)
```

### Package Management

```python
# Install npm packages
with SandboxSession(lang="javascript") as session:
    result = session.run("""
const axios = require('axios');
const lodash = require('lodash');

console.log('Packages loaded successfully!');

// Use lodash
const numbers = [1, 2, 3, 4, 5];
console.log('Sum:', lodash.sum(numbers));
console.log('Mean:', lodash.mean(numbers));
    """, libraries=["axios", "lodash"])
```

### Working with APIs

```python
with SandboxSession(lang="javascript") as session:
    result = session.run("""
const https = require('https');

// Make API request
https.get('https://api.github.com/users/github', (res) => {
    let data = '';

    res.on('data', (chunk) => {
        data += chunk;
    });

    res.on('end', () => {
        const user = JSON.parse(data);
        console.log('GitHub user:', user.name);
        console.log('Public repos:', user.public_repos);
    });
}).on('error', (err) => {
    console.error('Error:', err.message);
});
    """)
```

### Express.js Server

```python
with SandboxSession(lang="javascript") as session:
    result = session.run("""
const express = require('express');
const app = express();

app.get('/', (req, res) => {
    res.json({ message: 'Hello from Express!' });
});

// Note: Server won't actually be accessible from outside the container
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server would run on port ${PORT}`);
    console.log('(Not accessible from outside the sandbox)');

    // Gracefully exit after setup
    process.exit(0);
});
    """, libraries=["express"])
```

## Java

### Overview

Java support includes JDK 11 with automatic compilation and execution.

### Basic Usage

```python
with SandboxSession(lang="java") as session:
    result = session.run("""
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Java version: " + System.getProperty("java.version"));
        System.out.println("Hello from Java!");

        // Modern Java features
        var message = "Using var keyword!";
        System.out.println(message);
    }
}
    """)
    print(result.stdout)
```

### Object-Oriented Programming

```python
with SandboxSession(lang="java") as session:
    result = session.run("""
import java.util.*;

class Person {
    private String name;
    private int age;

    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }

    public String toString() {
        return String.format("Person{name='%s', age=%d}", name, age);
    }
}

public class Main {
    public static void main(String[] args) {
        List<Person> people = Arrays.asList(
            new Person("Alice", 30),
            new Person("Bob", 25),
            new Person("Charlie", 35)
        );

        System.out.println("People list:");
        people.forEach(System.out::println);

        // Stream API
        double avgAge = people.stream()
            .mapToInt(p -> p.age)
            .average()
            .orElse(0);

        System.out.printf("Average age: %.1f%n", avgAge);
    }
}
    """)
```

### Working with Collections

```python
with SandboxSession(lang="java") as session:
    result = session.run("""
import java.util.*;
import java.util.stream.Collectors;

public class CollectionsDemo {
    public static void main(String[] args) {
        // List operations
        List<Integer> numbers = new ArrayList<>(Arrays.asList(5, 2, 8, 1, 9, 3));
        System.out.println("Original: " + numbers);

        Collections.sort(numbers);
        System.out.println("Sorted: " + numbers);

        // Map operations
        Map<String, Integer> scores = new HashMap<>();
        scores.put("Alice", 95);
        scores.put("Bob", 87);
        scores.put("Charlie", 92);

        System.out.println("\nScores:");
        scores.forEach((name, score) ->
            System.out.printf("%s: %d%n", name, score)
        );

        // Stream operations
        List<Integer> filtered = numbers.stream()
            .filter(n -> n > 5)
            .map(n -> n * 2)
            .collect(Collectors.toList());

        System.out.println("\nFiltered and doubled: " + filtered);
    }
}
    """)
```

## C++

### Overview

C++ support includes GCC compiler with C++17 standard.

### Basic Usage

```python
with SandboxSession(lang="cpp") as session:
    result = session.run("""
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::cout << "C++ Standard: " << __cplusplus << std::endl;
    std::cout << "Hello from C++!" << std::endl;

    // Modern C++ features
    auto message = std::string("Using auto keyword!");
    std::cout << message << std::endl;

    // Range-based for loop
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::cout << "Numbers: ";
    for (const auto& n : numbers) {
        std::cout << n << " ";
    }
    std::cout << std::endl;

    return 0;
}
    """)
    print(result.stdout)
```

### STL and Algorithms

```python
with SandboxSession(lang="cpp") as session:
    result = session.run("""
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <map>

int main() {
    // Vector operations
    std::vector<int> vec = {5, 2, 8, 1, 9, 3};
    std::cout << "Original: ";
    for (int n : vec) std::cout << n << " ";
    std::cout << std::endl;

    // Sort
    std::sort(vec.begin(), vec.end());
    std::cout << "Sorted: ";
    for (int n : vec) std::cout << n << " ";
    std::cout << std::endl;

    // Algorithms
    int sum = std::accumulate(vec.begin(), vec.end(), 0);
    std::cout << "Sum: " << sum << std::endl;

    // Map
    std::map<std::string, int> scores = {
        {"Alice", 95},
        {"Bob", 87},
        {"Charlie", 92}
    };

    std::cout << "\nScores:\n";
    for (const auto& [name, score] : scores) {
        std::cout << name << ": " << score << std::endl;
    }

    return 0;
}
    """)
```

### Installing Libraries

```python
# Install system libraries
with SandboxSession(lang="cpp") as session:
    # Install Boost
    session.run("""
#include <iostream>
#include <boost/algorithm/string.hpp>

int main() {
    std::string text = "Hello, World!";
    boost::to_upper(text);
    std::cout << text << std::endl;
    return 0;
}
    """, libraries=["libboost-all-dev"])
```

## Go

### Overview

Go support includes the Go compiler and module management.

### Basic Usage

```python
with SandboxSession(lang="go") as session:
    result = session.run("""
package main

import (
    "fmt"
    "runtime"
)

func main() {
    fmt.Printf("Go version: %s\n", runtime.Version())
    fmt.Println("Hello from Go!")

    // Go features
    numbers := []int{1, 2, 3, 4, 5}
    sum := 0
    for _, n := range numbers {
        sum += n
    }
    fmt.Printf("Sum: %d\n", sum)
}
    """)
    print(result.stdout)
```

### Concurrency

```python
with SandboxSession(lang="go") as session:
    result = session.run("""
package main

import (
    "fmt"
    "sync"
    "time"
)

func worker(id int, wg *sync.WaitGroup) {
    defer wg.Done()
    fmt.Printf("Worker %d starting\n", id)
    time.Sleep(time.Millisecond * 100)
    fmt.Printf("Worker %d done\n", id)
}

func main() {
    var wg sync.WaitGroup

    for i := 1; i <= 5; i++ {
        wg.Add(1)
        go worker(i, &wg)
    }

    wg.Wait()
    fmt.Println("All workers completed")
}
    """)
```

### Using External Packages

```python
with SandboxSession(lang="go") as session:
    result = session.run("""
package main

import (
    "fmt"
    "github.com/spyzhov/ajson"
)

func main() {
    json := []byte(`{
        "name": "John",
        "age": 30,
        "city": "New York"
    }`)

    root, _ := ajson.Unmarshal(json)

    name, _ := root.GetString("name")
    age, _ := root.GetInt("age")

    fmt.Printf("Name: %s\n", name)
    fmt.Printf("Age: %d\n", age)
}
    """, libraries=["github.com/spyzhov/ajson"])
```

## Language Handler Architecture

### Custom Language Support

You can add support for additional languages:

```python
from llm_sandbox.language_handlers import AbstractLanguageHandler
from llm_sandbox.language_handlers.factory import LanguageHandlerFactory

class RustHandler(AbstractLanguageHandler):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.config = LanguageConfig(
            name="rust",
            file_extension="rs",
            execution_commands=["rustc {file} -o /tmp/program && /tmp/program"],
            package_manager="cargo add",
            is_support_library_installation=True
        )

    def get_import_patterns(self, module):
        return rf"use\s+{module}"

    @staticmethod
    def get_multiline_comment_patterns():
        return r"/\*[\s\S]*?\*/"

    @staticmethod
    def get_inline_comment_patterns():
        return r"//.*$"

# Register the handler
LanguageHandlerFactory.register_handler("rust", RustHandler)

# Use it
with SandboxSession(lang="rust", image="rust:latest") as session:
    result = session.run("""
fn main() {
    println!("Hello from Rust!");
}
    """)
```

### Language Detection

```python
def detect_language(code: str) -> str:
    """Simple language detection based on syntax"""
    patterns = {
        'python': [r'def\s+\w+\s*\(', r'import\s+\w+', r'print\s*\('],
        'javascript': [r'function\s+\w+\s*\(', r'const\s+\w+\s*=', r'console\.log'],
        'java': [r'public\s+class', r'public\s+static\s+void\s+main'],
        'cpp': [r'#include\s*<', r'int\s+main\s*\(', r'std::'],
        'go': [r'package\s+main', r'func\s+main\s*\('],
        'ruby': [r'def\s+\w+', r'puts\s+', r'class\s+\w+'],
    }

    for lang, lang_patterns in patterns.items():
        if any(re.search(pattern, code) for pattern in lang_patterns):
            return lang

    return 'python'  # Default

# Use detected language
code = "def hello():\n    print('Hello')\n"
lang = detect_language(code)

with SandboxSession(lang=lang) as session:
    result = session.run(code)
```

### Optimization Tips

1. **Pre-built Images**
   ```python
   # Build image with pre-installed packages
   FROM python:3.11
   RUN pip install numpy pandas matplotlib scikit-learn
   ```

2. **Keep Templates**
   ```python
   # Reuse containers for faster execution
   with SandboxSession(keep_template=True) as session:
       pass
   ```

3. **Language-Specific Optimizations**
   ```python
   # Python: Use slim images
   image="python:3.11-slim"

   # Java: Use JDK vs JRE based on needs
   image="openjdk:11-jre-slim"  # For running only

   # Go: Use multi-stage builds
   # Build in one stage, run in minimal image
   ```

## Next Steps

- Explore [LLM Integrations](integrations.md)
- Learn about [Security Policies](security.md)
- See practical [Examples](examples.md)
- Read the [API Reference](api-reference.md)
