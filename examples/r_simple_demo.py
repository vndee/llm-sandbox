import logging

import docker

from llm_sandbox import SandboxSession

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
client = docker.DockerClient(base_url="unix:///Users/vndee/.docker/run/docker.sock")


# Simple R example without plotting
with SandboxSession(
    client=client,
    lang="r",
    image="ghcr.io/vndee/sandbox-r-451-bullseye",
    verbose=True,
) as session:
    result = session.run(
        """
# Basic R operations
print("=== Basic R Demo ===")

# Create some data
numbers <- c(1, 2, 3, 4, 5, 10, 15, 20)
print(paste("Numbers:", paste(numbers, collapse=", ")))

# Basic statistics
print(paste("Mean:", mean(numbers)))
print(paste("Median:", median(numbers)))
print(paste("Standard Deviation:", sd(numbers)))

# Work with data frames
df <- data.frame(
    name = c("Alice", "Bob", "Charlie", "Diana"),
    age = c(25, 30, 35, 28),
    score = c(85, 92, 78, 96)
)

print("=== Data Frame ===")
print(df)

# Calculate average score
avg_score <- mean(df$score)
print(paste("Average Score:", avg_score))
        """
    )

    logger.info(result)

# Example with library installation and usage
with SandboxSession(
    client=client,
    lang="r",
    image="ghcr.io/vndee/sandbox-r-451-bullseye",
    verbose=True,
) as session:
    result = session.run(
        """
# Using dplyr for data manipulation
library(dplyr)

# Create sample data
employees <- data.frame(
    name = c("John", "Jane", "Bob", "Alice", "Charlie"),
    department = c("IT", "HR", "IT", "Finance", "HR"),
    salary = c(75000, 65000, 80000, 70000, 60000),
    experience = c(5, 3, 7, 4, 2)
)

print("=== Employee Data ===")
print(employees)

# Use dplyr to analyze data
it_employees <- employees %>%
    filter(department == "IT") %>%
    arrange(desc(salary))

print("=== IT Employees (sorted by salary) ===")
print(it_employees)

# Calculate average salary by department
avg_by_dept <- employees %>%
    group_by(department) %>%
    summarise(
        avg_salary = mean(salary),
        avg_experience = mean(experience),
        count = n()
    )

print("=== Average by Department ===")
print(avg_by_dept)
        """,
        libraries=["dplyr"],
    )

    logger.info(result)
