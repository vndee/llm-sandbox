# ruff: noqa: E501

"""Simple R Computing Demo.

This script demonstrates basic R programming capabilities, including data analysis, machine learning, and visualization.
It uses the tidyverse and data.table packages for data manipulation and analysis.
"""

import logging

import docker

from llm_sandbox import SandboxSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = docker.DockerClient(base_url="unix:///Users/vndee/.docker/run/docker.sock")

# Simple R Computing Demo - focused on core data science computation
with SandboxSession(
    client=client,
    lang="r",
    image="ghcr.io/vndee/sandbox-r-451-bullseye",
    verbose=True,
) as session:
    # Core Data Analysis Demo
    result = session.run(
        """
library(tidyverse)
library(data.table)

print("=== R Core Computing Demo ===")

# Generate synthetic dataset for analysis
set.seed(42)
n <- 1000

# Create comprehensive dataset
data <- tibble(
    id = 1:n,
    age = rnorm(n, mean = 35, sd = 10),
    income = exp(rnorm(n, mean = 10.5, sd = 0.5)),
    education = sample(c("High School", "Bachelor", "Master", "PhD"), n, replace = TRUE),
    experience = pmax(0, age - rnorm(n, mean = 22, sd = 3)),
    performance_score = 70 + 0.3 * experience + 0.0001 * income + rnorm(n, mean = 0, sd = 5),
    department = sample(c("IT", "HR", "Finance", "Marketing", "Operations"), n, replace = TRUE)
)

print("=== Data Summary ===")
print(glimpse(data))

# Advanced data manipulation with dplyr
summary_stats <- data %>%
    group_by(department, education) %>%
    summarise(
        count = n(),
        avg_age = mean(age, na.rm = TRUE),
        avg_income = mean(income, na.rm = TRUE),
        avg_performance = mean(performance_score, na.rm = TRUE),
        .groups = 'drop'
    ) %>%
    arrange(desc(avg_performance))

print("=== Performance Analysis ===")
print(summary_stats)

# High-performance data.table operations
dt <- as.data.table(data)

# Fast aggregations with data.table
dt_summary <- dt[, .(
    avg_income = mean(income),
    median_performance = median(performance_score),
    count = .N
), by = .(department, education)]

print("=== Data.table Fast Aggregation ===")
print(dt_summary)

# Statistical computations
correlation_matrix <- data %>%
    select(age, income, experience, performance_score) %>%
    cor()

print("=== Correlation Matrix ===")
print(round(correlation_matrix, 3))
        """,
        libraries=["tidyverse", "data.table"],
    )

    logger.info(result)

# Machine Learning Demo
with SandboxSession(
    client=client,
    lang="r",
    image="ghcr.io/vndee/sandbox-r-451-bullseye",
    verbose=True,
) as session:
    result = session.run(
        """
library(caret)
library(randomForest)
library(broom)

print("=== Machine Learning Demo ===")

# Create dataset
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
lm_summary <- tidy(lm_model)
print("=== Linear Regression Coefficients ===")
print(lm_summary)

# Random Forest
rf_model <- randomForest(y ~ ., data = train_data, ntree = 100)
print("=== Random Forest Importance ===")
print(importance(rf_model))

# Make predictions
lm_pred <- predict(lm_model, test_data)
rf_pred <- predict(rf_model, test_data)

# Calculate RMSE
lm_rmse <- sqrt(mean((test_data$y - lm_pred)^2))
rf_rmse <- sqrt(mean((test_data$y - rf_pred)^2))

print("=== Model Performance (RMSE) ===")
print(paste("Linear Regression RMSE:", round(lm_rmse, 4)))
print(paste("Random Forest RMSE:", round(rf_rmse, 4)))
        """,
        libraries=["caret", "randomForest", "broom"],
    )

    logger.info(result)

# Time Series Analysis Demo
with SandboxSession(
    client=client,
    lang="r",
    image="ghcr.io/vndee/sandbox-r-451-bullseye",
    verbose=True,
) as session:
    result = session.run(
        """
library(forecast)
library(zoo)

print("=== Time Series Analysis Demo ===")

# Create synthetic time series
set.seed(42)
n <- 200
trend <- seq(100, 150, length.out = n)
seasonal <- 10 * sin(2 * pi * (1:n) / 12)
noise <- rnorm(n, 0, 3)
values <- trend + seasonal + noise

# Convert to time series
ts_data <- ts(values, frequency = 12, start = c(2020, 1))

print("=== Time Series Summary ===")
print(summary(ts_data))

# Decompose time series
decomp <- decompose(ts_data)
print("=== Seasonal Decomposition ===")
print(paste("Trend Range:", round(min(decomp$trend, na.rm = TRUE), 2), "to", round(max(decomp$trend, na.rm = TRUE), 2)))
print(paste("Seasonal Range:", round(min(decomp$seasonal, na.rm = TRUE), 2), "to", round(max(decomp$seasonal, na.rm = TRUE), 2)))

# Fit ARIMA model
arima_model <- auto.arima(ts_data)
print("=== ARIMA Model ===")
print(arima_model)

# Forecast
forecast_result <- forecast(arima_model, h = 12)
print("=== 12-Period Forecast ===")
print(as.data.frame(forecast_result))
        """,
        libraries=["forecast", "zoo"],
    )

    logger.info(result)
