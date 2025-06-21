# ruff: noqa: T201

import base64
import logging
from pathlib import Path

import docker

from llm_sandbox import ArtifactSandboxSession, SandboxBackend

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set up Docker client
client = docker.DockerClient(base_url="unix:///Users/vndee/.docker/run/docker.sock")

# R code that generates various types of plots
r_code = """
# Load required libraries for comprehensive plotting
library(ggplot2)
library(dplyr)
library(gridExtra)

print("=== R Artifact Demo: Creating Multiple Visualizations ===")

# Generate sample datasets
set.seed(42)
n <- 1000

# Dataset 1: Basic scatter data
scatter_data <- data.frame(
    x = rnorm(n, mean = 50, sd = 15),
    y = rnorm(n, mean = 30, sd = 10),
    category = sample(c("A", "B", "C", "D"), n, replace = TRUE),
    size_var = runif(n, 1, 5)
)

# Dataset 2: Time series data
dates <- seq(as.Date("2023-01-01"), by = "day", length.out = 365)
ts_data <- data.frame(
    date = dates,
    sales = cumsum(rnorm(365, mean = 10, sd = 5)) + 1000,
    profit = cumsum(rnorm(365, mean = 5, sd = 3)) + 500,
    expenses = cumsum(rnorm(365, mean = 8, sd = 4)) + 800
)

# Dataset 3: Categorical data for bar plots
category_data <- data.frame(
    department = c("Sales", "Marketing", "Engineering", "HR", "Finance"),
    employees = c(45, 23, 67, 12, 18),
    budget = c(450000, 230000, 890000, 120000, 180000)
)

print("Generated datasets successfully!")

# Plot 1: Advanced ggplot2 scatter plot with multiple aesthetics
p1 <- ggplot(scatter_data, aes(x = x, y = y, color = category, size = size_var)) +
    geom_point(alpha = 0.7) +
    geom_smooth(method = "lm", se = FALSE, color = "black", linetype = "dashed") +
    scale_color_brewer(type = "qual", palette = "Set2") +
    scale_size_continuous(range = c(1, 4)) +
    labs(
        title = "Advanced Scatter Plot with Multiple Aesthetics",
        subtitle = "Color by category, size by continuous variable",
        x = "X Variable",
        y = "Y Variable",
        color = "Category",
        size = "Size Variable"
    ) +
    theme_minimal() +
    theme(
        plot.title = element_text(size = 14, face = "bold"),
        plot.subtitle = element_text(size = 12, color = "gray60"),
        legend.position = "bottom"
    )

print(p1)

# Plot 2: Time series with multiple lines
p2 <- ts_data %>%
    tidyr::pivot_longer(cols = c(sales, profit, expenses),
                        names_to = "metric",
                        values_to = "value") %>%
    ggplot(aes(x = date, y = value, color = metric)) +
    geom_line(linewidth = 1.2, alpha = 0.8) +
    scale_color_manual(values = c("sales" = "#2E86AB", "profit" = "#A23B72", "expenses" = "#F18F01")) +
    labs(
        title = "Financial Metrics Over Time",
        subtitle = "Daily tracking of key business indicators",
        x = "Date",
        y = "Amount ($)",
        color = "Metric"
    ) +
    theme_minimal() +
    theme(
        plot.title = element_text(size = 14, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "top"
    ) +
    scale_y_continuous(labels = scales::comma)

print(p2)

# Plot 3: Sophisticated bar chart with dual y-axis concept
p3 <- ggplot(category_data, aes(x = reorder(department, employees))) +
    geom_col(aes(y = employees), fill = "#3498db", alpha = 0.8, width = 0.7) +
    geom_text(aes(y = employees, label = employees),
              vjust = -0.5, size = 4, fontface = "bold") +
    labs(
        title = "Employee Distribution by Department",
        subtitle = "Total headcount across organizational units",
        x = "Department",
        y = "Number of Employees"
    ) +
    theme_minimal() +
    theme(
        plot.title = element_text(size = 14, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid.major.x = element_blank()
    )

print(p3)

# Plot 4: Faceted plot showing distributions
p4 <- scatter_data %>%
    ggplot(aes(x = x, fill = category)) +
    geom_histogram(bins = 30, alpha = 0.7, color = "white") +
    facet_wrap(~category, scales = "free_y") +
    scale_fill_brewer(type = "qual", palette = "Set3") +
    labs(
        title = "Distribution of X Variable by Category",
        subtitle = "Faceted histograms showing different patterns",
        x = "X Variable",
        y = "Frequency"
    ) +
    theme_minimal() +
    theme(
        plot.title = element_text(size = 14, face = "bold"),
        strip.text = element_text(face = "bold"),
        legend.position = "none"
    )

print(p4)

# Plot 5: Base R plots for comparison
# High-quality base R scatter plot
par(bg = "white", mar = c(5, 5, 4, 2))
plot(scatter_data$x, scatter_data$y,
     col = rainbow(4)[as.factor(scatter_data$category)],
     pch = 19, cex = 0.8,
     main = "Base R: Scatter Plot with Colors",
     xlab = "X Variable", ylab = "Y Variable",
     cex.main = 1.3, cex.lab = 1.1)
legend("topright", legend = levels(as.factor(scatter_data$category)),
       col = rainbow(4), pch = 19, title = "Category")
grid(col = "gray90", lty = 1)

# Base R time series plot
par(bg = "white", mar = c(5, 5, 4, 2))
plot(ts_data$date, ts_data$sales, type = "l", col = "#2E86AB", lwd = 2,
     main = "Base R: Financial Time Series",
     xlab = "Date", ylab = "Amount ($)",
     cex.main = 1.3, cex.lab = 1.1,
     ylim = c(min(ts_data$sales, ts_data$profit, ts_data$expenses),
              max(ts_data$sales, ts_data$profit, ts_data$expenses)))
lines(ts_data$date, ts_data$profit, col = "#A23B72", lwd = 2)
lines(ts_data$date, ts_data$expenses, col = "#F18F01", lwd = 2)
legend("topleft", legend = c("Sales", "Profit", "Expenses"),
       col = c("#2E86AB", "#A23B72", "#F18F01"), lwd = 2)
grid(col = "gray90", lty = 1)

# Plot 6: Advanced statistical visualization
p5 <- scatter_data %>%
    ggplot(aes(x = category, y = x)) +
    geom_violin(aes(fill = category), alpha = 0.7) +
    geom_boxplot(width = 0.2, fill = "white", outlier.shape = NA) +
    geom_jitter(width = 0.1, alpha = 0.3, size = 0.8) +
    scale_fill_brewer(type = "qual", palette = "Pastel1") +
    labs(
        title = "Distribution Analysis: Violin + Box + Jitter Plot",
        subtitle = "Comprehensive view of data distribution across categories",
        x = "Category",
        y = "X Variable"
    ) +
    theme_minimal() +
    theme(
        plot.title = element_text(size = 14, face = "bold"),
        legend.position = "none"
    ) +
    stat_summary(fun = mean, geom = "point", shape = 23,
                 size = 3, fill = "red", color = "darkred")

print(p5)

print("=== All visualizations completed successfully! ===")
print(paste("Total plots generated:", length(list.files("/tmp/sandbox_plots", pattern = "\\.(png|svg|pdf)$"))))
"""

# Create plots directory
Path("plots/r_artifacts").mkdir(parents=True, exist_ok=True)

print("🎨 Starting R Artifact Demo with ArtifactSandboxSession...")

# Run the R artifact demo
with ArtifactSandboxSession(
    client=client,
    lang="r",
    image="ghcr.io/vndee/sandbox-r-451-bullseye",
    verbose=True,
    backend=SandboxBackend.DOCKER,
    enable_plotting=True,
) as session:
    print("📊 Executing R code with comprehensive plotting...")

    # Run the R code with required libraries
    result = session.run(r_code, libraries=["ggplot2", "dplyr", "gridExtra", "tidyr", "scales"])

    print("\n✅ Execution completed!")
    print(f"📈 Captured {len(result.plots)} plots")
    print("📝 Console output:")
    print(result.stdout)

    if result.stderr:
        print("⚠️  Warnings/Messages:")
        print(result.stderr)

    # Save all captured plots
    for i, plot in enumerate(result.plots):
        plot_path = Path("plots/r_artifacts") / f"r_plot_{i + 1:02d}.{plot.format.value}"
        with plot_path.open("wb") as f:
            f.write(base64.b64decode(plot.content_base64))

        print(f"💾 Saved plot {i + 1}: {plot_path}")
        print(f"   📏 Size: {len(plot.content_base64)} bytes (base64)")
        print(f"   🎨 Format: {plot.format.value}")

print(f"\n🎉 Demo completed! Check the 'plots/r_artifacts' directory for {len(result.plots)} generated plots.")
print("📁 Generated plots showcase:")
print("   • Advanced ggplot2 scatter plots with multiple aesthetics")
print("   • Time series visualizations with multiple metrics")
print("   • Professional bar charts with annotations")
print("   • Faceted histograms showing distributions")
print("   • Base R plots for comparison")
print("   • Statistical visualizations (violin + box + jitter plots)")
