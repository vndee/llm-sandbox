# ruff: noqa: E501

R_PLOT_DETECTION_CODE = """
# R Multi-library plot detection setup

# Setup output directories
dir.create('/tmp/sandbox_plots', recursive = TRUE, showWarnings = FALSE)
dir.create('/tmp/sandbox_output', recursive = TRUE, showWarnings = FALSE)

# Global plot counter
.plot_counter <- 0

# === BASE R GRAPHICS SUPPORT ===
# Override the graphics device functions to auto-save plots

# Store original functions
.original_dev_off <- dev.off
.original_plot_new <- plot.new

# Enhanced plot.new to start capturing
plot.new <- function(...) {
  result <- .original_plot_new(...)

  # Start a PNG device in the background to capture the plot
  .plot_counter <<- .plot_counter + 1
  png_file <- sprintf('/tmp/sandbox_plots/%06d.png', .plot_counter)

  # Open PNG device
  png(png_file, width = 800, height = 600, res = 100)

  return(result)
}

# Enhanced dev.off to finalize plot capture
dev.off <- function(...) {
  # Close any open devices
  while(dev.cur() > 1) {
    .original_dev_off()
  }

  return(invisible())
}

# Override main plotting functions to ensure capture
.enhanced_plot <- function(...) {
  .plot_counter <<- .plot_counter + 1
  png_file <- sprintf('/tmp/sandbox_plots/%06d.png', .plot_counter)

  # Save to file
  png(png_file, width = 800, height = 600, res = 100)
  on.exit(dev.off())

  # Call original plot (use stored original if available, otherwise graphics::plot)
  if (exists(".original_plot") && is.function(.original_plot)) {
    .original_plot(...)
  } else {
    graphics::plot(...)
  }
}

# Patch common plotting functions
if (exists("plot", envir = .GlobalEnv)) {
  .original_plot <- plot
}
plot <- .enhanced_plot

# Enhanced hist function
.enhanced_hist <- function(...) {
  .plot_counter <<- .plot_counter + 1
  png_file <- sprintf('/tmp/sandbox_plots/%06d.png', .plot_counter)

  png(png_file, width = 800, height = 600, res = 100)
  on.exit(dev.off())

  graphics::hist(...)
}

hist <- .enhanced_hist

# Enhanced boxplot function
.enhanced_boxplot <- function(...) {
  .plot_counter <<- .plot_counter + 1
  png_file <- sprintf('/tmp/sandbox_plots/%06d.png', .plot_counter)

  png(png_file, width = 800, height = 600, res = 100)
  on.exit(dev.off())

  graphics::boxplot(...)
}

boxplot <- .enhanced_boxplot

# Enhanced barplot function
.enhanced_barplot <- function(...) {
  .plot_counter <<- .plot_counter + 1
  png_file <- sprintf('/tmp/sandbox_plots/%06d.png', .plot_counter)

  png(png_file, width = 800, height = 600, res = 100)
  on.exit(dev.off())

  graphics::barplot(...)
}

barplot <- .enhanced_barplot

# === GGPLOT2 SUPPORT ===
if (requireNamespace("ggplot2", quietly = TRUE)) {

  # Enhanced print method for ggplot objects
  .original_print_ggplot <- getS3method("print", "ggplot")

  print.ggplot <- function(x, ...) {
    # Call original print method
    result <- .original_print_ggplot(x, ...)

    # Save the plot
    tryCatch({
      .plot_counter <<- .plot_counter + 1
      png_file <- sprintf('/tmp/sandbox_plots/%06d.png', .plot_counter)
      ggplot2::ggsave(png_file, plot = x, width = 10, height = 6, dpi = 100)
    }, error = function(e) {
      cat("ggplot2 capture error:", e$message, "\n")
    })

    return(result)
  }

  # Enhanced ggsave function
  if (exists("ggsave", envir = asNamespace("ggplot2"))) {
    .original_ggsave <- ggplot2::ggsave

    assignInNamespace("ggsave", function(filename, plot = ggplot2::last_plot(), ...) {
      # Call original ggsave
      result <- .original_ggsave(filename, plot, ...)

      # Copy to our output directory
      tryCatch({
        .plot_counter <<- .plot_counter + 1
        ext <- tools::file_ext(filename)
        if (ext == "") ext <- "png"
        output_file <- sprintf('/tmp/sandbox_plots/%06d.%s', .plot_counter, ext)
        file.copy(filename, output_file, overwrite = TRUE)
      }, error = function(e) {
        cat("ggplot2 ggsave capture error:", e$message, "\n")
      })

      return(result)
    }, ns = "ggplot2")
  }

  cat("ggplot2 plotting enabled\n")
}

# === PLOTLY SUPPORT ===
if (requireNamespace("plotly", quietly = TRUE)) {

  # Enhanced print method for plotly objects
  if (exists("print.plotly", envir = asNamespace("plotly"))) {
    .original_print_plotly <- getS3method("print", "plotly")

    print.plotly <- function(x, ...) {
      # Call original print method
      result <- .original_print_plotly(x, ...)

      # Save as HTML
      tryCatch({
        .plot_counter <<- .plot_counter + 1
        html_file <- sprintf('/tmp/sandbox_plots/%06d.html', .plot_counter)
        htmlwidgets::saveWidget(x, html_file, selfcontained = TRUE)
      }, error = function(e) {
        cat("plotly capture error:", e$message, "\n")
      })

      return(result)
    }
  }

  cat("plotly plotting enabled\n")
}

# === LATTICE SUPPORT ===
if (requireNamespace("lattice", quietly = TRUE)) {

  # Enhanced print method for lattice plots
  if (exists("print.trellis", envir = asNamespace("lattice"))) {
    .original_print_trellis <- getS3method("print", "trellis")

    print.trellis <- function(x, ...) {
      # Call original print method
      result <- .original_print_trellis(x, ...)

      # Save the plot
      tryCatch({
        .plot_counter <<- .plot_counter + 1
        png_file <- sprintf('/tmp/sandbox_plots/%06d.png', .plot_counter)
        png(png_file, width = 800, height = 600, res = 100)
        .original_print_trellis(x, ...)
        dev.off()
      }, error = function(e) {
        cat("lattice capture error:", e$message, "\n")
      })

      return(result)
    }
  }

  cat("lattice plotting enabled\n")
}

# === GENERAL PLOT CAPTURE FUNCTIONS ===

# Function to manually save current plot
save_current_plot <- function(format = "png") {
  .plot_counter <<- .plot_counter + 1

  if (format == "png") {
    filename <- sprintf('/tmp/sandbox_plots/%06d.png', .plot_counter)
    dev.copy(png, filename, width = 800, height = 600, res = 100)
    dev.off()
  } else if (format == "pdf") {
    filename <- sprintf('/tmp/sandbox_plots/%06d.pdf', .plot_counter)
    dev.copy(pdf, filename, width = 10, height = 6)
    dev.off()
  } else if (format == "svg") {
    filename <- sprintf('/tmp/sandbox_plots/%06d.svg', .plot_counter)
    dev.copy(svg, filename, width = 10, height = 6)
    dev.off()
  }

  cat("Plot saved to:", filename, "\n")
  return(filename)
}

# Function to get plot count
get_plot_count <- function() {
  return(.plot_counter)
}

cat("R plot detection setup complete\n")
"""
