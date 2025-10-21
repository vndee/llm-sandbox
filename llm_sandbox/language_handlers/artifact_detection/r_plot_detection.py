# ruff: noqa: E501

R_PLOT_DETECTION_CODE = """
# R Multi-library plot detection setup

# Setup output directories
dir.create('/tmp/sandbox_plots', recursive = TRUE, showWarnings = FALSE)
dir.create('/tmp/sandbox_output', recursive = TRUE, showWarnings = FALSE)

# Global plot counter with file-based persistence
.counter_file <- '/tmp/sandbox_plots/.counter'

# Helper to read counter from file
.read_counter <- function() {
  if (file.exists(.counter_file)) {
    tryCatch({
      as.integer(readLines(.counter_file, n = 1, warn = FALSE))
    }, error = function(e) {
      0L
    })
  } else {
    0L
  }
}

# Helper to save counter to file
.save_counter <- function() {
  writeLines(as.character(.plot_counter), .counter_file)
}

# Initialize counter from file
.plot_counter <- .read_counter()

# === BASE R GRAPHICS SUPPORT ===
# Override the graphics device functions to auto-save plots

# Store original functions
.original_dev_off <- dev.off
.original_plot_new <- plot.new

# Enhanced plot.new to start capturing
# Note: We no longer override plot.new as it interferes with legend() and other graphics functions
# Instead, we rely on enhanced plotting functions below to capture plots

# Enhanced dev.off to finalize plot capture
dev.off <- function(which = dev.cur(), ...) {
  # If closing our tracked device, clear the tracker
  if (exists(".current_plot_device") && which == .current_plot_device) {
    .current_plot_device <<- NULL
  }

  # Close the device
  if (which > 1) {
    .original_dev_off(which)
  }

  return(invisible())
}

# Function to ensure all plot devices are closed at the end
.finalize_plots <- function(env) {
  # Close any remaining plot device
  if (exists(".current_plot_device", envir = .GlobalEnv) && !is.null(.GlobalEnv$.current_plot_device) && .GlobalEnv$.current_plot_device > 1) {
    tryCatch({
      dev.off(.GlobalEnv$.current_plot_device)
    }, error = function(e) {
      # Device may already be closed
    })
    .GlobalEnv$.current_plot_device <- NULL
  }

  # Close all remaining devices
  while(dev.cur() > 1) {
    tryCatch({
      .original_dev_off()
    }, error = function(e) {
      break
    })
  }
}

# Register the finalizer to run when the script exits
reg.finalizer(.GlobalEnv, .finalize_plots, onexit = TRUE)

# Override main plotting functions to ensure capture
.enhanced_plot <- function(...) {
  # Close any existing plot device before starting a new one
  if (exists(".current_plot_device") && !is.null(.current_plot_device) && .current_plot_device > 1) {
    dev.off(.current_plot_device)
    .current_plot_device <<- NULL
  }

  .plot_counter <<- .plot_counter + 1
  .save_counter()
  png_file <- sprintf('/tmp/sandbox_plots/%06d.png', .plot_counter)

  # Save to file - keep device open for subsequent additions (legend, grid, etc.)
  .current_plot_device <<- png(png_file, width = 800, height = 600, res = 100)

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
  # Close any existing plot device before starting a new one
  if (exists(".current_plot_device") && !is.null(.current_plot_device) && .current_plot_device > 1) {
    dev.off(.current_plot_device)
    .current_plot_device <<- NULL
  }

  .plot_counter <<- .plot_counter + 1
  .save_counter()
  png_file <- sprintf('/tmp/sandbox_plots/%06d.png', .plot_counter)

  .current_plot_device <<- png(png_file, width = 800, height = 600, res = 100)

  graphics::hist(...)
}

hist <- .enhanced_hist

# Enhanced boxplot function
.enhanced_boxplot <- function(...) {
  # Close any existing plot device before starting a new one
  if (exists(".current_plot_device") && !is.null(.current_plot_device) && .current_plot_device > 1) {
    dev.off(.current_plot_device)
    .current_plot_device <<- NULL
  }

  .plot_counter <<- .plot_counter + 1
  .save_counter()
  png_file <- sprintf('/tmp/sandbox_plots/%06d.png', .plot_counter)

  .current_plot_device <<- png(png_file, width = 800, height = 600, res = 100)

  graphics::boxplot(...)
}

boxplot <- .enhanced_boxplot

# Enhanced barplot function
.enhanced_barplot <- function(...) {
  # Close any existing plot device before starting a new one
  if (exists(".current_plot_device") && !is.null(.current_plot_device) && .current_plot_device > 1) {
    dev.off(.current_plot_device)
    .current_plot_device <<- NULL
  }

  .plot_counter <<- .plot_counter + 1
  .save_counter()
  png_file <- sprintf('/tmp/sandbox_plots/%06d.png', .plot_counter)

  .current_plot_device <<- png(png_file, width = 800, height = 600, res = 100)

  graphics::barplot(...)
}

barplot <- .enhanced_barplot

# === GGPLOT2 SUPPORT ===
# Function to set up ggplot2 hooks after package is loaded
.setup_ggplot2_hooks <- function() {
  if (!isNamespaceLoaded("ggplot2")) {
    return(FALSE)
  }

  tryCatch({
    # Get ggplot2's S3 methods for printing - try common class names
    .ggplot_classes <- c("ggplot", "gg")
    .original_print_ggplot <- NULL

    for (class_name in .ggplot_classes) {
      .original_print_ggplot <- tryCatch({
        utils::getS3method("print", class_name, envir = asNamespace("ggplot2"))
      }, error = function(e) NULL)

      if (!is.null(.original_print_ggplot)) {
        cat("Found ggplot2 S3 method for class:", class_name, "\n")
        break
      }
    }

    # If we couldn't find it, use a fallback that calls ggplot2's rendering
    if (is.null(.original_print_ggplot)) {
      cat("Using fallback ggplot2 print method\n")
      .original_print_ggplot <- function(x, newpage = TRUE, vp = NULL, ...) {
        if (newpage) grid::grid.newpage()
        grDevices::recordGraphics(
          requireNamespace("ggplot2", quietly = TRUE),
          list(),
          getNamespace("ggplot2")
        )
        ggplot2::ggplot_build(x)
        ggplot2::ggplot_gtable(ggplot2::ggplot_build(x))
        grid::grid.draw(ggplot2::ggplotGrob(x))
        invisible(x)
      }
    }

    # Store in global environment
    .GlobalEnv$.original_print_ggplot <- .original_print_ggplot

    # Create enhanced print function
    print.ggplot <- function(x, newpage = is.null(vp), vp = NULL, ...) {
      # Close any existing base R plot device first
      if (exists(".current_plot_device", envir = .GlobalEnv) && !is.null(.GlobalEnv$.current_plot_device) && .GlobalEnv$.current_plot_device > 1) {
        tryCatch({
          dev.off(.GlobalEnv$.current_plot_device)
        }, error = function(e) {})
        .GlobalEnv$.current_plot_device <- NULL
      }

      # Save the plot
      tryCatch({
        .plot_counter <<- .plot_counter + 1
        .save_counter()
        png_file <- sprintf('/tmp/sandbox_plots/%06d.png', .plot_counter)
        ggplot2::ggsave(png_file, plot = x, width = 10, height = 6, dpi = 100)
      }, error = function(e) {
        cat("ggplot2 capture error:", e$message, "\n")
      })

      # Call original print method
      .GlobalEnv$.original_print_ggplot(x, newpage = newpage, vp = vp, ...)
    }

    # Register as S3 method in global environment - this will take precedence
    .GlobalEnv$print.ggplot <- print.ggplot

    # Enhanced ggsave function
    if (exists("ggsave", envir = asNamespace("ggplot2"))) {
      .original_ggsave <- ggplot2::ggsave

      assignInNamespace("ggsave", function(filename, plot = ggplot2::last_plot(), ...) {
        # Call original ggsave
        result <- .original_ggsave(filename, plot, ...)

        # Copy to our output directory
        tryCatch({
          .plot_counter <<- .plot_counter + 1
          .save_counter()
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
    return(TRUE)
  }, error = function(e) {
    cat("Error setting up ggplot2 hooks:", e$message, "\n")
    return(FALSE)
  })
}

# Hook into library/require to setup ggplot2 when it's loaded
.original_library <- base::library
.original_require <- base::require

library <- function(...) {
  result <- .original_library(...)
  # Check if ggplot2 was just loaded
  if (isNamespaceLoaded("ggplot2") && !exists(".ggplot2_hooks_setup", envir = .GlobalEnv)) {
    if (.setup_ggplot2_hooks()) {
      .ggplot2_hooks_setup <<- TRUE
    }
  }
  return(result)
}

require <- function(...) {
  result <- .original_require(...)
  # Check if ggplot2 was just loaded
  if (isNamespaceLoaded("ggplot2") && !exists(".ggplot2_hooks_setup", envir = .GlobalEnv)) {
    if (.setup_ggplot2_hooks()) {
      .ggplot2_hooks_setup <<- TRUE
    }
  }
  return(result)
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
        .save_counter()
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
        .save_counter()
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
  .save_counter()

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
