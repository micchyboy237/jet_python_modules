# Install required packages if not already installed
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2")
}
if (!requireNamespace("scales", quietly = TRUE)) {
  install.packages("scales")
}
if (!requireNamespace("stringr", quietly = TRUE)) {
  install.packages("stringr")
}

library(ggplot2)
library(scales)
library(stringr)

create_topic_bar_chart <- function(csv_path, output_path) {
  # Read the CSV file
  if (!file.exists(csv_path)) {
    stop("CSV file does not exist: ", csv_path)
  }
  
  data <- read.csv(csv_path, stringsAsFactors = FALSE)
  
  # Debug: Log CSV content
  cat("CSV content:\n")
  print(data)
  
  # Validate data
  if (!all(c("Topic", "Count") %in% colnames(data))) {
    stop("CSV must contain 'Topic' and 'Count' columns")
  }
  if (nrow(data) == 0) {
    stop("CSV file is empty")
  }
  if (nrow(data) == 1) {
    warning("Only one topic found; plot will show a single bar")
  }
  
  # Clean and shorten topic names
  data$Topic <- str_trunc(data$Topic, 20, "right")
  data$Topic <- str_replace_all(data$Topic, "[^[:alnum:]]", "_")
  
  # Define colors for bars
  colors <- c("steelblue", "darkorange", "purple", "darkred", "forestgreen")
  data$Color <- colors[seq_len(nrow(data)) %% length(colors) + 1]
  
  # Create bar chart
  p <- ggplot(data, aes(x = reorder(Topic, -Count), y = Count, fill = Color)) +
    geom_bar(stat = "identity") +
    scale_fill_identity() +
    theme_minimal() +
    labs(title = "Document Distribution by Topic",
         x = "Topic",
         y = "Number of Documents") +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 12, face = "bold"),
      axis.text.y = element_text(size = 12),
      axis.title = element_text(size = 14),
      plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
      panel.grid.major.x = element_blank()
    ) +
    scale_y_continuous(breaks = pretty_breaks(), expand = c(0, 0.5))
  
  # Save plot with higher resolution
  ggsave(output_path, plot = p, width = 10, height = 6, dpi = 600, bg = "white")
  cat("Plot saved to ", output_path, "\n")
}