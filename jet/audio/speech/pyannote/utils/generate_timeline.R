# jet/audio/speech/pyannote/utils/generate_timeline.R
#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(plotly)
  library(htmlwidgets)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("Usage: generate_timeline.R <turns_json> <total_sec> <html_out> <audio_name> <libdir_path>")
}

turns_json_path <- args[1]
total_sec       <- as.numeric(args[2])
html_path       <- args[3]
audio_name      <- args[4]
libdir_path     <- args[5]   # ← can be relative or absolute

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
if (!file.exists(turns_json_path)) stop("turns.json not found: ", turns_json_path)
turns <- read_json(turns_json_path, simplifyVector = FALSE)

df <- data.frame(
  speaker    = sapply(turns, function(x) x$speaker %||% "UNKNOWN"),
  start      = sapply(turns, `[[`, "start_sec"),
  end        = sapply(turns, `[[`, "end_sec"),
  duration   = sapply(turns, `[[`, "duration_sec"),
  confidence = sapply(turns, function(x) x$confidence %||% 0.0),
  stringsAsFactors = FALSE
)

df$text <- with(df, sprintf(
  "<b>%s</b><br>Time: %.3f → %.3f s<br>Duration: %.3f s<br><b>Confidence:</b> %.4f",
  speaker, start, end, duration, confidence
))

speakers   <- sort(unique(df$speaker))
n_speakers <- length(speakers)

# Professional palette
colors <- c("#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f",
            "#bcbd22","#17becf","#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5","#c49c94",
            "#f7b6d2","#c7c7c7","#dbdb8d","#9edae5")
color_map <- setNames(colors[1:n_speakers], speakers)

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
p <- plot_ly(df, height = 340 + n_speakers * 62) %>%
  add_segments(
    x = ~start, xend = ~end,
    y = ~speaker, yend = ~speaker,
    line = list(width = 38, color = ~speaker),
    color = ~speaker,
    colors = color_map,
    text = ~text,
    hovertemplate = "%{text}<extra></extra>",
    name = ~speaker,
    showlegend = TRUE
  ) %>%
  layout(
    title = list(text = paste0("Speaker Diarization Timeline — ", audio_name),
                 font = list(size = 22)),
    xaxis = list(title = "Time (seconds)",
                 range = c(0, total_sec * 1.05),
                 showgrid = TRUE, gridcolor = "lightgray"),
    yaxis = list(title = "Speaker",
                 categoryorder = "array",
                 categoryarray = rev(speakers)),
    plot_bgcolor  = "white",
    paper_bgcolor = "white",
    hovermode = "closest",
    legend = list(title = list(text = "Speakers")),
    margin = list(l = 120, r = 60, t = 100, b = 60)
  )

# ------------------------------------------------------------------
# Save as normal HTML + assets (no pandoc needed)
# ------------------------------------------------------------------
saveWidget(p, html_path, selfcontained = FALSE, libdir = basename(libdir_path))

cat("Success: Timeline saved to", html_path, "\n")
cat("Plotly assets saved to ./plotly_assets/\n")