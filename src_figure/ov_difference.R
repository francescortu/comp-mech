library(ggplot2)
library(dplyr)
library(tidyr)
library(ggpubr)


# Accept command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check if a file name is provided
if (length(args) == 0) {
  stop("No file name provided. Usage: Rscript script_name.R <filename>")
}
folder_name <- args[1]

data <- read.csv(paste(folder_name, "ov_difference_data.csv", sep = "/"))
data <- read.csv("ov_difference_data.csv")
# Select 8 layer heads
selected_combinations <- data.frame(layer = c(10, 11, 10, 10, 9, 9, 11, 11),
                                    head = c(0, 10, 7, 1, 10, 6, 9, 1))


# Filter the data to only include the selected layer heads
data <- data %>%
  mutate(combination = paste("Layer", layer, "Head", head)) %>%
  filter(combination %in% paste("Layer", selected_combinations$layer, "Head", selected_combinations$head))

# create a new column for subtitle
data$subtitle <- paste("Layer", data$layer, "Head", data$head)

# Create the scatter plot with modified titles and themes
combined_plot <- ggplot(data, aes(x = mem_input, y = cp_input)) +
  geom_point(col="#357EDD", alpha=0.8) +
  facet_wrap(~ subtitle, ncol = 2, nrow = 4) +
  theme_light() +
  labs(x = "Source token = Subject", y = "Source token = Altered") +
  theme(
    axis.text.x = element_text(size=15, angle=90),
    axis.text.y = element_text(size=15),
    axis.title.x = element_text(size=20),
    axis.title.y = element_text(size=20),
    title = element_text(size=20),
    strip.text = element_text(size=20)
  )
combined_plot


ggsave(paste(folder_name, "ov_scatterplot.pdf", sep = "/"), combined_plot, width = 16, height = 10)


