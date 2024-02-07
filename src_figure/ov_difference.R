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
folder_name <- "results/copyVSfact/ov_difference/gpt2_full"

data <- read.csv(paste(folder_name, "ov_difference_data.csv", sep = "/"))
# Select 8 layer heads
selected_combinations <- data.frame(layer = c(11, 10, 10,10, 9, 9),
                                     head = c(10, 0, 7,  10, 6, 9))


# Filter the data to only include the selected layer heads
data <- data %>%
  mutate(combination = paste("Layer", layer, "Head", head)) %>%
  filter(combination %in% paste("Layer", selected_combinations$layer, "Head", selected_combinations$head))

data_sampled <- data %>%
  group_by(combination) %>%
  sample_frac(.3) # or use sample_n(n) for a fixed number of samples per group
# Reset the grouping
ungroup(data_sampled)


# create a new column for subtitle
data_sampled$subtitle <- paste("Layer", data_sampled$layer, "Head", data_sampled$head)

# Create the scatter plot with modified titles and themes
combined_plot <- ggplot(data_sampled, aes(x = mem_input, y = cp_input)) +
  geom_point(col="#357EDD", alpha=0.8) +
  facet_wrap(~ subtitle, ncol = 2, nrow = 3) +
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


ggsave(paste(folder_name, "copyVSfact_ov_scatterplot.pdf", sep = "/"), combined_plot, width = 10, height = 14)


# Assuming your dataframe is named df
# First, filter for the specific layer and head
filtered_df <- data %>% filter(layer == 11, head == 10)
# Then, calculate the difference
filtered_df <- filtered_df %>%
  mutate(difference = mem_input - cp_input)
# Now, plot the histogram of the differences
ggplot(filtered_df, aes(x=difference)) +
  geom_histogram(binwidth = 0.1, fill="blue", color="black") +
  ggtitle("Histogram of Differences between mem_input and cp_input") +
  xlab("Difference") +
  ylab("Frequency")

