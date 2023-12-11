# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tidyr)

# Load the dataset (Assuming the CSV is in the current working directory)
data <- read.csv("results/plot_data/ov_mem_cp.csv")

# Remove the 'Unnamed: 0' column
data <- data[ , !(names(data) %in% c('Unnamed: 0'))]

# Randomly select 10 indices
set.seed(123) # for reproducibility
selected_indices <- sample(0:143, 10) # Assuming there are 144 pairs of head_X_mem and head_X_cp

get_layer_head <- function(X) {
  layer <- X %/% 12  # Integer division by 12
  head <- X %% 12    # Modulo operation to get the remainder
  return(c(layer, head))
}

get_original_index <- function(layer, head) {
  X <- layer * 12 + head
  return(X)
}


selected_indices <- c(
  get_original_index(11, 0),
  get_original_index(8, 5),
  get_original_index(10, 0),
  get_original_index(7, 6),
  get_original_index(9, 6),
  get_original_index(11, 10),
  get_original_index(10, 7),
  get_original_index(11, 3),
  get_original_index(8, 6),
  get_original_index(0, 10)
)

# Prepare the data for plotting in a more efficient way
head_cols <- paste0("head_", selected_indices)
plot_data <- data.frame(
  index = rep(selected_indices, each = nrow(data)),
  head_mem = as.vector(as.matrix(data[, paste0(head_cols, "_mem")])),
  head_cp = as.vector(as.matrix(data[, paste0(head_cols, "_cp")]))
)

# Map the indices to "L{layer}H{head}" format
plot_data$subtitle <- paste0("L", plot_data$index %/% 12, "H", plot_data$index %% 12)

# Ensure 'index' is a factor for plotting
plot_data$subtitle <- as.factor(plot_data$subtitle)


subtitle_order <- paste0("L", selected_indices %/% 12, "H", selected_indices %% 12)
plot_data$subtitle <- factor(plot_data$subtitle, levels = subtitle_order)

# Define custom labeller

custom_labeller <- function(variable, value) {
  return(as.character(value))
}
sample_data <- sample_n(plot_data, 50000)
# Create the scatter plot with modified titles
ggplot(sample_data, aes(x = head_mem, y = head_cp)) +
  geom_point(col="#357EDD", alpha=0.8) +
  facet_wrap(~ subtitle, ncol = 2, nrow=5, labeller = custom_labeller) +
  theme_light()+
  labs(x = "Source token = Subject", y = "Source token = Altered")+
  theme(
    axis.text.x = element_text(size=15, angle = 90), # Increase x-axis label size and angle
    axis.text.y = element_text(size=15), # Increase y-axis label size
    axis.title.x = element_text(size = 20), # Increase x-axis label size and angle
    axis.title.y = element_text(size = 20), # Increase y-axis label size
    title = element_text(size=20),
    strip.text = element_text(size = 20), # Increase subtitle size
  )
ggsave("results/plots/ov_mem_cp.pdf", width = 20, height = 30, units = "cm", dpi = 100)









