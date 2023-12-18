library(ggplot2)
library(dplyr)
library(tidyr)
# Accept command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check if a file name is provided
if (length(args) == 0) {
  stop("No file name provided. Usage: Rscript script_name.R <filename>")
}
folder_name <- args[1]
std_dev <- as.numeric(args[2])

#################################################################################################

create_heatmap <- function(data, x, y, fill, title) {
  # Convert strings to symbols for tidy evaluation
  x_sym <- rlang::sym(x)
  y_sym <- rlang::sym(y)
  fill_sym <- rlang::sym(fill)
  
  p<- ggplot(data, aes(!!x_sym, !!y_sym, fill = !!fill_sym)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
    theme(axis.text.x = element_text(angle = 0, vjust = 0.5, hjust=1)) +
    labs(x = x, y = y, title = title) +
    geom_text(aes(label = sprintf("%.2f", !!fill_sym)), color = "black", size = 3)
  return(p)
}

plot_pattern <- function(l, h, data){
  selected_layer <- l
  selected_head <- h
  data_head <- data %>% filter(layer == selected_layer & head == selected_head)
  max_source_position <- max(as.numeric(data_head$source_position))
  max_dest_position <- max(as.numeric(data_head$dest_position))
  data_head$source_position <- factor(data_head$source_position, levels = c(0:max_source_position))
  data_head$dest_position <- factor(data_head$dest_position, levels = c(0:max_dest_position))
  
  #reorder the source_position and dest_position contrary to the order of the factor
  data_head$source_position <- factor(data_head$source_position, levels = rev(levels(data_head$source_position)))
  
  p <- create_heatmap(data_head, "dest_position", "source_position", "value", paste("Layer", selected_layer, "Head", selected_head, sep = " "))
  return(p)
}
#################################################################################################


data <- read.csv(paste(folder_name, "head_pattern_data.csv", sep = "/"))
#data <- read.csv("head_pattern_data.csv")

number_of_layers <- max(as.numeric(data$layer))
number_of_heads <- max(as.numeric(data$head))

for (l in 0:number_of_layers){
  for (h in 0:number_of_heads){
    p <- plot_pattern(l, h, data)
    ggsave(paste(folder_name, paste("head_pattern_layer_", l, "_head_", h, ".png", sep = ""), sep="/"), p, width = 10, height = 10)
  }
}





