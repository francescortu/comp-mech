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

#folder_name <- 

######################################### function #################################################################

create_heatmap <- function(data, x, y, fill, title) {
  # Convert strings to symbols for tidy evaluation
  x_sym <- rlang::sym(x)
  y_sym <- rlang::sym(y)
  fill_sym <- rlang::sym(fill)
  
  p<- ggplot(data, aes(!!x_sym, !!y_sym, fill = !!fill_sym)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    labs(x = x, y = y, title = title) +
    geom_text(aes(label = sprintf("%.2f", !!fill_sym)), color = "black", size = 3)
  return(p)
}
########################################## script #############################################################
# Set working directory and read data
data <- read.csv(paste(folder_name, "logit_attribution_data.csv", sep = "/"))

#############################################################################################################
###################################       HEAD              #################################################
#############################################################################################################

## filter the data to only include label of type LiHj

data_head <- data %>% filter(grepl("^L[0-9]+H[0-9]+$", label))
number_of_position <- max(as.numeric(data_head$position))
## filter to have just position 12
for (pos in 0:number_of_position) {

  data_head_ <- data_head %>% filter(position == pos)
  
  # for each row split L and H and create a new column for each
  data_head_ <- data_head_ %>% separate(label, c("layer", "head"), sep = "H")
  #remove L from layer
  data_head_$layer <- gsub("L", "", data_head_$layer)
  
  max_layer <- max(as.numeric(data_head_$layer))
  max_head <- max(as.numeric(data_head_$head))
  data_head_$layer <- factor(data_head_$layer, levels = c(0:max_layer))
  data_head_$head <- factor(data_head_$head, levels = c(0:max_head))
  
  # Your existing ggplot code
  p <- create_heatmap(data_head_, "head", "layer", "diff_mean", paste("Head Attribution for Position", pos, sep=" "))
  if (std_dev == 1) {
    long_data <- data_head_ %>% 
      gather(key = "attribute", value = "value", diff_mean, diff_std)
  
    # Creating the heatmap with facet_wrap
    p <- create_heatmap(long_data, "head", "layer", "value",  paste("Head Attribution for Position", pos, sep=" ")) +
      facet_wrap(~ attribute, ncol = 1)
  }
  # save the plot
  #ggsave( paste("logit_attribution_head_position", pos, ".pdf", sep=""), p, width = 10, height = 10, units = "in")
  
  ggsave(paste(folder_name, paste("logit_attribution_head_position", pos, ".pdf", sep=""), sep="/"), p, width = 10, height = 10, units = "in")
  #ggsave(paste(folder_name, "logit_attribution_head.pdf", sep = "/"), p, width = 10, height = 10, units = "in")
}

####################################################################################################
####################################  MLP OUT #####################################################
####################################################################################################

#filter position f"{i}_mlp_out"
data_mlp <- data %>% filter(grepl("^[0-9]+_mlp_out$", label))
data_mlp <- data_mlp %>% separate(label, c("layer"), sep = "_mlp_out")
max_layer <- max(as.numeric(data_mlp$layer))
max_position <- max(as.numeric(data_mlp$position))
#create layer column

data_mlp$layer <- factor(data_mlp$layer, levels = c(0:max_layer))
data_mlp$position <- factor(data_mlp$position, levels = c(0:max_position))

p <- create_heatmap(data_mlp, "position", "layer", "diff_mean", "MLP Out Attribution")

if (std_dev == 1) {
  long_data <- data_mlp %>% 
    gather(key = "attribute", value = "value", diff_mean, diff_std)

  # Creating the heatmap with facet_wrap
  p <- create_heatmap(long_data, "position", "layer", "value", "MLP Out Attribution") +
    facet_wrap(~ attribute, ncol = 1)
}
# save the plot
ggsave(paste(folder_name, "logit_attribution_mlp_out.pdf", sep = "/"), p, width = 10, height = 10, units = "in")


####################################################################################################
####################################  ATTN OUT #####################################################
####################################################################################################

#filter position f"{i}_mlp_out"
data_attn <- data %>% filter(grepl("^[0-9]+_attn_out$", label))
data_attn <- data_attn %>% separate(label, c("layer"), sep = "_attn_out")
max_layer <- max(as.numeric(data_attn$layer))
max_position <- max(as.numeric(data_attn$position))
#create layer column

data_attn$layer <- factor(data_attn$layer, levels = c(0:max_layer))
data_attn$position <- factor(data_attn$position, levels = c(0:max_position))

p <- create_heatmap(data_attn, "position", "layer", "diff_mean", "Attn Out Attribution")

if (std_dev == 1) {
  long_data <- data_attn %>% 
    gather(key = "attribute", value = "value", diff_mean, diff_std)
  
  # Creating the heatmap with facet_wrap
  p <- create_heatmap(long_data, "position", "layer", "value", "Attn Out Attribution") +
    facet_wrap(~ attribute, ncol = 1)
}
# save the plot
ggsave(paste(folder_name, "logit_attribution_attn_out.pdf", sep = "/"), p, width = 10, height = 10, units = "in")

