library(ggpubr)
library(ggplot2)
library(dplyr)
library(tidyr)

args <- commandArgs(trailingOnly = TRUE)


if (length(args) == 0) {
  stop("No file name provided. Usage: Rscript script_name.R <filename>")
}
folder_name <- args[1]
std_dev <- as.numeric(args[2])
#std_dev <- as.numeric(args[2])

data <- read.csv(paste(folder_name, "ablation_data.csv", sep = "/"))
#data <- read.csv("ablation_data.csv")


############################ function ########################################
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

############################ mlp_out ########################################
data_mlp_out <- data %>% filter(grepl("mlp_out", component))
max_layer <- max(as.numeric(data_mlp_out$layer))
max_position <- max(as.numeric(data_mlp_out$position))
data_mlp_out$layer <- factor(data_mlp_out$layer, levels = c(0:max_layer))
data_mlp_out$position <- factor(data_mlp_out$position, levels = c(0:max_position))

####### mem
p_mem <- create_heatmap(data_mlp_out, "position", "layer", "mem", "MLP out ablation - mem")
###### cp
p_cp <- create_heatmap(data_mlp_out, "position", "layer", "cp", "MLP out ablation - cp")
###### mem - cp
p_diff <- create_heatmap(data_mlp_out, "position", "layer", "diff", "MLP out ablation - diff")

if (std_dev == 1) {
  p_mem_std <- create_heatmap(data_mlp_out, "position", "layer", "mem_std", "MLP out ablation - mem std")
  p_cp_std <- create_heatmap(data_mlp_out, "position", "layer", "cp_std", "MLP out ablation - cp std")
  p_diff_std <- create_heatmap(data_mlp_out, "position", "layer", "diff_std", "MLP out ablation - diff std")
  # arrange all plots in 3 x 2 grid
  p <- ggarrange(p_mem, p_cp, p_diff, p_mem_std, p_cp_std, p_diff_std, nrow = 3, ncol = 2)
  ggsave(paste(folder_name, "mlp_out_ablation.pdf", sep = "/"), p, width = 15, height = 10, units = "in", dpi = 300)
} else {
  p <- ggarrange(p_mem, p_cp, p_diff, nrow = 3)
  ggsave(paste(folder_name, "mlp_out_ablation.pdf", sep = "/"), p, width = 10, height = 10, units = "in", dpi = 300)
}

############################ attn_out ########################################
data_attn_out <- data %>% filter(grepl("attn_out", component))
max_layer <- max(as.numeric(data_attn_out$layer))
max_position <- max(as.numeric(data_attn_out$position))
data_attn_out$layer <- factor(data_attn_out$layer, levels = c(0:max_layer))
data_attn_out$position <- factor(data_attn_out$position, levels = c(0:max_position))

####### mem
p_mem <- create_heatmap(data_attn_out, "position", "layer", "mem", "Attn out ablation - mem")
p_cp <- create_heatmap(data_attn_out, "position", "layer", "cp", "Attn out ablation - cp")
p_diff <- create_heatmap(data_attn_out, "position", "layer", "diff", "Attn out ablation - diff")

if (std_dev == 1) {
  p_mem_std <- create_heatmap(data_attn_out, "position", "layer", "mem_std", "Attn out ablation - mem std")
  p_cp_std <- create_heatmap(data_attn_out, "position", "layer", "cp_std", "Attn out ablation - cp std")
  p_diff_std <- create_heatmap(data_attn_out, "position", "layer", "diff_std", "Attn out ablation - diff std")
  # arrange all plots in 3 x 2 grid
  p <- ggarrange(p_mem, p_cp, p_diff, p_mem_std, p_cp_std, p_diff_std, nrow = 3, ncol = 2)
  ggsave(paste(folder_name, "attn_out_ablation.pdf", sep = "/"), p, width = 15, height = 10, units = "in", dpi = 300)
} else {
  p <- ggarrange(p_mem, p_cp, p_diff, nrow = 3)
  ggsave(paste(folder_name, "attn_out_ablation.pdf", sep = "/"), p, width = 10, height = 10, units = "in", dpi = 300)
}

########################### resid_pre ########################################
data_resid_pre <- data %>% filter(grepl("resid_pre", component))
max_layer <- max(as.numeric(data_resid_pre$layer))
max_position <- max(as.numeric(data_resid_pre$position))
data_resid_pre$layer <- factor(data_resid_pre$layer, levels = c(0:max_layer))
data_resid_pre$position <- factor(data_resid_pre$position, levels = c(0:max_position))
data_resid_pre$position <- factor(data_resid_pre$position, levels = c(0:max_position))

####### mem
p_mem <- create_heatmap(data_resid_pre, "position", "layer", "mem", "Resid pre ablation - mem")
p_cp <- create_heatmap(data_resid_pre, "position", "layer", "cp", "Resid pre ablation - cp")
p_diff <- create_heatmap(data_resid_pre, "position", "layer", "diff", "Resid pre ablation - diff")

if (std_dev == 1) {
  p_mem_std <- create_heatmap(data_resid_pre, "position", "layer", "mem_std", "Resid pre ablation - mem std")
  p_cp_std <- create_heatmap(data_resid_pre, "position", "layer", "cp_std", "Resid pre ablation - cp std")
  p_diff_std <- create_heatmap(data_resid_pre, "position", "layer", "diff_std", "Resid pre ablation - diff std")
  # arrange all plots in 3 x 2 grid
  p <- ggarrange(p_mem, p_cp, p_diff, p_mem_std, p_cp_std, p_diff_std, nrow = 3, ncol = 2)
  ggsave(paste(folder_name, "resid_pre_ablation.pdf", sep = "/"), p, width = 15, height = 10, units = "in", dpi = 300)
} else {
  p <- ggarrange(p_mem, p_cp, p_diff, nrow = 3)
  ggsave(paste(folder_name, "resid_pre_ablation.pdf", sep = "/"), p, width = 10, height = 10, units = "in", dpi = 300)
}



############################## HEAD ##########################################
data_head <- data %>% filter(grepl("head", component))
max_layer <- max(as.numeric(data_head$layer))
max_head <- max(as.numeric(data_head$head))
data_head$layer <- factor(data_head$layer, levels = c(0:max_layer))
data_head$head <- factor(data_head$head, levels = c(0:max_head))

p_mem <- create_heatmap(data_head, "head", "layer", "mem", "Head ablation - mem")
p_cp <- create_heatmap(data_head, "head", "layer", "cp", "Head ablation - cp")
p_diff <- create_heatmap(data_head, "head", "layer", "diff", "Head ablation - diff")

if (std_dev == 1) {
  p_mem_std <- create_heatmap(data_head, "head", "layer", "mem_std", "Head ablation - mem std")
  p_cp_std <- create_heatmap(data_head, "head", "layer", "cp_std", "Head ablation - cp std")
  p_diff_std <- create_heatmap(data_head, "head", "layer", "diff_std", "Head ablation - diff std")
  # arrange all plots in 3 x 2 grid
  p <- ggarrange(p_mem, p_cp, p_diff, p_mem_std, p_cp_std, p_diff_std, nrow = 3, ncol = 2)
  ggsave(paste(folder_name, "head_ablation.pdf", sep = "/"), p, width = 15, height = 10, units = "in", dpi = 300)
} else {
  p <- ggarrange(p_mem, p_cp, p_diff, nrow = 3)
  ggsave(paste(folder_name, "head_ablation.pdf", sep = "/"), p, width = 10, height = 10, units = "in", dpi = 300)
}








