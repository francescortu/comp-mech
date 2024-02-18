library(ggplot2)
library(dplyr)
library(tidyr)

args <- commandArgs(trailingOnly = TRUE)


if (length(args) == 0) {
  stop("No file name provided. Usage: Rscript script_name.R <filename>")
}
folder_name <- args[1]
#folder_name <- "results/copyVSfact/logit_lens/gpt2_full"
#std_dev <- as.numeric(args[2])

data <- read.csv(paste(folder_name, "logit_lens_data.csv", sep = "/"))
#data <- read.csv("logit_lens_data.csv")
# data <- read.csv("logit_lens_data.csv")
number_of_position <- max(as.numeric(data$position))
########################### resid_post ########################################
data_resid_post <- data %>% filter(grepl("resid_post", component))
p <- ggplot(data_resid_post, aes(x=data$position, y=data$layer, fill=data$mem)) +
  #geom_tile()+
  geom_point(shape=21, stroke=0.6, size=33)+
  geom_text(aes(label=paste(paste(round(data$mem, 1)))), color="#0f0f0f", size=10)+
  scale_fill_gradient2(low = "#000022",
                       mid = "#ffffff",
                       high="#357EDD",
                       labels = function(x) paste0(round(x, 1), "%"))+
  labs(x='Postion', y='Layers', fill='')+
  scale_y_continuous(breaks = seq(0, 11, 1))+
  scale_x_continuous(breaks = seq(0, number_of_position, 1))+
  theme_minimal() +  # Minimal theme  
  theme(
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.title = element_text(size=20),
    legend.text = element_text(size=30),
    legend.key.size = unit(2, "cm"),
    axis.text.x = element_text(size=30, angle = 90), # Increase x-axis label size and angle
    axis.text.y = element_text(size=30), # Increase y-axis label size
    axis.title.x = element_text(size=40), # Increase x-axis label size and angle
    axis.title.y = element_text(size=40), # Increase y-axis label size
    title = element_text(size=30)
  )
ggsave(paste(folder_name, "resid_post_mem.pdf", sep = "/"), p,  width = 20, height = 15)

p <- ggplot(data_resid_post, aes(x=data$position, y=data$layer, fill=data$cp)) +
  #geom_tile()+
  geom_point(shape=21, stroke=0.6, size=33)+
  geom_text(aes(label=paste(paste(round(data$cp, 1)))), color="#0f0f0f", size=10)+
  scale_fill_gradient2(low = "#000022",
                       mid = "#ffffff",
                       high="#FF725C",
                       labels = function(x) paste0(round(x, 1), "%"))+
  labs(x='Postion', y='Layers', fill='')+
  scale_y_continuous(breaks = seq(0, 11, 1))+
  scale_x_continuous(breaks = seq(0, number_of_position, 1))+
  theme_minimal() +  # Minimal theme  
  theme(
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.title = element_text(size=20),
    legend.text = element_text(size=30),
    legend.key.size = unit(2, "cm"),
    axis.text.x = element_text(size=30, angle = 90), # Increase x-axis label size and angle
    axis.text.y = element_text(size=30), # Increase y-axis label size
    axis.title.x = element_text(size=40), # Increase x-axis label size and angle
    axis.title.y = element_text(size=40), # Increase y-axis label size
    title = element_text(size=30)
  )
ggsave(paste(folder_name, "resid_post_cp.pdf", sep = "/"), p, width = 20, height = 15)

