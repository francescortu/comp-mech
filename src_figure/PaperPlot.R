library(ggplot2)
library(readr)
library(dplyr)
library(binom)
library(latex2exp)
library(tidyr)
library(ggpubr)
setwd("/home/francesco/Repository/Competition_of_Mechanisms/results")

palette <- c("#D5008F", "#19A974", "#A463F2", "#FFB700", "#FF6300")
palette <- c("GPT2" = "#003f5c", "GPT2-medium" = "#58508d", "GPT2-large" = "#bc5090", "GPT2-xl" = "#ff6361", "Pythia-6.9b" = "#ffa600")

#define the color palette

################################## EXPERIMENT CONFIGS ########################################
experiment <- "copyVSfact"
n_positions <- 12
positions_name <- c("-", "Subject", "2nd Subject", "3rd Subject", "Relation", "Relation Last", "Attribute*", "-", "Subject Repeat", "2nd Subject repeat", "3nd Subject repeat", "Relation repeat", "Last")
relevant_position <- c("Subject", "Relation", "Relation Last", "Attribute*", "Subject repeat", "Relation repeat", "Last")
n_relevant_position <- 7
#GPT2
layer_pattern <- c(11,10,10,10,9,9)
head_pattern <- c(10,0,7,10,6,9)
#Pythia-6.9b
#layer_pattern <- c(21,20,20,19)
#head_pattern <- c(8,2,18,31)
  
#experiment <- "contextVSfact"
#n_positions <- 8
#positions_name <- c("-", "object", "-", "last", "1st subject", "2nd subject", "3rd subject", "-", "last")

################################ MODEL CONFIGS ################################################
model <- "gpt2"
model_folder <- "gpt2_full"
n_layers <- 12


# model <- "pythia-6.9b"
# model_folder <- "pythia-6.9b_full"
# n_layers <- 32


############################### PLOT CONFIGS ##################################################
AXIS_TITLE_SIZE <- 60
AXIS_TEXT_SIZE <- 40
HEATMAP_SIZE <- 10

###################################### functions ############################################
create_heatmap_base <- function(data, x, y, fill, midpoint = 0, text=FALSE) {
  # Convert strings to symbols for tidy evaluation
  x_sym <- rlang::sym(x)
  y_sym <- rlang::sym(y)
  fill_sym <- rlang::sym(fill)
  if (text==TRUE){
    p<- ggplot(data, aes(!!x_sym, !!y_sym, fill = !!fill_sym)) +
      geom_tile(colour = "grey") +
      scale_fill_gradient2(low = "#a00000", mid = "white", high = "#1a80bb", midpoint = midpoint) +
      theme(axis.text.x = element_text(angle = 0, vjust = 0.5, hjust=1)) +
      geom_text(aes(label = sprintf("%.2f", !!fill_sym)), color = "black", size = HEATMAP_SIZE)+
      labs(x = x, y = y)
  }else{
  p<- ggplot(data, aes(!!x_sym, !!y_sym, fill = !!fill_sym)) +
    geom_tile(colour = "grey") +
   scale_fill_gradient2(low = "#a00000", mid = "white", high = "#1a80bb") +
    theme(axis.text.x = element_text(angle = 0, vjust = 0.5, hjust=1)) +
   # geom_text(aes(label = sprintf("%.2f", !!fill_sym)), color = "black", size = HEATMAP_SIZE)+
    labs(x = x, y = y) 
  }
  return(p)
}

#################################################################################################################
################################################### Ablation ####################################################
#################################################################################################################
experiment <- "copyVSfact_ablation_window"
model_folder <- "gpt2_0_full_total_effect"
data_gpt2 <- read.csv(sprintf("%s/ablation/%s/ablation_data_attn_out_pattern.csv", experiment, model_folder))
model_folder <- "gpt2-medium_0_full_total_effect"
data_gpt2_medium <- read.csv(sprintf("%s/ablation/%s/ablation_data_attn_out_pattern.csv", experiment, model_folder))
model_folder <- "gpt2-large_0_full_total_effect"
data_gpt2_large <- read.csv(sprintf("%s/ablation/%s/ablation_data_attn_out_pattern.csv", experiment, model_folder))
model_folder <- "gpt2-xl_0_full_total_effect"
data_gpt2_xl <- read.csv(sprintf("%s/ablation/%s/ablation_data_attn_out_pattern.csv", experiment, model_folder))
model_folder <- "pythia-6.9b_0_full_total_effect"
data_pythia <- read.csv(sprintf("%s/ablation/%s/ablation_data_attn_out_pattern.csv", experiment, model_folder))

#for each model, save only the columns layer, position, mem_sum
data_gpt2 <- data_gpt2 %>% select(layer, position, mem_sum, cp_sum)
data_gpt2_medium <- data_gpt2_medium %>% select(layer, position, mem_sum, cp_sum)
data_gpt2_large <- data_gpt2_large %>% select(layer, position, mem_sum, cp_sum)
data_gpt2_xl <- data_gpt2_xl %>% select(layer, position, mem_sum, cp_sum)
data_pythia <- data_pythia %>% select(layer, position, mem_sum, cp_sum)

#filter only the relevant positions, just select position 6
data_gpt2 <- data_gpt2 %>% filter(position == 6)
data_gpt2_medium <- data_gpt2_medium %>% filter(position == 6)
data_gpt2_large <- data_gpt2_large %>% filter(position == 6)
data_gpt2_xl <- data_gpt2_xl %>% filter(position == 6)
data_pythia <- data_pythia %>% filter(position == 6)

# compute percentage
data_gpt2$mem_perc <- (data_gpt2$mem_sum / 10000) * 100
data_gpt2$cp_perc <- (data_gpt2$cp_sum / 10000) * 100
data_gpt2_medium$mem_perc <- (data_gpt2_medium$mem_sum / 10000) * 100
data_gpt2_medium$cp_perc <- (data_gpt2_medium$cp_sum / 10000) * 100
data_gpt2_large$mem_perc <- (data_gpt2_large$mem_sum / 10000) * 100
data_gpt2_large$cp_perc <- (data_gpt2_large$cp_sum / 10000) * 100
data_gpt2_xl$mem_perc <- (data_gpt2_xl$mem_sum / 10000) * 100
data_gpt2_xl$cp_perc <- (data_gpt2_xl$cp_sum / 10000) * 100
data_pythia$mem_perc <- (data_pythia$mem_sum / 10000) * 100
data_pythia$cp_perc <- (data_pythia$cp_sum / 10000) * 100

#put all the data together and add a column with the model name
data <- rbind(data_gpt2, data_gpt2_medium, data_gpt2_large, data_gpt2_xl, data_pythia)
data$model <- c(rep("gpt2", nrow(data_gpt2)), rep("gpt2-medium", nrow(data_gpt2_medium)), rep("gpt2-large", nrow(data_gpt2_large)), rep("gpt2-xl", nrow(data_gpt2_xl)), rep("pythia", nrow(data_pythia)))

#for each model select the row with the maximum memory usage
data_max <- data %>% group_by(model) %>% filter(mem_sum == max(mem_sum))

#modify the name of the models to make them more readable
data_max$model <- factor(data_max$model, levels = c("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "pythia"), labels = c("GPT2", "GPT2-medium", "GPT2-large", "GPT2-xl", "Pythia-6.9b"))
#plot a bar plot with the memory usage for each model
ggplot(data_max)+
  geom_bar(aes(x = model, y = mem_perc, fill=model), stat="identity")+
  scale_fill_manual(values = palette)+
  labs(x = "", y = "Factual Recall at the best couple of layer")+
  theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=1)) +
  scale_y_continuous(limits = c(0, 100)) +
  ggtitle("Factual recall") +
  theme(plot.title = element_text(hjust = 0.5))






data <- data %>% filter(grepl("6", position))
data$layer <- as.numeric(data$layer)
data$mem_perc <- (data$mem_sum / 10000) * 100
data$cp_perc <- (data$cp_sum / 10000) * 100
ggplot(data)+
  geom_line(aes(x = layer, y = mem_perc))+
  geom_point(aes(x = layer, y = mem_perc))
library(tidyr) # For pivoting the data into a long format

ggplot(data)+
  geom_bar(aes(x = layer, y = mem_perc), stat="identity", fill="#a00000")+
  scale_x_discrete(limits = c(0,1,2,3,4,5,6,7), labels=c("0-3","4-7","8-11","12-15","16-19","20-23","24-27","28-31"))+
  #geom_hline(yintercept = 4.13, linetype="dashed", color = "red")+
  #add title and labels
  labs(x = "Layer", y = "Percentage", title = "Pythia, Percentage of wins for the factual mechanism \nwhen the altered token position is ablated in the Attention Pattern.")

  
experiment <- "copyVSfact_factual_prova"
model_folder <- "gpt2_0_full_total_effect"
data <- read.csv(sprintf("%s/ablation/%s/ablation_data_attn_out_pattern.csv", experiment, model_folder))
data <- data %>% filter(grepl("6", position))
data$layer <- as.numeric(data$layer)
data$mem_perc <- (data$mem_sum / 10000) * 100
data$cp_perc <- (data$cp_sum / 10000) * 100


ggplot(data)+
  geom_bar(aes(x = layer, y = cp_perc), stat="identity", fill="#a00000")+
  #geom_hline(yintercept = 4.13, linetype="dashed", color = "red")+
  #add title and labels
  labs(x = "Layer", y = "Percentage", title = "Percentage of wins for the factual mechanism \nwhen the altered token position is ablated in the Attention Pattern.")

experiment <- "copyVSfact"
model_folder <- "gpt2_full_total_effect"
data <- read.csv(sprintf("%s/ablation/%s/ablation_data_attn_out_pattern.csv", experiment, model_folder))
data <- data %>% filter(grepl("6", position))
data$layer <- as.numeric(data$layer)
data$mem_perc <- (data$mem_sum / 10000) * 100
data$cp_perc <- (data$cp_sum / 10000) * 100

ggplot(data)+
  geom_bar(aes(x = layer, y = cp_perc), stat="identity", fill="#a00000")+
  #geom_hline(yintercept = 4.13, linetype="dashed", color = "red")+
  #add title and labels
  labs(x = "Layer", y = "Percentage", title = "Percentage of wins for the factual mechanism \nwhen the altered token position is ablated in the Attention Pattern.")

data_long <- tidyr::pivot_longer(data, cols = c(cp_perc, mem_perc), names_to = "variable", values_to = "value")
ggplot(data_long, aes(x = layer, y = value, fill = variable)) +
  geom_bar(stat="identity", position=position_dodge(width=0.7)) +
  geom_hline(yintercept = 4.13, linetype="dashed", color = "red") +
  scale_fill_manual(values = c("cp_perc" = "#1a80bb", "mem_perc" = "#a00000")) +
  labs(x = "Layer", y = "Percentage", title = "Percentage of wins for the factual mechanism \nwhen the altered token position is ablated in the Attention Pattern.") +
  theme_minimal()

#################################################################################################################
##################################### LOGIT LENS ################################################################
#################################################################################################################

AXIS_TITLE_SIZE <- 60
AXIS_TEXT_SIZE <- 60
HEATMAP_SIZE <- 10

create_heatmap <- function(data, x, y, fill, high_color) {
  p <- create_heatmap_base(data, x, y, fill) +
    scale_fill_gradient2(low = "black", mid = "white", high = high_color, limits = c(0,31), name = "Logit") +
    #scale_fill_gradient2(low = "black", mid = "white", high = high_color, name = "Logit") +
    #scale_fill_gradient2(low = "black", mid= "white", high = high_color, name = "Logit") +
    
    theme_minimal() +
    #addforce to have all the labels
    scale_x_continuous(breaks = seq(0,n_layers-1 ,1)) +
    scale_y_continuous(breaks = seq(0,n_relevant_position -1,1), labels = relevant_position) +
    scale_y_reverse(breaks = seq(0,n_relevant_position -1,1), labels = relevant_position)+
    labs(x = "Layer", y = "")+
    #fix intenxity of fill
    theme(
      axis.text.x = element_text(size=AXIS_TEXT_SIZE),
      axis.text.y = element_text(size=AXIS_TEXT_SIZE,),
      #remove background grid
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.title.x = element_text(size = AXIS_TITLE_SIZE),
      axis.title.y = element_text(size = AXIS_TITLE_SIZE),
      legend.text = element_text(size = 30),
      legend.title = element_text(size = AXIS_TEXT_SIZE),
      #remove the legend\
      legend.position = "bottom",
      #increase the legend size
      legend.key.size = unit(2, "cm"),
      # move the y ticks to the right
    ) 
  return(p)
}

experiment <- "copyVSfact"
model_folder <- "gpt2_full"
data <- read.csv(sprintf("%s/logit_lens/%s/logit_lens_data_logit.csv", experiment, model_folder))
number_of_position <- max(as.numeric(data$position))
data_resid_post <- data %>% filter(grepl("resid_post", component))
data_resid_post$position_name <- positions_name[data_resid_post$position + 1]
#filter just the relevant positions
data_resid_post <- data_resid_post %>% filter(position == 1 | position==4 | position == 5 | position== 6 | position==8 |  position == 11 | position== 12)
unique_positions <- unique(data_resid_post$position)
position_mapping <- setNames(seq(0, length(unique_positions) - 1), unique_positions)
# Apply the mapping to create a new column
data_resid_post$mapped_position <- unname(position_mapping[as.character(data_resid_post$position)])

#rename the columns name of data_resid_post$mem


p <- create_heatmap(data_resid_post, "layer","mapped_position", "mem",  "#E31B23")
p
ggsave(sprintf("PaperPlot/%s_%s_residual_stream/resid_post_mem.pdf", model, experiment), p, width = 50, height = 30, units = "cm")

p <- create_heatmap(data_resid_post, "layer", "mapped_position", "cp",  "#005CAB")
p
ggsave(sprintf("PaperPlot/%s_%s_residual_stream/resid_post_cp.pdf", model, experiment), p, width = 50, height = 30, units = "cm")

data_resid_post$ratio<- data_resid_post$cp / data_resid_post$mem 

p <- create_heatmap(data_resid_post, "layer", "mapped_position", "ratio",  "darkgreen")
p
ggsave(sprintf("PaperPlot/%s_%s_residual_stream/resid_post_cp.pdf", model, experiment), p, width = 50, height = 30, units = "cm")


########### INDEX @@@@@@@@@@
data_resid_post <- data_resid_post %>% filter(position ==6)
ggplot(data_resid_post, aes(x=layer))+
  geom_line(aes(y=cp_idx, color="cp"))+
  scale_color_manual(values = c("cp" = "blue"))+
  labs(y="Logit Rank for position 6", color="Altered")
ggplot(data_resid_post, aes(x=layer))+
  geom_line(aes(y=mem_idx, color="mem"))+
  scale_color_manual(values = c("mem" = "red"))+
  labs(y="Logit Rank for position 6", color="Factual")




library(ggplot2)
library(patchwork)

# First Plot for 'mem'
p1 <- ggplot(data = data_resid_post, aes(x = layer, y = mem, group = position_name, color = position_name)) + 
  geom_line(size=2) +
  geom_point(size=5) + # Optional, adds points to the lines
  theme_minimal() +
  scale_x_continuous(breaks = seq(0,n_layers,1)) +
  labs(
       x = "Layer",
       y = "Logit for Factual Token",
       color = "Position") +
  #ylim(0, 8)+
  scale_color_discrete(name = "Position") + 
  theme(
    axis.text.x = element_text(size=30),
    axis.text.y = element_text(size=30),
    #remove background grid
    axis.title.x = element_text(size = 35 ),
    axis.title.y = element_text(size = 35 ),
    legend.text = element_text(size = 30),
    legend.title = element_text(size = 35),
    #remove the legend\
    legend.position = "bottom",
    # move the y ticks to the right
  ) +
  #two lines legend
  guides(color = guide_legend(ncol = 2))

data_resid_post <- data_resid_post %>% filter(position_name == "Subject" |
                                              position_name == "Relation Last" |
                                              position_name == "Attribute*" |
                                              position_name == "Subject Repeat"|
                                              position_name == "Last")


# Second Plot for 'cp'
p2 <- ggplot(data = data_resid_post, aes(x = layer, y = diff, group = position_name, color = position_name)) + 
  geom_line(size=2) +
  geom_point(size=4) + # Optional, adds points to the lines
  geom_hline(yintercept = 0, linetype="dashed", color = "red", size=2) +
  #add notation
  annotate("text", x = 11, y = 1.2, label = "Competition Threshold", color = "red", size = 8, hjust = 1) +
  theme_minimal() +
  scale_x_continuous(breaks = seq(0,n_layers,1)) +
  labs(x = "Layer",
       y = "Logit for Altered Token", # And the y-axis label change
       color = "Position") +
  #ylim(0, 8)+
  scale_color_discrete(name = "Position") +
  theme(
      axis.text.x = element_text(size=30),
      axis.text.y = element_text(size=30),
      #remove background grid
      axis.title.x = element_text(size = 35 ),
      axis.title.y = element_text(size = 35 ),
      legend.text = element_text(size = 30),
      legend.title = element_text(size = 35),
      #remove the legend\
      legend.position = "bottom",
      # move the y ticks to the right
  ) +
  #two lines legend
  guides(color = guide_legend(ncol = 2)) 
p2

#ggsave(sprintf("PaperPlot/%s_%s_residual_stream/resid_post_mem.pdf", model, experiment), p1, width = 40, height = 35, units = "cm")
ggsave(sprintf("PaperPlot/%s_%s_residual_stream/resid_post_diff.pdf", model, experiment), p2, width = 40, height = 35, units = "cm")




############################################################################################################
######################################## LOGIT ATTRIBUTION ################################################
############################################################################################################
experiment <- "copyVSfact"
model_folder <- "gpt2_full"
data <- read.csv(sprintf("%s/logit_attribution/%s/logit_attribution_data.csv", experiment, model_folder))

create_heatmap <- function(data, x, y, fill, head=FALSE) {
  if(head){
    scale_x <- scale_x_discrete(breaks = seq(0,n_layers,1)) 
    angle = 0
  } else {
    scale_x <- scale_x_discrete(breaks = seq(0, n_positions,1), labels = positions_name)
    angle = 90
  }
  print(n_positions)
  p <- create_heatmap_base(data, x, y, fill) +
    theme_minimal() +
    #addforce to have all the labels
    scale_y_discrete(breaks = seq(0,n_layers,1)) +
    scale_x +
    labs(fill = "Logit Diff") +
    theme(
      axis.text.x = element_text(size=AXIS_TEXT_SIZE, angle = angle),
      axis.text.y = element_text(size=AXIS_TEXT_SIZE),
      #remove background grid
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.title.x = element_text(size = AXIS_TITLE_SIZE),
      axis.title.y = element_text(size = AXIS_TITLE_SIZE),
      legend.text = element_text(size = 30),
      legend.title = element_text(size = 30),
      #remove the legend\
      legend.position = "bottom",
      # increase the size of the legend
      legend.key.size = unit(2, "cm"),
      # move the y ticks to the right
    )
  return(p)
}



######## head #######
data_head <- data %>% filter(grepl("^L[0-9]+H[0-9]+$", label))
number_of_position <- max(as.numeric(data_head$position))
## filter to have just position 12
data_head_ <- data_head %>% filter(position == number_of_position)
# for each row split L and H and create a new column for each
data_head_ <- data_head_ %>% separate(label, c("layer", "head"), sep = "H")
#renominating the columns layer and head to Layer and Head
#remove L from layer
data_head_$layer <- gsub("L", "", data_head_$layer)

max_layer <- max(as.numeric(data_head_$layer))
max_head <- max(as.numeric(data_head_$head))
data_head_$layer <- factor(data_head_$layer, levels = c(0:max_layer))
data_head_$head <- factor(data_head_$head, levels = c(0:max_head))
colnames(data_head_)[1] <- "Layer"
colnames(data_head_)[2] <- "Head"
data_head_$diff_mean <- -data_head_$diff_mean
p <- create_heatmap(data_head_, "Layer", "Head", "diff_mean", head=TRUE)
p <- create_heatmap_base(data_head_, "Layer", "Head", "diff_mean") +
  theme_minimal() +
  #addforce to have all the labels
  scale_y_discrete(breaks = seq(0,n_layers,1)) +
  scale_x_discrete(breaks = seq(0,n_layers,1))  +
  labs(fill = expression(Delta[alt])) +
  theme(
    axis.text.x = element_text(size=AXIS_TEXT_SIZE, angle = 0),
    axis.text.y = element_text(size=AXIS_TEXT_SIZE),
    #remove background grid
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.title.x = element_text(size = AXIS_TITLE_SIZE),
    axis.title.y = element_text(size = AXIS_TITLE_SIZE),
    legend.text = element_text(size = 50),
    legend.title = element_text(size = 90),
    #remove the legend\
    legend.position = "bottom",
    # increase the size of the legend
    legend.key.size = unit(2.5, "cm"),
    # move the y ticks to the right
  )
p
ggsave(sprintf("PaperPlot/%s_%s_logit_attribution/logit_attribution_head_position%s.pdf", model, experiment, number_of_position), p, width = 50, height = 50, units = "cm")

### count the impact of the positive head ###
#sum all the negative values
factual_impact <- data_head_ %>% group_by(Layer) %>% summarise(positive_impact = sum(diff_mean[diff_mean < 0]))
# sum also across layers
factual_impact <- factual_impact %>% summarise(positive_impact = sum(positive_impact))
l10h7 <- data_head_ %>% filter(Layer == 10, Head == 7)
l10h7 <- l10h7$diff_mean
l11h10 <- data_head_ %>% filter(Layer == 11, Head == 10)
l11h10 <- l11h10$diff_mean

l10h7 <- 100 * l10h7 / sum(factual_impact)
print(l10h7)
l11h10 <- 100 * l11h10 / sum(factual_impact)
print(l11h10)
#compute how much l10h7 and l11h10 are responsible for the positive impact (in percentage)

#summ also across 


data_head_


############ ATtn MLP lineplot ########################
data_mlp <- data %>% filter(grepl("^[0-9]+_mlp_out$", label))
data_mlp <- data_mlp %>% separate(label, c("layer"), sep = "_mlp_out")
data_attn <- data %>% filter(grepl("^[0-9]+_attn_out$", label))
data_attn <- data_attn %>% separate(label, c("layer"), sep = "_attn_out")
max_position <- max(as.numeric(data_mlp$position))
#take just the last position
data_mlp <- data_mlp %>% filter(position == max_position)
data_attn <- data_attn %>% filter(position == max_position)

#merge the two dataframe
data_barplot <- data_mlp
data_barplot$attn_dif <- data_attn$diff_mean
data_barplot$attc_cp <- data_attn$cp_mean
data_barplot$attc_mem <- data_attn$mem_mean
#rename columns diff_mean to mlp_dif
data_barplot <- data_barplot %>% rename("mlp_dif" = diff_mean)
data_barplot <- data_barplot %>% rename("mlp_cp" = cp_mean)
data_barplot <- data_barplot %>% rename("mlp_mem" = mem_mean)

#pivoting attn in order to plot mem and cp in the same barplot
data_attn <- data_barplot %>% pivot_longer(cols = c("attc_cp", "attc_mem"), names_to = "Block", values_to = "value")
data_mlp <- data_barplot %>% pivot_longer(cols = c("mlp_cp", "mlp_mem"), names_to = "Block", values_to = "value")
#modify mlp_cp and mlp_mem to Altered and Factual
data_mlp$Block <- gsub("mlp_", "", data_mlp$Block)
data_mlp$Block <- gsub("cp", "Altered", data_mlp$Block)
data_mlp$Block <- gsub("mem", "Factual", data_mlp$Block)
data_attn$Block <- gsub("attc_", "", data_attn$Block)
data_attn$Block <- gsub("cp", "Altered", data_attn$Block)
data_attn$Block <- gsub("mem", "Factual", data_attn$Block)


#barplot MLP Block
ggplot(data_mlp, aes(x = as.numeric(layer), y = value, fill = Block)) +
  geom_col(position = position_dodge(),color="black") +
  scale_fill_manual(values = c("Factual" = "#E31B23", "Altered" = "#005CAB"))+
  scale_y_continuous(limits = c(0, 3.1)) +
  scale_x_continuous(breaks= seq(0, n_layers - 1, 1), labels = c("0","1","2","3","4","5","6","7","8","9","10","11")) +
  labs(x = "Layer", y = "Logit", fill="Token:") +
  theme_minimal() +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 35),
    axis.text.y = element_text(size = 35),
    axis.title.x = element_text(size = 40),
    axis.title.y = element_text(size = 40),
    legend.text = element_text(size = 30),
    legend.title = element_text(size = 30),
    legend.position = "top"
  ) +
  guides(fill = guide_legend(ncol = 2)) # Adjusting the legend
ggsave(sprintf("PaperPlot/%s_%s_logit_attribution/logit_mlp_position%s.pdf", model, experiment, max_position), width = 40, height = 30, units = "cm")

#barplot Attention Block E31B23 005CAB
ggplot(data_attn, aes(x = as.numeric(layer), y = value, fill = Block)) +
  geom_col(position = position_dodge(), color="black")+
  scale_fill_manual(values = c("Factual" = "#E31B23", "Altered" = "#005CAB"))+
  scale_y_continuous(limits = c(0, 3.1)) +
  scale_x_continuous(breaks= seq(0, n_layers - 1, 1), labels = c("0","1","2","3","4","5","6","7","8","9","10","11")) +
  labs(x = "Layer", y = "Logit", fill="Token:") +
  theme_minimal() +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 35),
    axis.text.y = element_text(size = 35),
    axis.title.x = element_text(size = 40),
    axis.title.y = element_text(size = 40),
    legend.text = element_text(size = 30),
    legend.title = element_text(size = 30),
    legend.position = "top"
  ) +
  guides(fill = guide_legend(ncol = 2)) # Adjusting the legend
ggsave(sprintf("PaperPlot/%s_%s_logit_attribution/logit_attn_position%s.pdf", model, experiment, max_position), width = 40, height = 30, , units = "cm")


library(ggplot2)
library(ggsci)
#rename "Diff Mean" to MLP Block an "Attn" to Attention Block
data_barplot <- data_barplot %>% rename("MLP Block" = mlp_dif )
data_barplot <- data_barplot %>% rename("Attention Block" = attn_dif)


data_barplot$`MLP Block` <- -data_barplot$`MLP Block`
data_barplot$`Attention Block` <- -data_barplot$`Attention Block`

data_barplot$layer <- as.numeric(data_barplot$layer) 
#barplot MLP Block


ggplot(data_barplot, aes(x = as.numeric(layer), y = `MLP Block`, fill = "MLP Block")) +
  geom_col(position = position_dodge(), color="black", size=1) +
  labs(x = "Layer", y = expression(Delta[alt]), fill = "") + # Naming the legend
  theme_minimal() +
  scale_fill_manual(values = c("MLP Block" = "#bc5090")) + # Assigning color to the "MLP Block"
  scale_y_continuous(limits = c(-1, 1.5)) +
  scale_x_continuous(breaks= seq(0, n_layers-1, 1), labels = c("0","1","2","3","4","5","6","7","8","9","10","11")) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 50),
    axis.text.y = element_text(size = 50),
    axis.title.x = element_text(size = 55),
    axis.title.y = element_text(size = 55),
    legend.text = element_text(size = 50),
    legend.title = element_text(size = 55),
    legend.position = "top"
  ) +
  guides(fill = guide_legend(ncol = 2.5)) # Adjusting the legend
ggsave(sprintf("PaperPlot/%s_%s_logit_attribution/logit_mlp_position%s_diff.pdf", model, experiment, max_position), width = 40, height = 30, units = "cm")


ggplot(data_barplot, aes(x = as.numeric(layer), y = `Attention Block`, fill = "Attention Block")) +
  geom_col(position = position_dodge(), color="black",size=1) +
  labs(x = "Layer", y = expression(Delta[alt]), fill = "") + # Naming the legend
  theme_minimal() +
  scale_fill_manual(values = c("Attention Block" = "#ffa600")) + # Assigning color to the "MLP Block"
  scale_y_continuous(limits = c(-1, 1.5)) +
  scale_x_continuous(breaks= seq(0, n_layers-1, 1), labels = c("0","1","2","3","4","5","6","7","8","9","10","11")) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(size = 50),
    axis.text.y = element_text(size = 50),
    axis.title.x = element_text(size = 55),
    axis.title.y = element_text(size = 55),
    legend.text = element_text(size = 50),
    legend.title = element_text(size = 55),
    legend.position = "top"
  ) +
  guides(fill = guide_legend(ncol = 2.5)) # Adjusting the legend
ggsave(sprintf("PaperPlot/%s_%s_logit_attribution/logit_attn_position%s_diff.pdf", model, experiment, max_position), width = 40, height = 30, , units = "cm")



# Assuming data_mlp is your dataframe and it has columns: layer, diff_mean (for MLP), attn (for Attention)
data_mlp_long <- pivot_longer(data_mlp, cols = c(`MLP Block`, `Attention Block`), names_to = "Block", values_to = "Value")
#Rename diff_mean to MLP Block and attn to Attention Block in the Block column
data_mlp_long$Block <- gsub("diff_mean", "MLP Block", data_mlp_long$Block)
data_mlp_long$Block <- gsub("attn", "Attention Block", data_mlp_long$Block)

ggplot(data_mlp_long, aes(x = as.numeric(layer), y = Value, fill = Block)) +
  geom_col(position = position_dodge()) +  # Use geom_col for bar plots; position_dodge for side by side
  geom_hline(yintercept = 0, linetype = "dashed", color = "#E7040F", size = 2) +
  annotate("text", x = 2, y = -0.05, label = "Competition Threshold", color = "#E7040F", size = 8, hjust = 1) +
  theme_minimal() +
  scale_x_continuous(breaks = seq(0, max(data_mlp$layer), 1)) +
  scale_fill_manual(values = c(palette[4], palette[3])) +
  labs(x = "Layer", y = "Logit Difference", fill = "Component:") +
  theme(
    axis.text.x = element_text(size = 30),
    axis.text.y = element_text(size = 30),
    axis.title.x = element_text(size = 35),
    axis.title.y = element_text(size = 35),
    legend.text = element_text(size = 30),
    legend.title = element_text(size = 35),
    legend.position = "bottom"
  ) +
  guides(fill = guide_legend(ncol = 2))
ggsave(sprintf("PaperPlot/%s_%s_logit_attribution/logit_attn_mlp_position%s_hist.pdf", model, experiment, max_position), width = 50, height = 40, units = "cm")


########################## APPENDIX #######################################################
######## mlp out #######

data_mlp <- data %>% filter(grepl("^[0-9]+_mlp_out$", label))
data_mlp <- data_mlp %>% separate(label, c("layer"), sep = "_mlp_out")
max_layer <- max(as.numeric(data_mlp$layer))
max_position <- max(as.numeric(data_mlp$position))
#create layer column

data_mlp$layer <- factor(data_mlp$layer, levels = c(0:max_layer))
data_mlp$position <- factor(data_mlp$position, levels = c(0:max_position))

colnames(data_mlp)[1] <- "Layer"
colnames(data_mlp)[2] <- "Position"

data_mlp <- data_mlp %>% filter(Position == 1 | Position==4 | Position == 5 | Position== 6 | Position==8 |  Position == 11 | Position== 12)
unique_positions <- unique(data_mlp$Position)
position_mapping <- setNames(seq(0, length(unique_positions) - 1), unique_positions)
# Apply the mapping to create a new column
data_mlp$mapped_position <- unname(position_mapping[as.character(data_mlp$Position)])
data_mlp$Layer <- as.numeric(data_mlp$Layer) +1

relevant_position <- c("Subject", "Relation", "Relation Last", "Attribute*", "Subject repeat", "Relation repeat", "Last")
n_relevant_position <- 7
data_mlp$diff_mean <- -data_mlp$diff_mean
p <- create_heatmap_base(data_mlp, "Layer", "mapped_position", "diff_mean") +
  theme_minimal() +
  #addforce to have all the labels
  scale_x_continuous(breaks = seq(1,max_layer+1,1), labels=as.character(seq(1,max_layer+1,1))) +
  scale_y_reverse(breaks = seq(0,n_relevant_position-1, 1), labels=relevant_position) +
  labs(fill = "Logit Diff", y="") +
  theme(
    axis.text.x = element_text(size=AXIS_TEXT_SIZE),
    axis.text.y = element_text(size=AXIS_TEXT_SIZE),
    #remove background grid
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.title.x = element_text(size = AXIS_TITLE_SIZE),
    axis.title.y = element_text(size = AXIS_TITLE_SIZE),
    legend.text = element_text(size = 35),
    legend.title = element_text(size = AXIS_TITLE_SIZE),
    #remove the legend\
    legend.position = "bottom",
    # increase the size of the legend
    legend.key.size = unit(3, "cm"),
    # move the y ticks to the right
  )

p
ggsave(sprintf("PaperPlot/%s_%s_logit_attribution/logit_attribution_mlp_out.pdf", model, experiment), p, width = 50, height = 50, units = "cm")

###### attn_out #######

#filter position f"{i}_mlp_out"
data_attn <- data %>% filter(grepl("^[0-9]+_attn_out$", label))
data_attn <- data_attn %>% separate(label, c("layer"), sep = "_attn_out")
max_layer <- max(as.numeric(data_attn$layer))
max_position <- max(as.numeric(data_attn$position))
#create layer column

data_attn$layer <- factor(data_attn$layer, levels = c(0:max_layer))
data_attn$position <- factor(data_attn$position, levels = c(0:max_position))

colnames(data_attn)[1] <- "Layer"
colnames(data_attn)[2] <- "Position"
data_attn <- data_attn %>% filter(Position == 1 | Position==4 | Position == 5 | Position== 6 | Position==8 |  Position == 11 | Position== 12)
unique_positions <- unique(data_attn$Position)
position_mapping <- setNames(seq(0, length(unique_positions) - 1), unique_positions)
# Apply the mapping to create a new column
data_attn$mapped_position <- unname(position_mapping[as.character(data_attn$Position)])
data_attn$Layer <- as.numeric(data_attn$Layer) -1

relevant_position <- c("Subject", "Relation", "Relation Last", "Attribute*", "Subject repeat", "Relation repeat", "Last")
n_relevant_position <- 7
data_attn$diff_mean <- -data_attn$diff_mean

p <- create_heatmap_base(data_attn, "Layer", "mapped_position", "diff_mean") +
  theme_minimal() +
  #addforce to have all the labels
  scale_x_continuous(breaks = seq(1,max_layer+1,1), labels=as.character(seq(1,max_layer+1,1))) +
  scale_y_reverse(breaks = seq(0,n_relevant_position-1, 1), labels=relevant_position) +
  labs(fill = "Logit Diff", y="") +
  theme(
    axis.text.x = element_text(size=AXIS_TEXT_SIZE),
    axis.text.y = element_text(size=AXIS_TEXT_SIZE),
    #remove background grid
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.title.x = element_text(size = AXIS_TITLE_SIZE),
    axis.title.y = element_text(size = AXIS_TITLE_SIZE),
    legend.text = element_text(size = 35),
    legend.title = element_text(size = AXIS_TITLE_SIZE),
    #remove the legend\
    legend.position = "bottom",
    # increase the size of the legend
    legend.key.size = unit(3, "cm"),
    # move the y ticks to the right
  )

p
ggsave(sprintf("PaperPlot/%s_%s_logit_attribution/logit_attribution_attn_out.pdf", model, experiment), p, width = 50, height = 50, units = "cm")



########################################################################################################################
######################################## HEAD PATTERN modificated  #####################################################
########################################################################################################################

# Load your data
experiment <- "copyVSfactNoBos"
data <- read.csv(sprintf("%s/head_pattern/%s/head_pattern_data.csv", experiment, model_folder))
create_heatmap <- function(data, x, y, fill) {
  p <- ggplot(data,aes( x=x, y=y, fill=fill)) +
    geom_tile(aes(fill=value))+
    theme_minimal() +
    #addforce to have all the labels
    scale_fill_gradient2(low = "white" , high ="#58508d", midpoint = 0) + #"#1f6f6f"
    scale_x_continuous(breaks = seq(0, length(relevant_position) - 1,1), labels = relevant_position) +
    labs(x = "", y = "", fill="Attention Score:") +
    theme(
      axis.text.x = element_text(size=60, angle = 90),
      axis.text.y = element_text(size=60, angle = 0),
      #remove background grid
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.title.x = element_text(size = 60),
      axis.title.y = element_text(size = 60),
      legend.text = element_text(size = 50),
      legend.title = element_text(size = 60),
      #remove the legend\
      legend.position = "bottom",
      legend.key.size = unit(2.5, "cm"),
    )
      # move the y ticks to the right
  return(p)
}

# Assuming your dataset is named 'data'
# Step 1: Filter for dest_position equal to 12
data_filtered <- data %>% filter(source_position == 12)

# Step 2: Filter based on specific layer and head patterns
pattern_df <- data.frame(layer = layer_pattern, head = head_pattern)

data_final <- data_filtered %>% 
  inner_join(pattern_df, by = c("layer", "head"))
# Step 3: Prepare the data for plotting
data_final$y_label <- paste("Layer ", data_final$layer, " | Head ", data_final$head, sep="")
#filter just the relevant positions
data_final <- data_final %>% filter(dest_position == 1 | dest_position==4 | dest_position == 5 | dest_position== 6 | dest_position==8 |  dest_position == 11 | dest_position== 12)
unique_positions <- unique(data_final$dest_position)
position_mapping <- setNames(seq(0, length(unique_positions) - 1), unique_positions)
# Apply the mapping to create a new column
data_final$mapped_position <- unname(position_mapping[as.character(data_final$dest_position)])
# Create and plot the heatmap
data_final <- data_final %>%
  mutate(color = ifelse((y_label =="Layer 10 | Head 7" | y_label=="Layer 11 | Head 10"), "Target", "Other")) # Add color column

library(ggnewscale) # for using new color scales within the same plot
# Your original plot for 'Other'

heatmap_plot <- ggplot(data_final %>% filter(color == "Other"), aes(x = mapped_position, y = y_label, fill = value)) +
  geom_tile(colour = "grey") +
  scale_x_continuous(breaks = seq(0, length(relevant_position) - 1,1), labels = relevant_position) +
  scale_fill_gradient(low = "white", high = "#005CAB") +
  labs(fill = "Attention\nScore:") +
  theme_minimal() +
  new_scale_fill() + # This tells ggplot to start a new fill scale
  geom_tile(data = data_final %>% filter(color == "Target"), aes(x = mapped_position, y = y_label, fill = value), colour="grey") +
  scale_fill_gradient(low = "white", high = "#E31B23") +
  scale_x_continuous(breaks = seq(0, length(relevant_position) - 1,1), labels = relevant_position) +
  labs(fill = "Attention\nScore:") +
  theme(
    axis.text.x = element_text(size=60, angle = 90),
    axis.text.y = element_text(size=60, angle = 0),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    legend.text = element_text(size = 45),
    legend.title = element_text(size = 50),
    legend.position = "right",
    legend.key.size = unit(1.3, "cm"),
  )
  
# dummy_data <- data.frame(value = seq(min(data_final$value), max(data_final$value), length.out = 100))
# dummy_plot <- ggplot(dummy_data, aes(x = value, y = 1, fill = value)) +
#   geom_tile() +
#   scale_fill_gradient(low = "white", high = "black", name = "Intensity") +
#   theme_void() + 
#   theme(legend.position = "bottom")

# Plot the heatmap
heatmap_plot

# Display the dummy plot to show the legend separately
# This can be used to manually adjust the legend in your presentation or report
dummy_plot


experiment <- "copyVSfact"
ggsave(sprintf("PaperPlot/%s_%s_heads_pattern/head_pattern_layer.pdf", model, experiment), heatmap_plot, width = 53, height = 38, units = "cm")

##################################################################################################################
########################################### ABLATION #############################################################
##################################################################################################################
create_heatmap <- function(data, x, y, fill, head=FALSE) {
  if(head){
    scale_x <- scale_x_discrete(breaks = seq(0,n_layers,1)) 
    angle = 0
  } else {
    scale_x <- scale_x_discrete(breaks = seq(0, n_positions,1), labels = positions_name)
    angle = 90
  }
  print(n_positions)
  p <- create_heatmap_base(data, x, y, fill, midpoint = -4.10) +
    theme_minimal() +
    #addforce to have all the labels
    scale_y_discrete(breaks = seq(0,n_layers,1)) +
    scale_x +
    #labs(fill = "% difference") +
    theme(
      axis.text.x = element_text(size=AXIS_TEXT_SIZE, angle = angle),
      axis.text.y = element_text(size=AXIS_TEXT_SIZE),
      #remove background grid
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.title.x = element_text(size = AXIS_TITLE_SIZE),
      axis.title.y = element_text(size = AXIS_TITLE_SIZE),
      legend.text = element_text(size = 30),
      legend.title = element_text(size = AXIS_TITLE_SIZE),
      #remove the legend\
      legend.position = "none",
      # move the y ticks to the right
    )
  return(p)
}
############################ mlp_out ########################################
mlp_out <- function(data) {
  # base_diff <- data %>% filter(component == "base")
  # base_diff <- base_diff %>% select(diff)
  # base_diff <- base_diff[1,1]
  
  data_mlp_out <- data %>% filter(grepl("mlp_out", component))
  max_layer <- max(as.numeric(data_mlp_out$layer))
  max_position <- max(as.numeric(data_mlp_out$position))
  data_mlp_out$layer <- factor(data_mlp_out$layer, levels = c(0:max_layer))
  data_mlp_out$position <- factor(data_mlp_out$position, levels = c(0:max_position))
  
  #base_diff as the avarage across all the layers and positions
  # base_diff <- mean(data_mlp_out$diff)
  # 
  # data_mlp_out$diff <- (base_diff - data_mlp_out$diff)
  
  ####### mem
  #  p_mem <- create_heatmap(data_mlp_out, "position", "layer", "mem", "MLP out ablation - mem")
  ###### cp
  #  p_cp <- create_heatmap(data_mlp_out, "position", "layer", "cp", "MLP out ablation - cp")
  ###### mem - cp
  colnames(data_mlp_out)[colnames(data_mlp_out) == "position"] <- "Position"
  colnames(data_mlp_out)[colnames(data_mlp_out) == "layer"] <- "Layer"
  p <- create_heatmap(data_mlp_out, "Position", "Layer", "diff")
  ggsave(sprintf("PaperPlot/%s_%s_ablation/mlp_out.pdf", model, experiment), p, width = 50, height = 50, units = "cm")
}
############################ attn_out ########################################
attn_out <- function(data) {
  data_attn_out <- data %>% filter(grepl("attn_out", component))
  max_layer <- max(as.numeric(data_attn_out$layer))
  max_position <- max(as.numeric(data_attn_out$position))
  data_attn_out$layer <- factor(data_attn_out$layer, levels = c(0:max_layer))
  data_attn_out$position <- factor(data_attn_out$position, levels = c(0:max_position))
  # base_diff <- mean(data_attn_out$diff)
  # data_attn_out$diff <- ( data_attn_out$diff )
  ####### mem
  # p_mem <- create_heatmap(data_attn_out, "position", "layer", "mem", "Attn out ablation - mem")
  #p_cp <- create_heatmap(data_attn_out, "position", "layer", "cp", "Attn out ablation - cp")
  colnames(data_attn_out)[colnames(data_attn_out) == "position"] <- "Position"
  colnames(data_attn_out)[colnames(data_attn_out) == "layer"] <- "Layer"
  p <- create_heatmap(data_attn_out, "Position", "Layer", "diff")
  ggsave(sprintf("PaperPlot/%s_%s_ablation/attn_out.pdf", model, experiment), p, width = 50, height = 50, units = "cm")
}

############################## HEAD ##########################################
head <- function(data) {
  data_head <- data %>% filter(grepl("head", component))
  max_layer <- max(as.numeric(data_head$layer))
  max_head <- max(as.numeric(data_head$head))
  data_head$layer <- factor(data_head$layer, levels = c(0:max_layer))
  data_head$head <- factor(data_head$head, levels = c(0:max_head))
  base_diff <- mean(data_head$diff)
  data_head$diff <- (base_diff - data_head$diff)
  #p_mem <- create_heatmap(data_head, "head", "layer", "mem", "Head ablation - mem")
  #p_cp <- create_heatmap(data_head, "head", "layer", "cp", "Head ablation - cp")
  colnames(data_head)[colnames(data_head) == "head"] <- "Head"
  colnames(data_head)[colnames(data_head) == "layer"] <- "Layer"
  p <- create_heatmap(data_head, "Head", "Layer", "diff", TRUE)
  ggsave(sprintf("PaperPlot/%s_%s_ablation/head.pdf", model, experiment), p, width = 50, height = 50, units = "cm")
}



folder_name <- sprintf("~/Repository/Competition_of_Mechanisms/results/%s/ablation/%s_total_effect", experiment, model_folder_abl)
files <- list.files(path = folder_name, pattern = "*.csv", full.names = FALSE)
print(files)
#check if "ablation_data.csv" is in the list
if (!("ablation_data.csv" %in% files)) {
  if ("ablation_data_attn_out.csv" %in% files) {
    data <- read.csv(paste(folder_name, "ablation_data_attn_out.csv", sep = "/"))
    attn_out(data)
  }
  if ("ablation_data_mlp_out.csv" %in% files) {
    data <- read.csv(paste(folder_name, "ablation_data_mlp_out.csv", sep = "/"))
    mlp_out(data)
  }
  if ("ablation_data_head.csv" %in% files) {
    data <- read.csv(paste(folder_name, "ablation_data_head.csv", sep = "/"))
    head(data)
  }
  if ("ablation_data_resid_pre.csv" %in% files) {
    data <- read.csv(paste(folder_name, "ablation_data_resid_pre.csv", sep = "/"))
    resid_pre(data)
  }
} else {
  data <- read.csv(paste(folder_name, "ablation_data.csv", sep = "/"))
  mlp_out(data)
  attn_out(data)
  head(data)
  resid_pre(data)
}

############################################################################################################
########################################## SELF-SIMILARITY #############################################
############################################################################################################


# Load the DataFrame
experiment <- "copyVSfact"
originaldf <- read_csv(sprintf("%s/evaluate_mechanism_fix_partition.csv", experiment))
#originaldf <- read_csv(sprintf("%s/gpt2_evaluate_mechanism_ss_fixed_partition.csv", experiment))

# Filtering and calculating percentages for self-similarity
df <- originaldf %>% 
  filter(similarity_type == "self-similarity") %>%
  mutate(percentage_true = target_true / (target_true + target_false + other) * 100) %>%
  group_by(model_name, interval) %>%
  summarise(
    percentage_true = mean(percentage_true, na.rm = TRUE),
    n = n(),
    target_true_sum = sum(target_true),
    total = sum(target_true + target_false + other)
  ) %>%
  mutate(ci_lower = binom.confint(target_true_sum, total, methods = "exact")$lower * 100,
         ci_upper = binom.confint(target_true_sum, total, methods = "exact")$upper * 100)

# Calculating percentages for original similarity_type
basedf <- originaldf %>% 
  filter(similarity_type == "original") %>%
  mutate(percentage_true = target_true / (target_true + target_false + other) * 100) %>%
  group_by(model_name) %>%
  summarise(base_percentage = mean(percentage_true, na.rm = TRUE))

# Merging the base percentages with the main DataFrame
df <- df %>% 
  left_join(basedf, by = "model_name")

df <- df %>%
  mutate(model_name = case_when(
    model_name == "gpt2" ~ "GPT2",
    model_name == "gpt2-large" ~ "GPT2-large",
    model_name == "gpt2-medium" ~ "GPT2-medium",
    model_name == "gpt2-xl" ~ "GPT2-xl",
    model_name == "EleutherAI/pythia-6.9b" ~ "Pythia-6.9b",
    TRUE ~ model_name # Keeps other model names unchanged
  ))

# Assuming 'df' is your dataframe and 'interval' is the column to be transformed
percentile_labels <- c("0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%")
df$percentile_interval <- factor(df$interval, labels = percentile_labels)

palette <- c("GPT2" = "#003f5c", "GPT2-medium" = "#58508d", "GPT2-large" = "#bc5090", "GPT2-xl" = "#ff6361", "Pythia-6.9b" = "#ffa600")
# Now, plotting with the updated dataframe
p<-ggplot() +
  geom_line(data = df, aes(x = percentile_interval, y = percentage_true, group = model_name, color = model_name), size=1.1) +
  geom_point(data = df, aes(x = percentile_interval, y = percentage_true, group = model_name, color = model_name), size=2.3) +
  geom_line(data = df, aes(x = percentile_interval, y = base_percentage, group = model_name, color = model_name), linetype = "dotted",  size=1.1) +
  scale_color_manual(values = palette) +
  labs(x = "Similarity Score Bins (Percentiles)",
       y = "Percentage of Factual Recalling",
       color = "Model:",
       linetype = "") +
  scale_linetype_manual(values = c("Base Value" = "dotted")) + # Ensure "Base Value" is dotted
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size=20),
        axis.text.y= element_text(size=20),
        legend.title = element_blank(),
        legend.text = element_text(size = 20),
        axis.title = element_text(size = 23),
        legend.position = "bottom",
        legend.box = "horizontal",
        ) +
        guides(color = guide_legend(nrow = 3, title.position = "top", title.hjust = 0.5),
                                                                                              linetype = guide_legend(nrow = 1, title.position = "top", title.hjust = 0.5))
  #save plot
ggsave("PaperPlot/copyVSfact_self_similarity.pdf",p, width = 16, height = 21, units = "cm")












############################################### OLD OLD OLD OLD ###########################################################
# 
# ############################################################################################################
# ######################################## HEAD PATTERN  #####################################################
# ############################################################################################################
# 
# create_heatmap <- function(data, x, y, fill, head=FALSE) {
#   p <- create_heatmap_base(data, x, y, fill) +
#     theme_minimal() +
#     #addforce to have all the labels
#     scale_y_discrete(breaks = seq(0, n_positions,1), labels = positions_name) +
#     scale_x_discrete(breaks = seq(0, n_positions,1), labels = positions_name) +
#     #labs(fill = "% difference") +
#     theme(
#       axis.text.x = element_text(size=AXIS_TEXT_SIZE, angle = 90),
#       axis.text.y = element_text(size=AXIS_TEXT_SIZE),
#       #remove background grid
#       panel.grid.major = element_blank(),
#       panel.grid.minor = element_blank(),
#       axis.title.x = element_text(size = AXIS_TITLE_SIZE),
#       axis.title.y = element_text(size = AXIS_TITLE_SIZE),
#       legend.text = element_text(size = 30),
#       legend.title = element_text(size = 50),
#       #remove the legend\
#       legend.position = "none",
#       # move the y ticks to the right
#     )
#   return(p)
# }
# 
# 
# plot_pattern <- function(l, h, data){
#   selected_layer <- l
#   selected_head <- h
#   data_head <- data %>% filter(layer == selected_layer & head == selected_head)
#   max_source_position <- max(as.numeric(data_head$source_position))
#   max_dest_position <- max(as.numeric(data_head$dest_position))
#   data_head$source_position <- factor(data_head$source_position, levels = c(0:max_source_position))
#   data_head$dest_position <- factor(data_head$dest_position, levels = c(0:max_dest_position))
#   
#   #reorder the source_position and dest_position contrary to the order of the factor
#   data_head$source_position <- factor(data_head$source_position, levels = rev(levels(data_head$source_position)))
#   
#   p <- create_heatmap(data_head, "dest_position", "source_position", "value", paste("Layer", selected_layer, "Head", selected_head, sep = " "))
#   return(p)
# }
# 
# 
# data <- read.csv(sprintf("%s/head_pattern/%s/head_pattern_data.csv", experiment, model_folder))
# #data <- read.csv("head_pattern_data.csv")
# 
# number_of_layers <- max(as.numeric(data$layer))
# number_of_heads <- max(as.numeric(data$head))
# for (i in c(1:4)){
#     l <- layer_pattern[i]
#     h <- head_pattern[i]
#     p <- plot_pattern(l, h, data)
#     ggsave(sprintf("PaperPlot/%s_%s_heads_pattern/head_pattern_layer_%s_head_%s.pdf", model, experiment, l,h), p, width = 50, height = 50, units = "cm")
# }