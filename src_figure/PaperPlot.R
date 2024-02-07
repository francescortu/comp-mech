library(ggplot2)
library(readr)
library(dplyr)
library(binom)
setwd("/home/francesco/Repository/Competition_of_Mechanisms/results")



################################## EXPERIMENT CONFIGS ########################################
experiment <- "copyVSfact"
n_positions <- 12
positions_name <- c("-", "1st subject", "2nd subject", "3rd subject", "-", "last", "object", "-", "1st subject", "2nd subject", "3nd subject", "-", "last")
#GPT2
layer_pattern <- c(11,10,10,10)
head_pattern <- c(10,0,7,10)
#Pythia-6.9b
#layer_pattern <- c(21,20,20,19)
#head_pattern <- c(8,2,18,31)
  
#experiment <- "contextVSfact"

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
create_heatmap_base <- function(data, x, y, fill) {
  # Convert strings to symbols for tidy evaluation
  x_sym <- rlang::sym(x)
  y_sym <- rlang::sym(y)
  fill_sym <- rlang::sym(fill)
  
  p<- ggplot(data, aes(!!x_sym, !!y_sym, fill = !!fill_sym)) +
    geom_tile() +
    scale_fill_gradient2(low = "#1a80bb", mid = "white", high = "#a00000", midpoint = 0) +
    theme(axis.text.x = element_text(angle = 0, vjust = 0.5, hjust=1)) +
    labs(x = x, y = y) +
    geom_text(aes(label = sprintf("%.2f", !!fill_sym)), color = "black", size = HEATMAP_SIZE)
  return(p)
}

#################################################################################################################
##################################### LOGIT LENS ################################################################
#################################################################################################################



create_heatmap <- function(data, x, y, fill, high_color) {
  p <- create_heatmap_base(data, x, y, fill) +
    scale_fill_gradient2(low = "black", mid = "white", high = high_color, midpoint = 0) +
    theme_minimal() +
    #addforce to have all the labels
    scale_y_continuous(breaks = seq(0,n_layers,1)) +
    scale_x_continuous(breaks = seq(0,n_positions,1), labels = positions_name) +
    labs(x = "Position", y = "Layer")+
    theme(
      axis.text.x = element_text(size=AXIS_TEXT_SIZE, angle = 90),
      axis.text.y = element_text(size=AXIS_TEXT_SIZE),
      #remove background grid
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.title.x = element_text(size = AXIS_TITLE_SIZE),
      axis.title.y = element_text(size = AXIS_TITLE_SIZE),
      legend.text = element_text(size = 30),
      legend.title = element_text(size = 50),
      #remove the legend\
      legend.position = "none",
      # move the y ticks to the right
    )
  return(p)
}


data <- read.csv(sprintf("%s/logit_lens/%s/logit_lens_data.csv", experiment, model_folder))
number_of_position <- max(as.numeric(data$position))
data_resid_post <- data %>% filter(grepl("resid_post", component))
p <- create_heatmap(data_resid_post, "position", "layer", "mem",  "#E31B23")
ggsave(sprintf("PaperPlot/%s_%s_residual_stream/resid_post_mem.pdf", model, experiment), p, width = 50, height = 50, units = "cm")

p <- create_heatmap(data_resid_post, "position", "layer", "cp",  "#005CAB")
ggsave(sprintf("PaperPlot/%s_%s_residual_stream/resid_post_cp.pdf", model, experiment), p, width = 50, height = 50, units = "cm")



############################################################################################################
######################################## LOGIT ATTRIBUTION ################################################
############################################################################################################
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
p <- create_heatmap(data_head_, "Head", "Layer", "diff_mean", head=TRUE)
ggsave(sprintf("PaperPlot/%s_%s_logit_attribution/logit_attribution_head_position%s.pdf", model, experiment, number_of_position), p, width = 50, height = 50, units = "cm")

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
p <- create_heatmap(data_mlp, "Position", "Layer", "diff_mean", FALSE)
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
p <- create_heatmap(data_attn, "Position", "Layer", "diff_mean")
ggsave(sprintf("PaperPlot/%s_%s_logit_attribution/logit_attribution_attn_out.pdf", model, experiment), p, width = 50, height = 50, units = "cm")


############################################################################################################
######################################## HEAD PATTERN  #####################################################
############################################################################################################

create_heatmap <- function(data, x, y, fill, head=FALSE) {
  p <- create_heatmap_base(data, x, y, fill) +
    theme_minimal() +
    #addforce to have all the labels
    scale_y_discrete(breaks = seq(0, n_positions,1), labels = positions_name) +
    scale_x_discrete(breaks = seq(0, n_positions,1), labels = positions_name) +
    #labs(fill = "% difference") +
    theme(
      axis.text.x = element_text(size=AXIS_TEXT_SIZE, angle = 90),
      axis.text.y = element_text(size=AXIS_TEXT_SIZE),
      #remove background grid
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.title.x = element_text(size = AXIS_TITLE_SIZE),
      axis.title.y = element_text(size = AXIS_TITLE_SIZE),
      legend.text = element_text(size = 30),
      legend.title = element_text(size = 50),
      #remove the legend\
      legend.position = "none",
      # move the y ticks to the right
    )
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


data <- read.csv(sprintf("%s/head_pattern/%s/head_pattern_data.csv", experiment, model_folder))
#data <- read.csv("head_pattern_data.csv")

number_of_layers <- max(as.numeric(data$layer))
number_of_heads <- max(as.numeric(data$head))
for (i in c(1:4)){
    l <- layer_pattern[i]
    h <- head_pattern[i]
    p <- plot_pattern(l, h, data)
    ggsave(sprintf("PaperPlot/%s_%s_heads_pattern/head_pattern_layer_%s_head_%s.pdf", model, experiment, l,h), p, width = 50, height = 50, units = "cm")
}


########################################################################################################################
######################################## HEAD PATTERN modificated  #####################################################
########################################################################################################################

# Load your data
data <- read.csv(sprintf("%s/head_pattern/%s/head_pattern_data.csv", experiment, model_folder))
create_heatmap <- function(data, x, y, fill) {
  p <- create_heatmap_base(data, x, y, fill) +
    theme_minimal() +
    #addforce to have all the labels
    scale_fill_gradient2(low = "lightgray" , high = "#1f6f6f", midpoint = 0) +
    scale_x_continuous(breaks = seq(0, n_positions,1), labels = positions_name) +
    labs(x = "Position", y = "Heads") +
    theme(
      axis.text.x = element_text(size=30, angle = 90),
      axis.text.y = element_text(size=30),
      #remove background grid
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.title.x = element_text(size = 50),
      axis.title.y = element_text(size = 50),
      legend.text = element_text(size = 30),
      legend.title = element_text(size = 50),
      #remove the legend\
      legend.position = "none",
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
data_final$y_label <- paste("Layer", data_final$layer, "| Head", data_final$head)

# Create and plot the heatmap
heatmap_plot <- create_heatmap(data_final, "dest_position", "y_label", "value")
ggsave(sprintf("PaperPlot/%s_%s_heads_pattern/head_pattern_layer.pdf", model, experiment), heatmap_plot, width = 50, height = 20, units = "cm")

##################################################################################################################
########################################### ABLATION #############################################################
##################################################################################################################



############################################################################################################
########################################## SELF-SIMILARITY #############################################
############################################################################################################


# Load the DataFrame
originaldf <- read_csv(sprintf("%s/evaluate_mechanism_fix_partition.csv", experiment))

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

# Now, plotting with the updated dataframe
ggplot() +
  geom_line(data = df, aes(x = percentile_interval, y = percentage_true, group = model_name, color = model_name), size=0.8) +
  geom_point(data = df, aes(x = percentile_interval, y = percentage_true, group = model_name, color = model_name), size=2.3) +
  geom_line(data = df, aes(x = percentile_interval, y = base_percentage, group = model_name, color = model_name), linetype = "dotted",  size=0.8) +
  labs(x = "Similarity Score Bins (Percentiles)",
       y = "Percentage of Factual Recalling",
       color = "Model:",
       linetype = "") +
  scale_linetype_manual(values = c("Base Value" = "dotted")) + # Ensure "Base Value" is dotted
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size=12),
        axis.text.y= element_text(size=12),
        legend.title = element_text(size = 15),
        legend.text = element_text(size = 10),
        axis.title = element_text(size = 15),
        legend.position = "bottom",
        legend.box = "horizontal",
        ) +
        guides(color = guide_legend(nrow = 2, title.position = "top", title.hjust = 0.5),
                                                                                              linetype = guide_legend(nrow = 1, title.position = "top", title.hjust = 0.5))
  #save plot
ggsave("results/PaperPlot/copyVSfact_self_similarity.pdf", width = 14, height = 18, units = "cm")
