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
model <- "pythia-6.9b"
model_folder <- "pythia-6.9b"
n_layers <- 12


# model <- "pythia-6.9b"
# model_folder <- "pythia-6.9b_full"
# n_layers <- 32


############################### PLOT CONFIGS ##################################################
AXIS_TITLE_SIZE <- 60
AXIS_TEXT_SIZE <- 40
HEATMAP_SIZE <- 10

# Load the DataFrame
experiment <- "copyVSfact_prova"
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
df$interval <- max(df$interval) - df$interval 
palette <- c("GPT2" = "#003f5c", "GPT2-medium" = "#58508d", "GPT2-large" = "#bc5090", "GPT2-xl" = "#ff6361", "Pythia-6.9b" = "#ffa600")
# Now, plotting with the updated dataframe
p<-ggplot() +geom_line(data = df, aes(x = interval, y = percentage_true, group = model_name, color = model_name), size=1.1) +
  geom_point(data = df, aes(x = interval, y = percentage_true, group = model_name, color = model_name), size=2.3) +
  geom_line(data = df, aes(x = interval, y = base_percentage, group = model_name, color = model_name), linetype = "dotted",  size=1.1) +
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
  ) +  guides(color = guide_legend(nrow = 3, title.position = "top", title.hjust = 0.5))
         
p
