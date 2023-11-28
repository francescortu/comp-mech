library(ggplot2)
library(dplyr)
library(tidyr)
library(RColorBrewer)
library(scales) # For color manipulation
library(plotwidgets)
library(colorspace)
data <- read.csv("opt_evaluate_mechanism.csv")


# Calculate percentages
data <- data %>%
  mutate(total = target_true + target_false + other) %>%
  mutate(target_true_perc = (target_true / total) * 100,
         target_false_perc = (target_false / total) * 100,
         other_perc = (other / total) * 100,
         target_true_std_perc = ifelse(orthogonalize, target_true_std / total * 100, 0),
         target_false_std_perc = ifelse(orthogonalize, target_false_std / total * 100, 0),
         other_std_perc = ifelse(orthogonalize, other_std / total * 100, 0))

# Reshape the data to long format for ggplot
data_long <- data %>%
  select(model_name, orthogonalize, target_false_perc, target_true_perc, other_perc) %>%
  gather(key = "prediction", value = "percentage", target_false_perc:other_perc) %>%
  mutate(orthogonalize = ifelse(orthogonalize, "True", "False"),
         model_name = factor(model_name, levels = c("facebook/opt-2.7B","facebook/opt-1.3B", "facebook/opt-350m",  "facebook/opt-125m")),
         prediction = factor(prediction, levels = c("target_false_perc", "target_true_perc", "other_perc")))

# Define base colors for each model

# Create the plot with flipped coordinates and custom colors
ggplot(data_long, aes(fill=interaction(prediction, orthogonalize), x=model_name, y=percentage)) + 
  geom_bar(position="dodge", stat="identity") +
  coord_flip() +
  facet_wrap(~prediction, ncol = 1) +
  scale_fill_manual(values = c("#92ecd2","#f28cc0","#afacd3","#136d52","#730d41","#2f2c53")) +
  theme_minimal() +
  labs(title = "Comparison of Model Predictions by Category",
       y = "Model",
       x = "Percentage",
       fill = "Model and Orthogonalization") +
  theme(axis.text.y = element_text(angle = 0, hjust = 1))

############################################
# Calculate percentages
data <- read.csv("results/plot_data/gpt2_count.csv")
data$model[1] <- "gpt2-small"
data$model[2] <- "gpt2-medium"
data$model[3] <- "gpt2-large"
data$model[4] <- "gpt2-xl"
data$model <- factor(data$model, levels=c("gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"))
data <- data %>%
  mutate(total = target_true + target_false + other) %>%
  mutate(target_true_perc = (target_true / total) * 100,
         target_false_perc = (target_false / total) * 100,
         other_perc = (other / total) * 100,
         # Conditionally assign standard deviation percentages
         target_true_std_perc = ifelse(orthogonalize, (target_true_std / total) * 100, 0),
         target_false_std_perc = ifelse(orthogonalize, (target_false_std / total) * 100, 0),
         other_std_perc = ifelse(orthogonalize, (other_std / total) * 100, 0))

# Reshape the data to long format for ggplot
data_long <- data %>%
  select(model_name, orthogonalize, target_false_perc, target_true_perc, other_perc) %>%
  gather(key = "prediction", value = "percentage", target_false_perc:other_perc) %>%
  mutate(orthogonalize = ifelse(orthogonalize, "True", "False"),
         model_name = factor(model_name, levels = c("gpt2-xl","gpt2-large", "gpt2-medium",  "gpt2")),
         prediction = factor(prediction, levels = c("target_false_perc", "target_true_perc", "other_perc")))

# Define base colors for each model
#for row in data_long:
std_perc_mapping <- c("target_true_perc" = "target_true_std_perc",
                      "target_false_perc" = "target_false_std_perc",
                      "other_perc" = "other_std_perc")

# Step 2: Join data_long with the necessary columns of data
data_long <- left_join(data_long, 
                       select(data, model_name, orthogonalize, 
                              target_true_std_perc, target_false_std_perc, other_std_perc),
                       by = c("model_name", "orthogonalize"))

# Step 3: Calculate the std_percentage for each row in data_long
data_long <- data_long %>%
  mutate(std_percentage = case_when(
    prediction == "target_true_perc" ~ target_true_std_perc,
    prediction == "target_false_perc" ~ target_false_std_perc,
    prediction == "other_perc" ~ other_std_perc
  ))
data_long <- data_long %>%
  select(model_name, orthogonalize, prediction, percentage, std_percentage)
# Create the plot with flipped coordinates and custom colors
# Define custom colors and legend labels
legend_colors <- c("True" = "#003554", "False" = "#afacd3")
legend_colors <- c("True" = "#003554", "False" = "darkgrey")
legend_labels <- c(paste0("Modified altered token"), "Default" )
library(ggpattern)
data_long$pattern_type <- ifelse(data_long$orthogonalize, "stripe", "none")
data_long$model_name <- factor(data_long$model_name, levels = c("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"))
ggplot(data_long, aes(fill = interaction(prediction, orthogonalize), x = model_name, y = percentage)) + 
  geom_bar_pattern(aes(pattern_density=orthogonalize), position = "dodge", stat = "identity", colour="black", size=0.2) +
  geom_linerange(aes(ymin = percentage - std_percentage, ymax = percentage + std_percentage), 
                 width = .2, position = position_dodge(.9), col = "black", size=1.2) +
  coord_flip() +
  facet_wrap(~prediction, ncol = 1, labeller = labeller(prediction = c(target_true_perc = "Factual token", 
                                                                       target_false_perc = "Altered token", 
                                                                       other_perc = "Other token"))) +
  scale_fill_manual(values = c("#FF725C","#96CCFF","#9EEBCF","#FF725C","#96CCFF","#9EEBCF")) +

  #scale_fill_manual(values = c("True" = "#136d52", "False" = "#afacd3")) +  # Adjust colors as needed
  theme_minimal() +
  labs(y = "Count (%)",
       x = "Model",
       pattern_density = "Replace attribute token") +  # Update the legend title
  theme(axis.text.y = element_text(size=15, angle = 0, hjust = 1),
        axis.text.x = element_text(size=15, angle = 90), # Increase x-axis label size and angle
        axis.title.x = element_text(size = 20), # Increase x-axis label size and angle
        axis.title.y = element_text(size = 20), # Increase y-axis label size
        title = element_text(size=20),
        strip.text = element_text(size = 20),
        #increase legend text size
        legend.text = element_text(size=15),
        legend.position = "bottom",
        )+
  # Create a fake color scale for the 'orthogonalize' legend
  guides(fill = "none")   # Set legend labels
ggsave("plots/score_models.pdf", width = 10, height = 20, units = "in", dpi = 300)


##########################################################################################

# reorder based on specific order of models
data$model[1] <- "gpt2-small"
data$model[2] <- "gpt2-medium"
data$model[3] <- "gpt2-large"
data$model[4] <- "gpt2-xl"
data$model <- factor(data$model, levels=c("gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"))
#barplot
ggplot(data=data) + 
  geom_bar(data=data, aes(y=model, x=target_true, fill=model), stat="identity")+
  scale_fill_manual(values=c("#56641a", "#e6a176","#00678a", "#984464"))+
  labs(x = "% Factual Recalling", y = "Model")+
  theme_minimal() +  # Minimal theme  
  theme(
    panel.grid.major = element_blank(), 
    #panel.grid.minor = element_blank(),
    legend.position = "none",
    legend.title = element_text(size=20),
    legend.text = element_text(size=30),
    legend.key.size = unit(2, "cm"),
    axis.text.x = element_text(size=30, angle = 90), # Increase x-axis label size and angle
    axis.text.y = element_text(size=30), # Increase y-axis label size
    axis.title.x = element_text(size=40), # Increase x-axis label size and angle
    axis.title.y = element_text(size=40), # Increase y-axis label size
  )
ggsave("plots/fig1_barplot.pdf", width = 20, height = 10)
  
  
  