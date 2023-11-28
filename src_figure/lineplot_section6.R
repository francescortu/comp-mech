library(ggplot2)
library(readr)
library(dplyr)

# Load the DataFrame
df <- read_csv("results/gpt2_evaluate_mechanism.csv")


data <- df %>%
      group_by(model_name, interval) %>%
  summarize(total_true = sum(target_true), .groups = 'drop') %>%
  mutate(percentage_true = total_true/sum(total_true)*100) 

# Convert interval to a factor
data$interval <- as.factor(data$interval)
data$model_name <- factor(data$model_name, levels = c("GPT2-small", "GPT2-medium", "GPT2-large", "GPT2-xl"))

ggplot(data, aes(x=model_name, y= percentage_true, fill=interval)) +
  geom_bar(stat = "identity", position ="dodge", size=0.5, colour="black")+
  coord_flip() +
  scale_fill_manual(values = c("#999999", "#FD816D","#D8689C","#B34ECB","#8E35FA"), name="Similarity level:",
                    labels = c("Original", "Level 0 (less similar)", "Level 1", "Level 2", "Level 3 (most similar)")) +
  theme_minimal()+
  labs(y="Number of factual prediction (%)",
       x="",
       legend="ciao")+
  theme(axis.text.y = element_text(size=80, angle = 0, hjust = 1),
        axis.text.x = element_text(size=70, angle = 0), # Increase x-axis label size and angle
        axis.title.x = element_text(size = 80), # Increase x-axis label size and angle
        axis.title.y = element_text(size = 0), # Increase y-axis label size
        title = element_text(size=20),
        strip.text = element_text(size = 20),
        #increase legend text size
        legend.text = element_text(size=80),
        legend.title = element_text(size=90),
        legend.position = "bottom",
  )+
  guides(fill = guide_legend(ncol = 1))
  ggsave("results/plots/score_model.pdf", width = 32, height = 45, units = "in") 
  
  