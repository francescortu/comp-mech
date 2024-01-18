library(ggplot2)
library(readr)
library(dplyr)

# Load the DataFrame
df <- read_csv("gpt2_evaluate_mechanism.csv")

#filter per similarity type
df <- df %>% filter(similarity_type == "word2vec")
df <- df %>% filter(premise == "Redefine")
data <- df %>%
      group_by(model_name, interval) %>%
  summarize(total_true = sum(target_true), .groups = 'drop') %>%
  mutate(percentage_true = total_true/10000*100) 

# Convert interval to a factor
data$interval <- as.factor(data$interval)
data$model_name <- factor(data$model_name, levels = c("GPT2-small", "GPT2-medium", "GPT2-large", "GPT2-xl"))

ggplot(data, aes(x=model_name, y= percentage_true, fill=interval)) +
  geom_bar(stat = "identity", position ="dodge", size=0.5, colour="black")+
  coord_flip() +
  scale_fill_manual(values = c("#999999", "#829cbc","#6290c8","#376996","#1f487e"), name="Similarity level:",
                    labels = c("Original", "Level 1 (most similar)", "Level 2", "Level 3", "Level 4 (less similar)")) +
  theme_minimal()+
  labs(y="Percentage of cases where Mechanism 1 wins (%)",
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
  ggsave("PaperPlot/score_model_w2v.pdf", width = 37, height = 45, units = "in") 
  
  