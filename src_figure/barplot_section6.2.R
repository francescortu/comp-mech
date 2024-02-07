library(ggplot2)
library(readr)
library(dplyr)

# Load the DataFrame
df <- read_csv("gpt2_evaluate_mechanism_NEW.csv")


#filter per similarity type
df <- df %>% filter(similarity_type == "word2vec")
df <- df %>% filter(premise == "Redefine")
df <- df %>% filter(orthogonalize == "TRUE")
#df <- df %>% filter(interval != 0)
data <- df %>%
      group_by(model_name, interval) %>%
  summarize(total_true = sum(target_true), 
            total_true_std = mean(target_true_std),
            .groups = 'drop') %>%
  mutate(percentage_true = total_true/10000*100, 
         percentage_true_std = total_true_std/10000*100) 

# Convert interval to a factor
data$interval <- as.factor(data$interval)
data$model_name <- factor(data$model_name, levels = c("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/pythia-6.9b"))

# ggplot(data, aes(x=model_name, y= percentage_true, fill=interval)) +
#   geom_bar(stat = "identity", position ="dodge", size=0.5, colour="black")+
#   coord_flip() +
#   scale_fill_manual(values = c("#999999", "#829cbc","#6290c8","#376996","#1f487e"), name="Similarity level:",
#                     labels = c("Original", "Level 1 (most similar)", "Level 2", "Level 3", "Level 4 (less similar)")) +
#   theme_minimal()+
#   labs(y="Percentage of cases where Mechanism 1 wins (%)",
#        x="",
#        legend="ciao")+
#   theme(axis.text.y = element_text(size=80, angle = 0, hjust = 1),
#         axis.text.x = element_text(size=70, angle = 0), # Increase x-axis label size and angle
#         axis.title.x = element_text(size = 80), # Increase x-axis label size and angle
#         axis.title.y = element_text(size = 0), # Increase y-axis label size
#         title = element_text(size=20),
#         strip.text = element_text(size = 20),
#         #increase legend text size
#         legend.text = element_text(size=80),
#         legend.title = element_text(size=90),
#         legend.position = "bottom",
#   )+
#   guides(fill = guide_legend(ncol = 1))
df <- read_csv("gpt2_evaluate_mechanism.csv")
original_data <- df %>%
  filter(similarity_type == "original") %>%
  group_by(model_name, interval) %>%
  summarize(total_true = sum(target_true), .groups = 'drop') %>%
  mutate(percentage_true = total_true / 10000 * 100)

  
p <-  ggplot(data, aes(x = as.factor(interval), y = percentage_true, group = model_name, color = model_name)) +
    geom_line() +
    geom_point() +
    geom_errorbar(aes(ymin = percentage_true - percentage_true_std, 
                      ymax = percentage_true + percentage_true_std), width = 0.2) +
    theme_minimal() +
    labs(title = "Word2vec Mechanism strength across Models",
         x = "Similarity Interval (1 = most similar, 4 = less similar)",
         y = "Percentage True",
         color = "Model Name") +
    # change 1 tick mark at a time
    scale_color_brewer(palette = "Set1")
 for(i in 1:nrow(original_data)) {
  p <- p + geom_hline(yintercept = original_data$percentage_true[i], linetype = "dashed",
                      color = scales::hue_pal()(nrow(original_data))[i], alpha = 0.5)
 }
p

  ggsave("PaperPlot/score_model_w2v.pdf", width = 7, height = 4, units = "in") 
  
  
  