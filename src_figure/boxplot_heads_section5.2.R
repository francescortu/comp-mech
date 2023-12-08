library(ggplot2)
library(dplyr)
library(tidyr)
library(latex2exp)
library(gridExtra)

df_head <- read.csv("results/plot_data/gpt2_df_head.csv") #example, head, cp_value, mem_value
# keep the 20 heads with a lowest cp_value mean across all examples
head_select <- df_head %>%
  group_by(head) %>%
  summarise(cp_value = mean(cp_value)) %>%
  arrange(cp_value) %>%
  head(20)
head_means <- df_head %>%
  group_by(head) %>%
  summarise(mean_cp_value = mean(cp_value)) %>%
  arrange(mean_cp_value) %>%
  head(20)

# Join with the original data to get the cp_value for each of the top 20 heads
head_select_data <- df_head %>%
  filter(head %in% head_means$head)

# Convert head to a factor and order it based on the mean cp_value
head_select_data$head <- factor(head_select_data$head, levels = head_means$head)

# Plot the boxplots with ordered heads
cp_plot <- ggplot(head_select_data) +
  geom_boxplot(aes(x = head, y = cp_value), color="#0f0f0f", fill="#FF725C") +
  #geom_boxplot(aes(x = head, y = mem_value), color="#0f0f0f", fill="#B6D094")
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Head", y = TeX("% variation $t_{alt}$"))+
  theme_minimal() +  # Minimal theme  
  theme(
    panel.grid.major = element_blank(), 
   # panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.title = element_text(size=20),
    legend.text = element_text(size=30),
    legend.key.size = unit(2, "cm"),
    axis.text.x = element_text(size=30, angle = 90), # Increase x-axis label size and angle
    axis.text.y = element_text(size=30), # Increase y-axis label size
    axis.title.x = element_text(size=40), # Increase x-axis label size and angle
    axis.title.y = element_text(size=40), # Increase y-axis label size
  )
cp_plot



# keep the 20 heads with a lowest cp_value mean across all examples
head_select <- df_head %>%
  group_by(head) %>%
  summarise(mem_value = mean(mem_value)) %>%
  arrange(mem_value) %>%
  head(20)
head_means <- df_head %>%
  group_by(head) %>%
  summarise(mean_mem_value = mean(mem_value)) %>%
  arrange(mean_mem_value) %>%
  head(20)

# Join with the original data to get the mem_value for each of the top 20 heads
head_select_data <- df_head %>%
  filter(head %in% head_means$head)

# Convert head to a factor and order it based on the mean mem_value
head_select_data$head <- factor(head_select_data$head, levels = head_means$head)

# Plot the boxplots with ordered heads
mem_plot <- ggplot(head_select_data, aes(x = head, y = mem_value)) +
  geom_boxplot() +
  geom_boxplot(color="#0f0f0f", fill="#357EDD") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Head", y = TeX("% variation $t_{fact}$"))+
  theme_minimal() +  # Minimal theme  
  theme(
    panel.grid.major = element_blank(), 
    #panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.title = element_text(size=20),
    legend.text = element_text(size=30),
    legend.key.size = unit(2, "cm"),
    axis.text.x = element_text(size=30, angle = 90), # Increase x-axis label size and angle
    axis.text.y = element_text(size=30), # Increase y-axis label size
    axis.title.x = element_text(size=40), # Increase x-axis label size and angle
    axis.title.y = element_text(size=40), # Increase y-axis label size
  )


plot <- arrangeGrob(cp_plot, mem_plot, ncol=1, nrow=2)
ggsave("results/plots/boxplot.pdf", plot, width = 20, height = 20)
