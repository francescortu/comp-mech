library(ggplot2)
library(dplyr)
library(tidyr)
library(latex2exp)
library(gridExtra)

df_head <- read.csv("plot_data/gpt2_df_head.csv") #example, head, cp_value, mem_value
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
  geom_boxplot(aes(x = head, y = cp_value), color="#0f0f0f", fill="#B6D094") +
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
  geom_boxplot(color="#0f0f0f", fill="#993955") +
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
ggsave("plot_data/boxplot.pdf", plot, width = 20, height = 20)

################################################################################

############################################################################
heads_count <- read.csv("plot_data/gpt2_count.csv")

#compute percentage
heads_count$mem_positive <- 100 * heads_count$mem_positive/heads_count$total
heads_count$mem_negative <- 100 * heads_count$mem_negative/heads_count$total
heads_count$cp_positive <- 100 * heads_count$cp_positive/heads_count$total
heads_count$cp_negative <- 100 * heads_count$cp_negative/heads_count$total
#count the avarage value of mem_positive, mem_negative, cp_positive, cp_negative
print(mean(heads_count$mem_positive))
print(mean(heads_count$mem_negative))
print(mean(heads_count$cp_positive))
print(mean(heads_count$cp_negative))

#select the first 20 heads with mem_positive higher
best_mem_positive <- heads_count[order(heads_count$mem_positive, decreasing = TRUE),]
best_mem_negative <- heads_count[order(heads_count$mem_negative, decreasing = TRUE),]
best_cp_positive <- heads_count[order(heads_count$cp_positive, decreasing = TRUE),]
best_cp_negative <- heads_count[order(heads_count$cp_negative, decreasing = TRUE),]


####
best_cp_negative_ordered <- best_cp_negative %>%
  arrange(desc(cp_negative))

# Reshape the ordered data to long format
long_data <- gather(best_cp_negative_ordered, key = "type", value = "value", cp_negative, mem_negative)

# Factor the head variable based on the order in best_mem_negative
long_data$head <- factor(long_data$head, levels = best_cp_negative_ordered$head)

# Plotting side by side bar plot
ggplot(long_data, aes(x = head, y = value, fill = type)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = c("cp_negative" = "red", "mem_negative" = "blue")) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(x = "Head", y = "Percentage", title = "The best 20 heads with mem_negative higher")


#plot the scatter plot of cp_negative and mem_positive
ggplot(heads_count, aes(x = cp_negative, y = mem_positive)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "cp_negative", y = "mem_positive", title = "The scatter plot of cp_negative and mem_positive")

#print the pearson correlation coefficient
print(cor(heads_count$cp_negative, heads_count$mem_positive))

# plot the scatter plot of cp_positive and mem_negative
ggplot(heads_count, aes(x = cp_positive, y = mem_negative)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "cp_positive", y = "mem_negative", title = "The scatter plot of cp_positive and mem_negative")

# print the pearson correlation coefficient
print(cor(heads_count$cp_positive, heads_count$mem_negative))



head_data <- read.csv("plot_data/gpt2_df_head.csv")
#plot correlation between cp_value and mem_value
ggplot(head_data, aes(x = cp_value, y = mem_value)) +
  geom_point() +
  labs(x = "cp_value", y = "mem_value", title = "The scatter plot of cp_value and mem_value")

