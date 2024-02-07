library(ggplot2)
library(readr)
library(dplyr)
library(binom)

# Load the DataFrame
originaldf <- read_csv("results/copyVSfact/gpt2_evaluate_mechanism_NEW.csv")
originaldf <- read_csv("results/contextVSfact/gpt2_evluate_mechanism_fix_ticks.csv")
originaldf <- read_csv("results/copyVSfact/evaluate_mechanism_fix_partition.csv")

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

# Plotting the data using ggplot2
ggplot() +
  geom_line(data = df, aes(x = interval, y = percentage_true, group = model_name, color = model_name)) +
  geom_point(data = df, aes(x = interval, y = percentage_true, group = model_name, color = model_name)) +
  geom_line(data = df, aes(x = interval, y = base_percentage, group = model_name, color = model_name), linetype = "dotted") +
  labs(title = "Percentage of Factual Recalling at Each Interval",
       x = "Similarity level",
       y = "Percentage True") +
  theme_minimal()

df$model_name <- factor(df$model_name, levels = c("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "EleutherAI/pythia-6.9b"))

ggplot(df, aes(x = factor(interval), y = percentage_true, fill = model_name)) +
  geom_bar(stat = "identity", position = position_dodge(), width = 0.7) +
  #geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper), width = 0.2, position = position_dodge(width = 0.7)) +
  scale_fill_brewer(palette = "Set1") + 
  labs(title = "Percentage of Factual Recalling at Each Interval",
       x = "Similarity level",
       y = "Percentage True") +
  theme_minimal() +
  facet_wrap(~model_name, scales = "fixed")
