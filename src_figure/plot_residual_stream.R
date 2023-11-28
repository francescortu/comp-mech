library(ggplot2)

data <- read.csv("results/plot_data/gpt2_residual_stream_mem.csv")

ggplot(data, aes(x=data$column, y=data$index, fill=data$value)) +
  #geom_tile()+
  geom_point(shape=21, stroke=0.6, size=33)+
  geom_text(aes(label=paste(paste(round(data$value, 1)))), color="#0f0f0f", size=10)+
  scale_fill_gradient2(low = "#000022",
                       mid = "#ffffff",
                       high="#357EDD",
                       labels = function(x) paste0(round(x, 1), "%"))+
  labs(x='Postion', y='Layers', fill='Above Mean (%)')+
  scale_y_continuous(breaks = seq(0, 11, 1))+
  scale_x_continuous(breaks = seq(0, 12, 1), labels = c("-", "Subect 1", "Subject 2", "Subject 3", "-", "last", "attribute","-","Subject 1", "Subject 2", "Subject 3", "-", "last"))+
  theme_minimal() +  # Minimal theme  
  theme(
      panel.grid.major = element_blank(), 
      panel.grid.minor = element_blank(),
      legend.position = "right",
      legend.title = element_text(size=20),
      legend.text = element_text(size=30),
      legend.key.size = unit(2, "cm"),
      axis.text.x = element_text(size=30, angle = 90), # Increase x-axis label size and angle
      axis.text.y = element_text(size=30), # Increase y-axis label size
      axis.title.x = element_text(size=40), # Increase x-axis label size and angle
      axis.title.y = element_text(size=40), # Increase y-axis label size
      title = element_text(size=30)
      )
ggsave("results/plots/gpt2_residual_stream_mem.pdf", width=20, height=15, dpi=300)



data <- read.csv("results/plot_data/gpt2_residual_stream_copy.csv")

ggplot(data, aes(x=data$column, y=data$index, fill=data$value)) +
  #geom_tile()+
  geom_point(shape=21, stroke=0.6, size=33)+
  geom_text(aes(label=paste(paste(round(data$value, 1)))), color="#0f0f0f", size=10)+
  scale_fill_gradient2(low = "#000022",
                       mid = "#ffffff",
                       high="#FF725C",
                       labels = function(x) paste0(round(x, 1), "%"))+
  labs(x='Postion', y='Layers', fill='Above Mean (%)')+
  scale_y_continuous(breaks = seq(0, 11, 1))+
  scale_x_continuous(breaks = seq(0, 12, 1), labels = c("-", "Subect 1", "Subject 2", "Subject 3", "-", "last", "attribute","-","Subject 1", "Subject 2", "Subject 3", "-", "last"))+
  theme_minimal() +  # Minimal theme  
  theme(
    panel.grid.major = element_blank(), 
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.title = element_text(size=20),
    legend.text = element_text(size=30),
    legend.key.size = unit(2, "cm"),
    axis.text.x = element_text(size=30, angle = 90), # Increase x-axis label size and angle
    axis.text.y = element_text(size=30), # Increase y-axis label size
    axis.title.x = element_text(size=40), # Increase x-axis label size and angle
    axis.title.y = element_text(size=40), # Increase y-axis label size
    title = element_text(size=30)
  )
ggsave("results/plots/gpt2_residual_stream_copy.pdf", width=20, height=15, dpi=300)

