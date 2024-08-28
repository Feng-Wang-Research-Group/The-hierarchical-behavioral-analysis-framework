library(linkET) 
library(dplyr)
library(ggplot2)
library(ggcor)
library(MASS)

setwd('F:/spontaneous_behavior/re-submit/sFigure10-11_Movement_spatiotemporal_pattern/Mov_time_loc_data')
Movement_csv_data <- read.csv('stress_animal_singleMice_Mov.csv')
movement_label <- Movement_csv_data

Location_time_Movement_csv_data <- read.csv('stress_animal_Temporal_locMoc.csv')


r.p.data.plot <- Location_time_Movement_csv_data %>% 
  mutate(r.sign = cut(r, breaks = c(-Inf, 0, Inf), 
                      labels = c("Negative", "Positive")),
         p.sign = cut(p, breaks = c(0, 0.05, Inf), 
                      labels = c("P<0.05", "P>=0.05"),
                      include.lowest = TRUE,
                      right = FALSE), 
         r.abs = cut(abs(r), breaks = c(0, 0.3, 0.7, 1),
                     labels = c("<0.3 weak","0.3-0.7 moderate", ">0.7 strong"),
                     include.lowest = TRUE,
                     right = TRUE), 
  )  

quickcor(movement_label,cor.test = TRUE, type = "upper",show.diag = T,grid.size =1.2,grid.colour = 'black',lwd = 4) + 
  geom_circle2(data = get_data(p.value < 0.05, type = "upper")) + 
  anno_link(data = r.p.data.plot,  
            aes(colour = r.sign,
                size = r.abs,
                linetype = p.sign),width = 3, 
            nudge_x = 1,
            curvature = 0.1) + 
  scale_size_manual(values = c("<0.3 weak" = 0.5,
                               "0.3-0.7 moderate" = 2,
                               ">0.7 strong" = 5)) +  #
  scale_colour_manual(values = c("Positive" = "#ca6720",
                                 "Negative" = "#2a6295"))+  
  scale_linetype_manual(values = c("P<0.05" = "solid",
                                   "P>=0.05" = "dashed"))+
  scale_fill_gradientn(colours = rev(c("#5b0018", "#e0745a", "#fbfbf7", "#63a1cb", "#052452")),  
                       breaks = seq(-1, 1, 0.2),
                       limits = c(-1, 1))+ #set legend
  #add_diag_label()+
  guides(
    fill = guide_colorbar(title = "Pearson's r"), 
    linetype = guide_legend(title = NULL),
    colour = guide_legend(title = 'correlation with time'),
    size = guide_legend(title = "strength relationship")
  )+
  theme(legend.position="none")

ggsave('F:/spontaneous_behavior/GitHub/The-hierarchical-behavioral-analysis-framework/sFigure19_spatiotemporal_movement_pattern/Output_figure/stress_animal_lightOn.png',dpi=600)