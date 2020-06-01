file1path<-'/Users/yaofeifan/Documents/Tsinghua/Lesson/模式识别/Project2/data/北京空气质量/2015/plot1.csv'
myPalette1<- c("#34C7DF", "#2E1E38", "#383064", "#DA2728", "#D28888", "#EC761B", "#2490C5", "#782B2F")
data1<-read.csv(file1path,1)
data1$chem=factor(data1$chem)
ggplot(data1,aes(x=array(1:29288),y=value ,color=chem))+
  geom_line()+
  facet_grid(chem~.)+
  theme_classic()+
  xlab('Time')+
  ylab('Value')+
  guides(color=FALSE)+
  scale_color_manual(values = myPalette1)+
  ggtitle('2015 Air Quality')

file2path<-'/Users/yaofeifan/Documents/Tsinghua/Lesson/模式识别/Project2/data/北京空气质量/2015/plot2.csv'
myPalette2<- c("#E22109", "#262228", "#F3CC15", "#29C5E0", "#807982", "#238CCC", "#84271F", "#CE8680", "#D9801C")
data2<-read.csv(file2path,1)
data2$quality=factor(data2$quality)
ggplot(data2,aes(x=array(1:21966),y=value ,color=quality))+
  geom_line()+
  facet_grid(quality~.)+
  theme_classic()+
  xlab('Time')+
  ylab('Value')+
  guides(color=FALSE)+
  scale_color_manual(values = myPalette2)+
  ggtitle('2015 Air Quality')

myPalette3<- c("#97BF87", "#674E6D", "#496973")
file3path<-'/Users/yaofeifan/Documents/Tsinghua/Lesson/模式识别/Project2/Parameter.xlsx'
data3<-read.xlsx(file3path,1)
ggplot(data3,aes(x=Neuro,y=RMSE,color=Activation,linetype=Dataa))+
  geom_line()+
  geom_point(size=3)+
  theme_classic()+
  xlab('Neuro')+
  ylab('RMSE')+
  ggtitle('Different Network Parameters')+
  scale_x_continuous(breaks = c(10,30,50,70,90))+
  scale_color_manual(values = myPalette3)+
  labs(linetype='Train Data')+
  scale_linetype_discrete(labels = c('2015-2020','2015'))

myPalette4<- c("#D17B5D", "#D22F46")
file4path<-'/Users/yaofeifan/Documents/Tsinghua/Lesson/模式识别/Project2/Feature.xlsx'
data4<-read.xlsx(file4path,1)
data4$Feature = factor(data4$Feature, levels=c('With','Without','PM2.5'))
data4$value = round(data4$value, 3)
ggplot(data4,aes(x =Feature,y = value,fill = index))+
  geom_bar(stat = 'identity',position="dodge")+
  xlab('Different Feature')+
  ylab('Value')+
  ggtitle('Feature Combination')+
  theme_classic()+
  scale_fill_manual(values = myPalette4)+
  scale_x_discrete(labels=c('With O3','Without O3',' Only PM2.5'))+
  geom_text(aes(label = value), position = position_dodge(width =0.8),vjust = -0.3)




