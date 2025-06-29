######################## packages
library(pROC)
library(ROCR)
require(caret)
library(ggplot2)
library(ggsci)
library(PRROC)

####################### ARGVs
outdir="./"   ## out dir
al<-"LR"

############################ for R model: SVM+RF+LR+GBM+xgboost
trainProbReal=paste(al,".tainSam.pred.real.txt",sep="") ## SVM.tainSam.pred.real.txt;
predProbReal=paste(al,".predSam.predict.real.txt",sep="")  ## SVM.predSam.predict.real.txt
train1<-read.table(trainProbReal,header = T,sep="\t",row.names = 1)
pred1<-read.table(predProbReal,header = T,sep="\t",row.names = 1)

################## for python model: DL+lightGBM
trainProbReal=paste(al,"_train_risk_scores.csv",sep="") ## SVM.tainSam.pred.real.txt;
predProbReal=paste(al,"_test_risk_scores.csv",sep="")  ## SVM.predSam.predict.real.txt
train1<-read.table(trainProbReal,header = T,sep=",",row.names = 1)
pred1<-read.table(predProbReal,header = T,sep=",",row.names = 1)
colnames(train1)<-c("pred","real")
colnames(pred1)<-c("pred","real")
train1$real<-ifelse(train1$real==1,"Yes","No")
pred1$real<-ifelse(pred1$real==1,"Yes","No")

############################################ PRC
fgT<-train1$pred[which(train1$real=="Yes")]
bgT<-train1$pred[which(train1$real=="No")]
fgP<-pred1$pred[which(pred1$real=="Yes")]
bgP<-pred1$pred[which(pred1$real=="No")]

prT<-pr.curve(scores.class0 = fgT, scores.class1 = bgT,curve = TRUE)
prT
prP<-pr.curve(scores.class0 = fgP, scores.class1 = bgP,curve = TRUE)
prP

#pdf(file = "SVM.train.pdf")
pdf(paste(outdir,"/",al,".train.PRC.pdf",sep=""))
plot(prT,color=1)
dev.off()

pdf(paste(outdir,"/",al,".pred.PRC.pdf",sep=""))
plot(prP,color=2)
dev.off()



################################################# input
pred<-prediction(train1$pred,train1$real)
perf<-performance(pred,"tpr","fpr")
x2 <- unlist(perf@x.values)  ##提取x值
y2<- unlist(perf@y.values)
auc2<-round(pROC::auc(train1$real,train1$pred),3)
auc1=paste("AUC=",round(pROC::auc(train1$real,train1$pred),3),sep="")
n=length(x2)
tag=rep(auc1,n)
da <- data.frame(tag,x2,y2)
names(da)<-c("soft","fpr","tpr")

#################################### youden index
cutoffpro<-unlist(perf@alpha.values)
da$cutoffpro<-cutoffpro
da$youden=da$tpr-da$fpr
best_poin=which.max(da$youden)
best_poin_value=c(da$fpr[best_poin],da$tpr[best_poin],da$youden[best_poin],da$cutoffpro[best_poin])
best_poin_value<-as.data.frame(best_poin_value)
row.names(best_poin_value)<-c("fpr","tpr","youden","cutoff")
best_poin_value

####################### confusion matrix
tt=which(train1$pred>best_poin_value$best_poin_value[4])
train1$predClass="No"
train1$predClass[tt]="Yes"

mat=confusionMatrix(factor(train1$predClass), factor(train1$real),positive = "Yes")
unmat<-unlist(mat)
unmat1<-as.character(unmat)
unmat1[25]=auc2
rn.unmat<-names(unmat)
rn.unmat[2:5]<-c("pnrn","pyrn","pnry","pyry")
rn.unmat[25]<-c("trainROC")
unmat1[26]<-best_poin_value$best_poin_value[4]
rn.unmat[26]<-c("cutoff")
unmat2<-data.frame(unmat1)
row.names(unmat2)<-rn.unmat
write.table(unmat2,paste(outdir,"/",al,".training.confusionMatrix.best.txt",sep=""),sep="\t",quote = F)


######################### pred data confusion matrix according to yunden index
auc3<-round(pROC::auc(pred1$real,pred1$pred),3)

PP=which(pred1$pred>best_poin_value$best_poin_value[4])
pred1$predClass="No"
pred1$predClass[PP]="Yes"

matP=confusionMatrix(factor(pred1$predClass), factor(pred1$real),positive = "Yes")
unmatP<-unlist(matP)
unmatP1<-as.character(unmatP)
unmatP1[25]=auc3
rn.unmatP<-names(unmatP)
rn.unmatP[2:5]<-c("pnrn","pyrn","pnry","pyry")
rn.unmatP[25]<-c("predAUC")
unmatP2<-data.frame(unmatP1)
row.names(unmatP2)<-rn.unmatP
write.table(unmatP2,paste(outdir,"/",al,".pred.confusionMatrix.best.txt",sep=""),sep="\t",quote = F)


########################### AUC for train
pred<-prediction(train1$pred,train1$real)
perf<-performance(pred,"tpr","fpr")
x2 <- unlist(perf@x.values)  ##提取x值
y2<- unlist(perf@y.values)
auc2<-round(pROC::auc(train1$real,train1$pred),3)
auc1=paste("AUC=",round(pROC::auc(train1$real,train1$pred),3),sep="")
n=length(x2)
tag=rep(auc1,n)
da <- data.frame(tag,x2,y2)
names(da)<-c("soft","fpr","tpr")

pdf(paste(outdir,"/",al,".training.ROC.pdf",sep=""))
predpng<-ggplot(da,aes(x=fpr,y=tpr,color=soft))+geom_line(size=1)+
  ggtitle("") + xlab("False Positive Rate ") + ylab("True Positive Rate") +
  theme_bw()+
  theme(panel.grid.major=element_blank(),panel.grid.minor = element_blank())+
  theme(panel.border= element_rect(size=0.3,colour="black"))+
  theme(text=element_text(color='black' ,size=15),
        line=element_line(color='blue'),
        rect=element_rect(fill='white'),
        axis.text=element_text(color='black', size=11))+
  theme(
    legend.text=element_text(color='black', size=9,face= "bold"),
    legend.background=element_blank(), 
    legend.position = c(0.75,0.4))+
  labs(color="")+
  scale_color_d3()+geom_abline(intercept = 0, slope = 1,size=1) 
print(predpng)
dev.off()

#########################################AUC for pred
pred<-prediction(pred1$pred,pred1$real)
perf<-performance(pred,"tpr","fpr")
x2 <- unlist(perf@x.values)  ##提取x值
y2<- unlist(perf@y.values)
auc2<-round(pROC::auc(pred1$real,pred1$pred),3)
auc1=paste("AUC=",round(pROC::auc(pred1$real,pred1$pred),3),sep="")
n=length(x2)
tag=rep(auc1,n)
da <- data.frame(tag,x2,y2)
names(da)<-c("soft","fpr","tpr")

pdf(paste(outdir,"/",al,".prediction.ROC.pdf",sep=""))
predpng<-ggplot(da,aes(x=fpr,y=tpr,color=soft))+geom_line(size=1)+
  ggtitle("") + xlab("False Positive Rate ") + ylab("True Positive Rate") +
  theme_bw()+
  theme(panel.grid.major=element_blank(),panel.grid.minor = element_blank())+
  theme(panel.border= element_rect(size=0.3,colour="black"))+
  theme(text=element_text(color='black' ,size=15),
        line=element_line(color='blue'),
        rect=element_rect(fill='white'),
        axis.text=element_text(color='black', size=11))+
  theme(
    legend.text=element_text(color='black', size=9,face= "bold"),
    legend.background=element_blank(), 
    legend.position = c(0.75,0.4))+
  labs(color="")+
  scale_color_d3()+geom_abline(intercept = 0, slope = 1,size=1) 
print(predpng)
dev.off()

