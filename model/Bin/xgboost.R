###################### ARGS input
args=commandArgs(T)
outdir=args[1]   ## out dir
varFile=args[2]  ############# all exp
traingro=args[3] ############## sample\tgroup header is required !!!
predgro=args[4]  #################sample\tgroup header is required !!!
geneList=args[5] ############## gene. header is required !!!
caseName=args[6]  ############## PPROM
contName=args[7]
script_dir=args[8]

#######################
library(ggplot2)
library(pROC)
library(ROCR)
library(ggsci)
require(caret)
library(dplyr)
library(pheatmap)
library(xgboost)

###train2pred.R  plot AUC+predict;     #######pheat.R: pheatmap
#/home/sunjinghua/tfs11/ProjICP/comShell/mulRNApred/train_pred/train2pred.R
source(paste(script_dir,"train2pred_xgboost.R",sep="/"))
source(paste(script_dir,"pheat.R",sep="/"))

######################### files input #########################################
targ<-read.table(geneList,header = T) #
exp1<-read.table(varFile,header = T,sep="\t",row.names = 1,check.names=F)
exp<-exp1[row.names(exp1)%in%targ[,1],]
exp<-t(exp)
traingroD<-read.table(traingro,header = T,sep="\t",row.names = 1,check.names=F)
traingroD[,1]<-factor(traingroD[,1],levels=c(contName,caseName),labels=c("No","Yes"))
predgroD<-read.table(predgro,header = T,sep="\t",row.names = 1,check.names=F)
predgroD[,1]<-factor(predgroD[,1],levels=c(contName,caseName),labels=c("No","Yes"))

trainVar<-exp[row.names(exp)%in%row.names(traingroD),]
trainVarGrp<-merge(trainVar,traingroD,by="row.names")[,-1]
train_sam_order<-merge(trainVar,traingroD,by="row.names")[,1]
gen1<-trainVarGrp[,-length(colnames(trainVarGrp))]


predVar<-exp[row.names(exp)%in%row.names(predgroD),]
predVarGrp<-merge(predVar,predgroD,by="row.names")[,-1] 
pred_sam_order<-merge(predVar,predgroD,by="row.names")[,1]
kk<-predVarGrp[,-length(colnames(predVarGrp))]

########################## save sample order
train_sam<-data.frame(train_sam_order,trainVarGrp[,length(colnames(trainVarGrp))])
colnames(train_sam)<-c("sample","group")
write.table(train_sam,paste(outdir,"/train_sam.txt",sep=""),sep = "\t",quote = F,row.names=F, col.names=T)

pred_sam<-data.frame(pred_sam_order,predVarGrp[,length(colnames(predVarGrp))])
colnames(pred_sam)<-c("sample","group")
write.table(pred_sam,paste(outdir,"/pred_sam.txt",sep=""),sep = "\t",quote = F,row.names=F, col.names=T)

#######################################################


gen_raw=gen1  ##for heatmap
#gen1<-log2(gen1+1)
gen2=scale(gen1, center = TRUE, scale = TRUE);
gen1<-as.data.frame(gen2)
varLen<-length(names(gen1))
varAll<-names(gen1)

gr<-trainVarGrp[,length(colnames(trainVarGrp))]
gen1$PE<-gr
gen_raw$PE<-gen1$PE


kk_raw<-kk
#kk<-log2(kk+1)
kk <- as.data.frame(scale(kk,center = attr(gen2, "scaled:center"), scale = attr(gen2, "scaled:scale")))
grop<-predVarGrp[,length(colnames(trainVarGrp))]
kk$PE<-grop
kk_raw$PE<-kk$PE

################################
var2<-c("REPS1","MYO9B","CYP2E1","AC027237.2","LLPH","TUBGCP3","COX7A2L","TRIM37","FBXO11","TM2D1","RASAL3","LPCAT2","SELENBP1","INPP4B","KIAA0513","HDAC2","ST3GAL4","AC108134.2","LINC01842","SKAP1","hsa-miR-345-5p","TNFRSF10A-AS1","STOML2","CARNMT1","NAA16","AL049840.1","MICALL1","AC109927.2","AC090921.1","TTL","SMN2","LINC01702","GMDS-DT","OR5A1","CDK14","TCF20","KDM2B","TMEM158","LINC01152","CRTAP","MAP2K7","HNRNPH2","SYNPO","HNRNPA1","AL592295.3","NUP58","SEC11A","RMND5B","NBR1","POLR2C","ATXN10","PXK","FAXDC2","GRK3","STXBP6","SERPINA10","APP","FXR1","TMEM268","ZNF841","SERPINB6","LETMD1","ELOVL7","ZNF468","ECHDC1","KLHDC4","POLDIP3","CRAMP1","LINC00689","CD4","RAP2B","hsa-miR-140-5p","FOSB","LMAN2","hsa-let-7i-3p","AC008761.2","ZNF800","CAMSAP2","MEFV","HAVCR1","IAH1","AL353796.1","AC019185.2","FBXL20","YIPF3","VPS26B","TREM1","AC005842.1","RBM47","AP002812.2","hsa-miR-6511a-3p","ZNF599","MYL9")

var<-c(var2,"PE")
gen<- gen1[,names(gen1) %in% var]
gen_raw1<-gen_raw[,names(gen_raw) %in% var]
write.table(var2,file=paste(outdir,"/features.txt",sep=""),quote = F)
########################### plot features in training and predict set
#pdf(paste(outdir,"/training.features.pdf",sep=""))
feature_n=length(var2)
#heatmap(gen_raw1,feature_n,caseName,contName)
#dev.off()
kk<- kk[,names(kk) %in% var]
kk_raw1<- kk_raw[,names(kk_raw) %in% var]
#pdf(paste(outdir,"/prediction.features.pdf",sep=""))
#heatmap(kk_raw1,feature_n,caseName,contName)
#dev.off()


##########################################print importance for features
imp<-function(M1,model,outdir)
{
        importance = varImp(M1,scale = TRUE)
        impScoreOrder<-importance$importance[order(-importance$importance[,1]),]
        impGneOrder<-row.names(importance$importance)[order(-importance$importance[,1])]
        impOder<-data.frame(impGneOrder,impScoreOrder)
        write.table(impOder,file=paste(outdir,"/",model,".features.imp.txt",sep=""),quote = F,row.names = F)
}


############################################### training ########################################
set.seed(1024)
trControl <- trainControl(method = 'repeatedcv', number = 10,repeats = 1,classProbs = TRUE, summaryFunction = twoClassSummary,savePredictions = T)

############################ training:xgboost
set.seed(1024)
myParamGrid <- expand.grid(
  nrounds = c(50, 100, 150),  # 
  max_depth = c(3, 5, 7),     # 
  eta = c(0.01, 0.1, 0.3),    # 
  gamma = 0,                 # 
  colsample_bytree = 0.8,    # 
  min_child_weight = c(1, 3, 5), #
  subsample = c(0.6, 0.8) #
)

M1 <- train(PE~.,data=gen,method="xgbTree",trControl=trControl,metric = "ROC",tuneGrid = myParamGrid)
M1
imp(M1,"XGB",outdir)

M1$pred=M1$pred[which(M1$pred$nrounds==M1$bestTune$nrounds & M1$pred$max_depth==M1$bestTune$max_depth & M1$pred$eta==M1$bestTune$eta & M1$pred$colsample_bytree==M1$bestTune$colsample_bytree & M1$pred$subsample==M1$bestTune$subsample & M1$pred$min_child_weight==M1$bestTune$min_child_weight & M1$pred$gamma==M1$bestTune$gamma),]
train2pred("XGB",M1,gen,kk,outdir)







