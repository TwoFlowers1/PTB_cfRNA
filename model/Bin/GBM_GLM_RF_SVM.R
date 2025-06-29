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


library(ggplot2)
library(pROC)
library(ROCR)
library(ggsci)
require(caret)
library(dplyr)
library(glmnet)
library(randomForest)
library(gbm)
library(kernlab)
library(pheatmap)

###train2pred.R  plot AUC+predict;     #######pheat.R: pheatmap
source(paste(script_dir,"train2pred.R",sep="/"))
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
kk <- as.data.frame(scale(kk,center = attr(gen2, "scaled:center"), scale = attr(gen2, "scaled:scale")))
grop<-predVarGrp[,length(colnames(trainVarGrp))]
kk$PE<-grop
kk_raw$PE<-kk$PE

################################## feature
var2<-c("hsa-miR-345-5p","hsa-miR-140-5p","hsa-let-7i-3p","hsa-miR-6511a-3p","REPS1","MYO9B","CYP2E1","LLPH","TUBGCP3","COX7A2L","TRIM37","FBXO11","TM2D1","RASAL3","LPCAT2","SELENBP1","INPP4B","KIAA0513","HDAC2","ST3GAL4","LINC01842","SKAP1","TNFRSF10A-AS1","STOML2","CARNMT1","NAA16","MICALL1","TTL","SMN2","LINC01702","GMDS-DT","OR5A1","CDK14","TCF20","KDM2B","TMEM158","LINC01152","CRTAP","MAP2K7","HNRNPH2","SYNPO","HNRNPA1","NUP58","SEC11A","RMND5B","NBR1","POLR2C","ATXN10","PXK","FAXDC2","GRK3","STXBP6","SERPINA10","APP","FXR1","TMEM268","ZNF841","SERPINB6","LETMD1","ELOVL7","ZNF468","ECHDC1","KLHDC4","POLDIP3","CRAMP1","LINC00689","CD4","RAP2B","FOSB","LMAN2","ZNF800","CAMSAP2","MEFV","HAVCR1","IAH1","FBXL20","YIPF3","VPS26B","TREM1","RBM47","ZNF599","MYL9")

var<-c(var2,"PE")
gen<- gen1[,names(gen1) %in% var]
gen_raw1<-gen_raw[,names(gen_raw) %in% var]
write.table(var2,file=paste(outdir,"/features.txt",sep=""),quote = F)
########################### plot features in training and predict set
feature_n=length(var2)
kk<- kk[,names(kk) %in% var]
kk_raw1<- kk_raw[,names(kk_raw) %in% var]

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

############################ training:RF
set.seed(1024)
myParamGrid <- expand.grid(mtry=c(1:4)) #c(1,2,3,4)
M1 <- train(PE~.,data=gen,method="rf",trControl=trControl,metric = "ROC",tuneGrid = myParamGrid)
M1
imp(M1,"RF",outdir)
M1$pred=M1$pred[which(M1$pred$mtry==M1$bestTune$mtry),]
train2pred("RF",M1,gen,kk,outdir)

############################## training:glmnet
set.seed(1024)
myParamGrid=expand.grid(lambda=10^seq(-15,10,1/3),alpha=c(0,0.000001,0.00001,0.0001,0.001,0.01,0.9,1,0.1,0.2,0.3,0.5,0.7,0.8))   #
M1 <- train(PE~.,data=gen,method='glmnet',trControl=trControl,metric = "ROC",tuneGrid=myParamGrid)
imp(M1,"LR",outdir)
lambda_list=M1$bestTune$lambda
alpha_list=M1$bestTune$alpha
M1$pred=M1$pred[which(M1$pred$lambda==lambda_list & M1$pred$alpha== alpha_list),]
train2pred("LR",M1,gen,kk,outdir)


###############################training:SVM
set.seed(1024)
myParamGrid <- expand.grid(C=c(0.1,1,0.3,0.01,0.03,0.004,0.001,0.0001,0.002)) #0.0001-10000,greater the value,greater punishment for wrong cases,may result in Model overfitting
M1 <- train(PE~.,data=gen,method="svmLinear",trControl=trControl,tuneGrid=myParamGrid,metric = "ROC")
imp(M1,"SVM",outdir)
M1$pred=M1$pred[which(M1$pred$C==M1$bestTune$C),]
train2pred("SVM",M1,gen,kk,outdir)


############################ training:gbm  ## 100,300,500,800,1000
myParamGrid <- expand.grid(n.trees=c(20,50,80,100,150,200,300),shrinkage=c(0.01,0.005,0.001,0.0001),interaction.depth=c(1,2,3,4),n.minobsinnode = c(10,20)) #
M1 <- train(PE~.,data=gen,method="gbm",trControl=trControl,metric = "ROC",tuneGrid = myParamGrid)
imp(M1,"GBM",outdir)
M1$pred=M1$pred[which(M1$pred$n.trees==M1$bestTune$n.trees & M1$pred$shrinkage==M1$bestTune$shrinkage & M1$pred$interaction.depth==M1$bestTune$interaction.depth & M1$pred$n.minobsinnode==M1$bestTune$n.minobsinnode),]
train2pred("GBM",M1,gen,kk,outdir)
