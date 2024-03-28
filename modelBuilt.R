###################### ARGS input
args=commandArgs(T)
outdir=args[1]   ## out dir
varFile=args[2]  ############# all exp
traingro=args[3] ############## sample\tgroup header is required !!!
predgro=args[4]  #################sample\tgroup header is required !!!
geneList=args[5] ############## gene. header is required !!!
caseName=args[6]  ############## PPROM
contName=args[7]
#trainvar=args[2] ## training data variations ;row.name=genes name; col.names=sample;
#traingro=args[3] ## training data group; case=1; control=0; only one column= 1 1 1 0 0 0 ...
#predvar=args[4]  ## predict data variations ;row.name=genes; col.names=sample;
#predgro=args[5]  ## predict data group; case=1; control=0; only one column= 1 1 1 0 0 0 ...

#######################
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
#library(psych)

###train2pred.R  plot AUC+predict;     #######pheat.R: pheatmap
source("/home/sunjinghua/tfs11/ProjICP/comShell/mulRNApred/train_pred/train2pred.R")
#source("/home/sunjinghua/tfs11/ProjICP/comShell/mulRNApred/train_pred_lasso_RF/pheat.R")

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

################################

##################### features selection based on the lasso and random forest accuracy #################
features<-c() 
######################### remove highly_correlations variation
gen=gen_raw
new_gen = gen[,!names(gen) %in% c("PE")]
gen_y<-as.factor(gen[,names(gen) %in% c("PE")])
#cor_mat = cor(new_gen,method="pearson")
#highly_correlations = findCorrelation(cor_mat,cutoff = 0.6)
#hc<-names(new_gen)[highly_correlations]
hc<-c("aa","bb")
gen_var<-new_gen[,!names(new_gen) %in% hc]
varl<-length(names(gen_var))
var2<-c("AC005842.1","AC008761.2","AC019185.2","AC027237.2","AC090921.1","AC108134.2","AC109927.2","AL049840.1","AL353796.1","AL592295.3","AP002812.2","APP","ATXN10","CAMSAP2","CD4","CDK14","COX7A2L","CRAMP1","CRTAP","ECHDC1","FAXDC2","FBXL20","FBXO11","FOSB","GMDS-DT","GRK3","HDAC2","HNRNPA1","HNRNPH2","hsa-let-7i-3p","hsa-miR-140-5p","hsa-miR-345-5p","hsa-miR-6511a-3p","IAH1","INPP4B","KDM2B","KIAA0513","KLHDC4","LETMD1","LINC00689","LINC01152","LINC01702","LINC01842","LLPH","LMAN2","LPCAT2","MAP2K7","MICALL1","MYL9","MYO9B","NAA16","NUP58","OR5A1","CARNMT1","POLR2C","CYP2E1","ELOVL7","FXR1","HAVCR1","RAP2B","MEFV","RBM47","REPS1","RMND5B","NBR1","SEC11A","SELENBP1","POLDIP3","SERPINB6","SKAP1","PXK","SMN2","RASAL3","SERPINA10","STOML2","STXBP6","ST3GAL4","TCF20","SYNPO","TM2D1","TMEM158","TMEM268","TNFRSF10A-AS1","TREM1","TRIM37","TTL","TUBGCP3","VPS26B","YIPF3","ZNF468","ZNF599","ZNF800","ZNF841")
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
myParamGrid <- expand.grid(n.trees=c(20,50,80,100,150,200,300),shrinkage=c(0.01,0.005,0.001,0.0001),interaction.depth=c(1,2,3,4),n.minobsinnode = c(10,20)) #shrinkage参数在0.01-0.001之间，而n.trees参数在3000-10000之间
M1 <- train(PE~.,data=gen,method="gbm",trControl=trControl,metric = "ROC",tuneGrid = myParamGrid)
imp(M1,"GBM",outdir)
M1$pred=M1$pred[which(M1$pred$n.trees==M1$bestTune$n.trees & M1$pred$shrinkage==M1$bestTune$shrinkage & M1$pred$interaction.depth==M1$bestTune$interaction.depth & M1$pred$n.minobsinnode==M1$bestTune$n.minobsinnode),]
train2pred("GBM",M1,gen,kk,outdir)
