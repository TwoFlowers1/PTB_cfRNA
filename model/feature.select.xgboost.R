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
#trainvar=args[2] ## training data variations ;row.name=genes name; col.names=sample;
#traingro=args[3] ## training data group; case=1; control=0; only one column= 1 1 1 0 0 0 ...
#predvar=args[4]  ## predict data variations ;row.name=genes; col.names=sample;
#predgro=args[5]  ## predict data group; case=1; control=0; only one column= 1 1 1 0 0 0 ...

setwd("D:\\孙井花20180413\\项目\\2019-早产cfRNA\\数据分析\\早产\\14. 补充分析2\\01.T2P\\05.newModel\\newModel.xgboost")
outdir="./valid3"   ## out dir
varFile="../data/sam_path_group.list.773.mlmiRNA.TPM.07" ############# all exp
traingro="../data/sam.group.sPTB.train" ############## sample\tgroup header is required !!!
predgro="../data/sam.group.sPTB.valid3"  #################sample\tgroup header is required !!!
geneList="../data/gene.list" ############## gene. header is required !!!
caseName="sPTB"  ############## PPROM
contName="TB"
script_dir="../data"

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
#write.table(cor_mat,paste(outdir,"/cor.matrix.txt",sep=""),sep = "\t",quote = F)
#################################### random forest to select
#tryn=5
#seeds<-c(1:tryn)*100
#features<-c()
#for (seed1 in seeds)
#{
#	set.seed(seed1)
	# define the control using a selection function:ldaFuncs;treebagFuncs;rfFuncs;lrFuncs
# 	control <- rfeControl(functions=rfFuncs, method="cv", number=5) 
# 	if(varl < 20){subsets = varl}
# 	if(varl <= 100 & varl >= 20){subsets = seq(from=20,to=varl,by=20)} 
#	if(varl > 100){subsets = seq(from=50,to=varl,by=30)}
# 	if(varl > 100){subsets = c(10,20,50,100,200,300,500)}
# 	results <- rfe(gen_var, gen_y, sizes=subsets, rfeControl=control,metric="Accuracy")
# 	#print(results) # summarize the results
# 	var_rf<-predictors(results) # list the chosen features
# 	#plot(results, type=c("g", "o")) # plot the results
# 	features<-c(features,var_rf)
# }
# featuresFrq<-as.data.frame(table(features))
# write.table(featuresFrq,paste(outdir,"/featureFrq.RF.txt",sep=""),sep = "\t",quote = F)
# var_rf<-as.character(featuresFrq[which(featuresFrq$Freq>=tryn*0.5),1])

##########################################3 lasso selection 
# tryn=5
# seeds<-c(1:tryn)*100
# features<-c()
# for (seed1 in seeds)
# {
# 	set.seed(seed1)
# 	gen_var1<-as.matrix(gen_var)
# 	cv.fit<-cv.glmnet(gen_var1,gen_y,family="binomial",nfolds=5,type.measure="class",keep=T)
# # 	#plot(cv.fit)
# 	fit<-glmnet(gen_var1,gen_y,family="binomial")
# 	coefficients<-coef(fit,s=cv.fit$lambda.min)
# 	Active.Index<-which(coefficients!=0)     #系数不为0的特征索引
# 	var_lasso1<- (row.names(coefficients))[Active.Index]
# 	var_lasso<-var_lasso1[-1]
# 	features<-c(features,var_lasso)
# }
# featuresFrq<-as.data.frame(table(features))
# write.table(featuresFrq,paste(outdir,"/featureFrq.Lasso.txt",sep=""),sep = "\t",quote = F)
# var_lasso<-as.character(featuresFrq[which(featuresFrq$Freq>=tryn*0.5),1])

############# final features
# var2<-intersect(var_lasso,var_rf)
# if(is.na(var_lasso[1])){var2<-var_rf}
# if(is.na(var_rf[1])){var2<-var_lasso}
#if(is.na(intersect(var_lasso,var_rf))){var2<-var_rf}


#var2<-c("MYO9B","CRTAP","GRK3","FXR1","HNRNPA1","FBXL20","MYL9","FAXDC2","CD4","APP","ELOVL7","FBXO11","YIPF3","REPS1","MEFV","KIAA0513","CRAMP1","NBR1","RBM47","SYNPO","TREM1","AC108134.2","RAP2B","SEC11A","HDAC2","FOSB","ECHDC1","INPP4B","AL353796.1","TTL","TCF20","COX7A2L","KLHDC4","VPS26B","LMAN2","RASAL3","KDM2B","SKAP1","TMEM158","NUP58","POLR2C","ZNF468","MAP2K7","SERPINB6","PXK","CDK14","HNRNPH2","CARNMT1","TRIM37","SELENBP1","POLDIP3","LLPH","RMND5B","TMEM268","LPCAT2","SMN2","CAMSAP2","TUBGCP3","STOML2","ATXN10","MICALL1","ZNF800","ST3GAL4","IAH1","NAA16","TM2D1","ZNF841","OR5A1","LETMD1","CYP2E1","AC090921.1","GMDS-DT","AL049840.1","LINC01152","SERPINA10","AC019185.2","LINC01702","AC008761.2","ZNF599","LINC00689","STXBP6","HAVCR1","AP002812.2","AC005842.1","LINC01842","AC027237.2","AL592295.3","AC109927.2")  ## quake and PTB overlap
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


############################ training:xgboost
set.seed(1024)
myParamGrid <- expand.grid(
  nrounds = c(50, 100, 150),  # 迭代次数
  max_depth = c(3, 5, 7),     # 树的最大深度
  eta = c(0.01, 0.1, 0.3),    # 学习率
  gamma = 0,                 # 叶节点最小损失减少
  colsample_bytree = 0.8,    # 特征采样比例
  min_child_weight = c(1, 3, 5), #值较大时：模型更保守，值较小时：可能过拟合噪声
  subsample = c(0.6, 0.8) #每棵树训练时随机采样的样本比例
)

M1 <- train(PE~.,data=gen,method="xgbTree",trControl=trControl,metric = "ROC",tuneGrid = myParamGrid)
M1
imp(M1,"XGB",outdir)

M1$pred=M1$pred[which(M1$pred$nrounds==M1$bestTune$nrounds & M1$pred$max_depth==M1$bestTune$max_depth & M1$pred$eta==M1$bestTune$eta & M1$pred$colsample_bytree==M1$bestTune$colsample_bytree & M1$pred$subsample==M1$bestTune$subsample & M1$pred$min_child_weight==M1$bestTune$min_child_weight & M1$pred$gamma==M1$bestTune$gamma),]
train2pred("XGB",M1,gen,kk,outdir)







