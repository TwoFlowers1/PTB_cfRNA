train2pred<-function(al,M2,gen,kk,outdir)  #al:algorithms-RF-SVM; M2=M1;gen:training data; kk: predction data
{
  ############### saving model and training ROC
  save(M2,file = paste(outdir,"/",al,".model.Final.RData",sep=""))
  write.table(M2$bestTune,paste(outdir,"/",al,".model.Final.infor.txt",sep=""),sep="\t",quote = F,row.names = F)
  
  ############### plot training K fold ROC
  auc_all=0
  mm=unique(M2$pred$Resample)
  mm1=0
  for (i in mm)
  {
    mm1=mm1+1
    id<-which(M2$pred$Resample==i)
    predYes<-M2$pred$Yes[id]
    trueObj<-M2$pred$obs[id]
    pred<-prediction(predYes,trueObj)
    perf<-performance(pred,"tpr","fpr")
    x1 <- unlist(perf@x.values)  ##提取x值
    y1 <- unlist(perf@y.values) 
    n=length(x1)
    auc1=paste("AUC",mm1,"=",round(pROC::auc(trueObj,predYes),3),sep="")
    if(i=="Fold01.Rep1" || i=="Fold1.Rep1"){index=c(1:n);x=x1;y=y1;tag=rep(auc1,n);auc_all=rep(pROC::auc(trueObj,predYes),n)}
    else{index=c(index,c(1:n));x=c(x,x1);y=c(y,y1);tag=c(tag,rep(auc1,n));auc_all=c(auc_all,rep(pROC::auc(trueObj,predYes),n))}
  }
   # meanAUC= paste("meanAUC=",round(max(M2$results$ROC),3),sep="")   
    meanAUC= paste("meanAUC=",round(mean(auc_all),3),sep="")
    da <- data.frame(tag,x,y,index)
    names(da)<-c("soft","fpr","tpr","index")
    x1<-tapply(da$fpr,factor(da$index),mean)
    y1<-tapply(da$tpr,factor(da$index),mean)
    tag1<-rep(meanAUC,length(levels(factor(da$index))))
    index1<-c(1:length(levels(factor(da$index))))
  #  meanda<-data.frame(tag1,x1,y1,index1)
    x=c(x,x1);y=c(y,y1);tag<-c(tag,tag1);index=c(index,index1);
    da <- data.frame(tag,x,y,index)
    names(da)<-c("soft","fpr","tpr","index")
    
    pdf(file=paste(outdir,"/",al,".training.KF.ROC.pdf",sep=""))
    kfpng<-ggplot(da,aes(x=fpr,y=tpr,color=soft))+geom_line(size=1)+
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
      legend.position = c(0.80,0.45))+
      labs(color="")+
     scale_color_d3()+geom_abline(intercept = 0, slope = 1,size=1) 
    print(kfpng)
    dev.off()
    
    ############################ ROC for training data
    if(al=="LR" || al=="SVM" || al=="GBM" ){trainLR <- predict(M2,type='prob',newdata=gen);trainYesprob=trainLR$Yes;trainYesRaw=predict(M2,type='raw',newdata=gen)}
    if(al=="RF"){trainYesprob=M2$finalModel$votes[,2];trainYesRaw=M2$finalModel$predicted}
    pred3<-prediction(trainYesprob,gen$PE)
    perf3<-performance(pred3,"tpr","fpr")
    x3 <- unlist(perf3@x.values)  ##提取x值
    y3<- unlist(perf3@y.values)
    auc3=paste("trainAUC=",round(pROC::auc(gen$PE,trainYesprob),3),sep="")
    n=length(x3)
    tag=rep(auc3,n)
    da <- data.frame(tag,x3,y3)
    names(da)<-c("soft","fpr","tpr")
    
    pdf(paste(outdir,"/",al,".training.ROC.pdf",sep=""))
    trainpng<-ggplot(da,aes(x=fpr,y=tpr,color=soft))+geom_line(size=1)+
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
    print(trainpng)
    dev.off()
    
    #############confuse matrix for training data
    trainROC<-pROC::auc(gen$PE,trainYesprob)
    trainmat<-confusionMatrix(trainYesRaw, gen$PE,positive = "Yes")
    untrainmat<-unlist(trainmat)
    untrainmat1<-as.character(untrainmat)
    untrainmat1[25:26]=c(trainROC[1],max(M2$results$ROC))
    rn.untrainmat<-names(untrainmat)
    rn.untrainmat[2:5]<-c("pnrn","pyrn","pnry","pyry")
    rn.untrainmat[25:26]<-c("trainROC","KFmeanAUC")
    untrainmat2<-data.frame(untrainmat1)
    row.names(untrainmat2)<-rn.untrainmat
    write.table(untrainmat2,paste(outdir,"/",al,".training.confusionMatrix.txt",sep=""),sep="\t",quote = F)
    
    trainsam<-data.frame(trainYesprob,as.character(gen$PE))
    row.names(trainsam)<-row.names(gen)
    names(trainsam)<-c("pred","real")
    write.table(trainsam,paste(outdir,"/",al,".tainSam.pred.real.txt",sep=""),sep="\t",quote = F)
    
    ################################################################## prediction
    predict. <- predict(M2,type='prob',newdata=kk)
    predROC=pROC::auc(kk$PE,predict.$Yes)
    pred<-prediction(predict.$Yes,kk$PE)
    perf<-performance(pred,"tpr","fpr")
    x2 <- unlist(perf@x.values)  ##提取x值
    y2<- unlist(perf@y.values)
    auc1=paste("validaAUC=",round(pROC::auc(kk$PE,predict.$Yes),3),sep="")
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
    
    ############################Confusion Matrix2
    predict1 <- predict(M2,type='raw',newdata=kk)
    mat=confusionMatrix(predict1, kk$PE,positive = "Yes")
    unmat<-unlist(mat)
    unmat1<-as.character(unmat)
    unmat1[25]=predROC[1]
    rn.unmat<-names(unmat)
    rn.unmat[2:5]<-c("pnrn","pyrn","pnry","pyry")
    rn.unmat[25]<-c("predAUC")
    unmat2<-data.frame(unmat1)
    row.names(unmat2)<-rn.unmat
    write.table(unmat2,paste(outdir,"/",al,".pred.confusionMatrix.txt",sep=""),sep="\t",quote = F)
    
    predsam<-data.frame(predict.$Yes,as.character(kk$PE))
    row.names(predsam)<-row.names(kk)
    names(predsam)<-c("pred","real")
    write.table(predsam,paste(outdir,"/",al,".predSam.predict.real.txt",sep=""),sep="\t",quote = F)
}
