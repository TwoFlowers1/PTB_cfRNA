library("DESeq2")
#library("limma")
library(dplyr)

args=commandArgs(T)
outdir=args[1]
countFile=args[2]  ######## col.names=sample;row.name=gene
groupFile=args[3]  ######## row.names=sample;col.name=batch,group


exp <- read.table(countFile,row.names=1,sep="\t",header=T,check.names=F)  ########### col.names=sample;row.name=gene
exp1<-data.frame(t(exp))
exp1<-arrange(exp1,row.names(exp1))
countdata<-t(exp1)
row.names(countdata)<-row.names(exp)  ### avoiding gene names were changed
len <- length(colnames(countdata))


clin<-read.table(groupFile,header=T,row.names=1)
coldata<-arrange(clin,row.names(clin))

sam_n=len
countdata=countdata[rowSums(countdata<1)<0.7*sam_n,] 
countdata=round(countdata)

dds <- DESeqDataSetFromMatrix(countData=countdata, colData=coldata, design = ~ batch+group)
dds_batch<-dds
dds <- DESeq(dds,quiet=TRUE)
sizefactor <- sizeFactors(dds)

normalized_counts <- counts(dds, normalized=TRUE)
file_NC=paste(outdir,"/deseq2.count.normalized.xls",sep="")
#write.table(normalized_counts, file=file_NC,quote=F, sep="\t", row.names=T, col.names=T)
#system(paste("sed -i '1 s/^/gene_name\t/'", file_NC))

result1 <- results(dds,contrast=c("group", "sPTB", "TB"), cooksCutoff=FALSE, independentFiltering=FALSE, pAdjustMethod="BH")
file_DE1=paste(outdir,"/TB-VS-sPTB.deseq2.output",sep="")
write.table(result1, file=file_DE1, quote=FALSE, sep="\t")
system(paste("sed -i '1 s/^/gene_name\t/'", file_DE1))

