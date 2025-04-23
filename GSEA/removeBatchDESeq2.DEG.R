library("DESeq2")
library("limma")
library(dplyr)

args=commandArgs(T)
outdir=args[1]
countFile=args[2]  ######## col.names=sample;row.name=gene
groupFile=args[3]  ######## row.names=sample;col.name=batch,group


exp <- read.table(countFile,row.names=1,sep="\t",header=T,check.names=F)  ###########
exp1<-data.frame(t(exp))
exp1<-arrange(exp1,row.names(exp1))
countdata<-t(exp1)
row.names(countdata)<-row.names(exp)  ###### avoiding gene named were changed
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
write.table(normalized_counts, file=file_NC,quote=F, sep="\t", row.names=T, col.names=T)
system(paste("sed -i '1 s/^/gene_name\t/'", file_NC))

result1 <- results(dds,contrast=c("group", "PPROM", "TB"), cooksCutoff=FALSE, independentFiltering=FALSE, pAdjustMethod="BH")
file_DE1=paste(outdir,"/TB-VS-PPROM.deseq2.output",sep="")
write.table(result1, file=file_DE1, quote=FALSE, sep="\t")
system(paste("sed -i '1 s/^/gene_name\t/'", file_DE1))

result2 <- results(dds,contrast=c("group", "PTL", "TB"), cooksCutoff=FALSE, independentFiltering=FALSE, pAdjustMethod="BH")
file_DE2=paste(outdir,"/TB-VS-PTL.deseq2.output",sep="")
write.table(result2, file=file_DE2, quote=FALSE, sep="\t")
system(paste("sed -i '1 s/^/gene_name\t/'", file_DE2))


################################################################################ remove batch
#nsub<-sum(rowMeans(counts(dds_batch, normalized=FALSE)) > 5) ### avoid less than 'nsub' rows with mean normalized count > 5
vsd <- varianceStabilizingTransformation(dds_batch,blind=TRUE)  ######### for small RNA
#vsd <- vst(dds_batch)
vstMat<-assay(vsd)
file1<-paste(outdir,"batchBefore.txt",sep="/")
write.table(vstMat, file=file1,quote=F, sep="\t", row.names=T, col.names=T)
system(paste("sed -i '1 s/^/gene_name\t/'", file1))

pdf1<-paste(outdir,"PCA.batch.before.pdf",sep="/")
pdf(file = pdf1,width = 10,height = 10)
plotPCA(vsd, "batch") 
dev.off()

pdf1_1<-paste(outdir,"PCA.group.before.pdf",sep="/")
pdf(file = pdf1_1,width = 10,height = 10)
plotPCA(vsd, "group")
dev.off()



mat <- assay(vsd)
mm <- model.matrix(~group, colData(vsd))
mat <- limma::removeBatchEffect(mat, batch=as.vector(vsd$batch),design=mm)
assay(vsd) <- mat

file2<-paste(outdir,"batchAfter.txt",sep="/")
write.table(mat, file=file2,quote=F, sep="\t", row.names=T, col.names=T)
system(paste("sed -i '1 s/^/gene_name\t/'", file2))

pdf2<-paste(outdir,"PCA.batch.after.pdf",sep="/")
pdf(file = pdf2,width = 10,height = 10)
plotPCA(vsd,"batch")
dev.off()
pdf2_1<-paste(outdir,"PCA.group.after.pdf",sep="/")
pdf(file = pdf2_1,width = 10,height = 10)
plotPCA(vsd,"group")
dev.off()
