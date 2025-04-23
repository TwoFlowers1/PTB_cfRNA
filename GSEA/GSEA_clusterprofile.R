library(org.Hs.eg.db)
library(clusterProfiler)
library(pathview)
library(enrichplot)
library('R.utils')

setwd("D:\\孙井花20180413\\项目\\2019-早产cfRNA\\数据分析\\早产\\14. 补充分析2\\03.GSEA")
data <- read.table("W34.txt",header=T)
head(data)  ##GSEA分析只需要两列信息，SYMBOL和logFC
names(data)<-c("SYMBOL","logFC")
data<- na.omit(data)
  
gene <- data$SYMBOL
#开始ID转换，会有丢失
gene=bitr(gene,fromType="SYMBOL",toType="ENTREZID",OrgDb="org.Hs.eg.db") 
#去重
gene <- dplyr::distinct(gene,SYMBOL,.keep_all=TRUE)

data_all<-merge(gene,data,by="SYMBOL")

dim(data_all)
head(data_all)

data_all_sort <- arrange(data_all,desc(logFC))
head(data_all_sort)

geneList = data_all_sort$logFC #把foldchange按照从大到小提取出来
names(geneList) <- data_all_sort$ENTREZID #给上面提取的foldchange加上对应上ENTREZID
head(geneList)

gse.GO <- gseGO(
  geneList, #geneList
  ont = "BP",  # 可选"BP"、"MF"和"CC"或"ALL"
  OrgDb = org.Hs.eg.db, #人 注释基因
  keyType = "ENTREZID",
  pvalueCutoff = 0.05,
  pAdjustMethod = "BH",#p值校正方法
)
head(gse.GO,10)
gse.GO1 <- setReadable(gse.GO, OrgDb = org.Hs.eg.db, keyType="ENTREZID")
gse.GO.DF<-as.data.frame(gse.GO1)
write.table(gse.GO.DF, file="result.txt", sep="\t", row.names=F, col.names=T, quote=F)

