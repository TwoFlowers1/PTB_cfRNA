library(pheatmap)
heatmap<-function(data,varn,casename,controlname)  ### data:row.names=samples; col.names=gene; group is the last column; varn= gene number 
{
  dat<-t(data[,1:varn])
#  dat<-log2(dat+1)   ## if run with VST, this command line is not nessesary
  Group <- c(rep(casename,length(which(data[,varn+1]=="Yes"))),rep(controlname,length(which(data[,varn+1]=="No"))))
  annotation_col = data.frame(sample = factor(Group,levels=c(casename,controlname)))
  rownames(annotation_col) = colnames(dat)
  ann_colors = list(sample = c("#BC3C29FF","#0072B5FF"))
  names(ann_colors$sample)<-c(casename,controlname)
  pheatmap(dat,scale='row',
           cluster_rows = T,cluster_cols = F,
           annotation_col = annotation_col,
           #annotation_row = annotation_row,
           annotation_colors = ann_colors,
           legend = T,         show_rownames=T,
           show_colnames=F,
           treeheight_row=30,
           border_color="NA",
           fontsize=10,
           fontsize_row=8,
           color = colorRampPalette(colors = c("blue","white","red"))(100)
  )
}
