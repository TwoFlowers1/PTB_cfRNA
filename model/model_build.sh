################################## feature selection
python /Bin/01.FeatureSelection.py


######################################################### model building
## take internal validation as example
outdir="allRNA_interValid"   ## out dir
script_dir="/Bin"
mkdir -p $outdir

varFile=/data/sam_path_group.list.773.mlmiRNA.TPM.07
traingro="/data/sam.group.sPTB.train"
predgro="/data/sam.group.sPTB.valid1"
geneList="/data/gene.list" ############## gene. header is required !!!
feature_gene="/data/feature.gene.list"
caseName="sPTB" ############## PPROM
contName="TB"

Rscript /Bin/GBM_GLM_RF_SVM.R $outdir $varFile $traingro $predgro $geneList $caseName $contName $script_dir
Rscript /Bin/xgboost.R $outdir $varFile $traingro $predgro $geneList $caseName $contName $script_dir
python /Bin/DL.py $outdir $varFile $traingro $predgro $feature_gene $caseName $contName
python /Bin/lightGBM.py $outdir $varFile $traingro $predgro $feature_gene $caseName $contName


##########################################################3 younden index & AUROC
Rscript /Bin/younden_AUC_PRC.R
