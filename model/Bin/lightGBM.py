import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
import joblib
import sys
import os
###############################################################################input
#ref_dir = "/data/input/Files/ReferenceData/sunjinghua/ProjPTB/data"
#exp_file = ref_dir + "/" + "sam_path_group.list.773.mlmiRNA.TPM.07"
#train_labels_file= ref_dir + "/" + "sam.group.sPTB.train"
#feature_genes_file = "/data/work/python" + "/" + "feature.gene.list"
#test_labels_file = ref_dir + "/" + "sam.group.sPTB.valid3"
#outdir = "/data/work/result.lightGBM/valid3"
#caseName = 'sPTB'
#contName = 'TB'

outdir = sys.argv[0]
exp_file = sys.argv[1]
train_labels_file = sys.argv[2]
test_labels_file = sys.argv[3]
feature_genes_file = sys.argv[4]
caseName = sys.argv[5]
contName = sys.argv[6]

expr_data = pd.read_csv(exp_file, sep="\t", index_col=0)  # 输入文件1：表达量
train_labels = pd.read_csv(train_labels_file, sep="\t")         # 输入文件2：训练集标签
test_labels = pd.read_csv(test_labels_file,sep="\t")           # 输入文件3：测试集标签
feature_genes = pd.read_csv(feature_genes_file, sep="\t", header=None)[0].tolist()  # 输入文件4：特征基因
# 将标签转换为数值类型
#label_mapping = {'sPTB': 1, 'TB': 0}
label_mapping = {caseName: 1, contName: 0}
train_labels['group'] = train_labels['group'].map(label_mapping)
test_labels['group'] = test_labels['group'].map(label_mapping)
#print(expr_data.tail(10).iloc[:, -10:])
#print(f"表达量数据维度: {expr_data.shape} (样本×基因)")
#print(f"特征基因匹配度: {len(feature_genes)} / {expr_data.shape[1]}")

########################### pick training and testing data
train_samples = set(train_labels['sample'])
test_samples = set(test_labels['sample'])
train_samples_list = list(train_samples)
test_samples_list = list(test_samples)

train_expr = expr_data[train_samples_list]
test_expr = expr_data[test_samples_list]

train_expr_feature_genes = train_expr.T[feature_genes]
test_expr_feature_genes = test_expr.T[feature_genes]

def scale_data(data):
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    scaled_data = (data - means) / stds
    return scaled_data, means, stds

# 对训练集进行标准化
train_expr_scaled, train_means, train_stds = scale_data(train_expr_feature_genes)

# 使用训练集的均值和标准差对测试集进行标准化
test_expr_scaled = (test_expr_feature_genes - train_means) / train_stds

############################# merge expression and lables
train_labels.set_index('sample', inplace=True)
merged_train_data = train_expr_scaled.merge(train_labels, left_index=True, right_index=True)   ## label is group
#train_expr_feature_genes1 = train_expr_feature_genes.merge(train_labels, left_index=True, right_index=True)
test_labels.set_index('sample', inplace=True)
merged_test_data = test_expr_scaled.merge(test_labels, left_index=True, right_index=True)  ## label is group


# 定义LightGBM模型（优化不平衡数据处理）
model = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    n_jobs=-1,
    random_state=1024,
    scale_pos_weight=1.7  # 调整正负样本权重
)

param_grid = {
    'num_leaves': [63, 127],          # 推荐范围：样本量/10 ~ 样本量/5
    'learning_rate': [0.05, 0.1],     # 中等数据量建议0.05-0.1
    'max_depth': [8, 10],             # 树深度控制在5-15之间
    'min_child_samples': [80, 150],   # 叶节点最小样本量建议50-200
    'reg_alpha': [0.1, 0.5]           # L1正则化强度
}

# 网格搜索配置
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=10,random_state=1024, shuffle=True),
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

# determin parameters
X_train = merged_train_data.iloc[:, :-1]
y_train =  merged_train_data['group']
grid_search.fit(X_train, y_train)

################# refit final model
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)  ### best model to refit final model using all training data set

####################### best_params
best_params = grid_search.best_params_
#print(f"最佳参数: {grid_search.best_params_}")
with open(os.path.join(outdir, 'LGBM_best_params.txt'), 'w') as f:
    f.write(f"Best Parameters:\n")
    for k, v in best_params.items():
        f.write(f"{k}: {v}\n")
        
###################### feature importance
feature_importance = pd.DataFrame({
    'Gene': feature_genes,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
feature_importance.to_csv(os.path.join(outdir, 'LGBM_feature_importance.tsv'), sep='\t', index=False)

############################### CV result
#cv_results = lgb.cv(
#    grid_search.best_params_,
#    lgb.Dataset(X_train, label=y_train),
#    num_boost_round=100,
#    nfold=10,
#    stratified=True,
#    shuffle=True,
#    metrics='roc_auc',
#    #early_stopping_rounds=10,
#    seed=1024
#)

#cv_results_df = pd.DataFrame(cv_results)
#cv_results_df.to_csv('cv_results.csv', index=False)
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.to_csv(os.path.join(outdir, 'LGBM_cv_results.csv'), index=False)

################################### training risk score and AUC
#X_train = merged_train_data.iloc[:, :-1]
#y_train =  merged_train_data['group']
train_proba = best_model.predict_proba(X_train)[:, 1]
train_auc = roc_auc_score(y_train, train_proba)
train_auc
train_risk_scores = pd.DataFrame({
    'SampleID': X_train.index,
    'RiskScore': train_proba,
    'Diagnosis': y_train
    })
train_risk_scores.to_csv(os.path.join(outdir, 'LGBM_train_risk_scores.csv'), index=False)

################################### testing risk score and AUC
X_test = merged_test_data.iloc[:, :-1]
y_test =  merged_test_data['group']
test_proba = best_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_proba)  
test_auc
test_risk_scores = pd.DataFrame({
    'SampleID': X_test.index,
    'RiskScore': test_proba,
    'Diagnosis': y_test
    })
test_risk_scores.to_csv(os.path.join(outdir, 'LGBM_test_risk_scores.csv'), index=False)
#################################3 AUC
with open(os.path.join(outdir, "LGBM_auc_results.txt"), "w") as f:
    f.write(f"Test AUC: {test_auc:.4f}\n")
    f.write(f"Train AUC: {train_auc:.4f}\n")
######################################## save model
joblib.dump(best_model, os.path.join(outdir, 'LGBM_best_model.pkl'))

