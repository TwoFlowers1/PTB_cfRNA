import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid  #
import joblib
import os
import sys

# early_stopping & saving
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=20, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_dl_model.h5', save_best_only=True)

###############################################################################input
#ref_dir = "/data/input/Files/ReferenceData/sunjinghua/ProjPTB/data"
#exp_file = ref_dir + "/" + "sam_path_group.list.773.mlmiRNA.TPM.07"
#train_labels_file= ref_dir + "/" + "sam.group.sPTB.train"
#feature_genes_file = "/data/work/python" + "/" + "feature.gene.list"
#caseName = "sPTB"
#contName = "TB"
#test_labels_file = ref_dir + "/" + "sam.group.sPTB.valid1"
#outdir= "/data/work/result.DL/valid1"

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
label_mapping = {caseName: 1, contName: 0}
train_labels['group'] = train_labels['group'].map(label_mapping)
test_labels['group'] = test_labels['group'].map(label_mapping)
#print(expr_data.tail(10).iloc[:, -10:])
#print(f"表达量数据维度: {expr_data.shape} (样本×基因)")
#print(f"特征基因匹配度: {len(feature_genes)} / {expr_data.shape[1]}")

##################################################################################### preprocess
########################### pick training and testing data
train_samples = set(train_labels['sample'])
test_samples = set(test_labels['sample'])
train_samples_list = list(train_samples)
test_samples_list = list(test_samples)

train_expr = expr_data[train_samples_list]
test_expr = expr_data[test_samples_list]

train_expr_feature_genes = train_expr.T[feature_genes]
test_expr_feature_genes = test_expr.T[feature_genes]

######################## scale data set
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

X_train = merged_train_data.iloc[:, :-1]
y_train =  merged_train_data['group']
X_test = merged_test_data.iloc[:, :-1]
y_test =  merged_test_data['group']

############################################################################################## training 
# deep-learning model
def create_dl_model(input_dim):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# 超参数配置
dl_params = {
    'batch_size': [32, 64],
    'epochs': [100, 200],
    'learning_rate': [0.001, 0.0005],
    'optimizer': [tf.keras.optimizers.Adam, tf.keras.optimizers.RMSprop]
}

# cross-validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1024)
best_auc = 0
best_model = None
cv_results = []

fold_no = 1
for train_idx, val_idx in kfold.split(X_train.values, y_train.values):
    print(f'Training Fold {fold_no}...')
    
    # data spliting 
    X_tr, X_val = X_train.values[train_idx], X_train.values[val_idx]
    y_tr, y_val = y_train.values[train_idx], y_train.values[val_idx]
    
    
    # Hyperparameter Grid Search
    for params in ParameterGrid(dl_params):
        print(f'  Trying params: {params}')
        
        # build model
        model = create_dl_model(X_tr.shape[1])
        model.compile(
            optimizer=params['optimizer'](learning_rate=params['learning_rate']),
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.AUC(name='auc')]
        )
        
        # train
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            callbacks=[early_stopping, model_checkpoint],
            verbose=0
        )
        
        # test
        val_auc = max(history.history['val_auc'])
        if val_auc > best_auc:
            best_auc = val_auc
            best_model = tf.keras.models.load_model('best_dl_model.h5')
            best_params = params
        
         # save CV result
        cv_results.append({
            'fold': fold_no,
            'params': params,
            'val_auc': val_auc,
            'best_epoch': len(history.history['val_auc'])
        })
    
    fold_no += 1

# refit the best model using all training data
#scaler_full = StandardScaler().fit(X_train.values)
#X_train_scaled = scaler_full.transform(X_train.values)
best_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'])

# save CV result
cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_csv(os.path.join(outdir, 'DL_cv_results.csv'), index=False)

################################### training risk score and AUC
#X_train = merged_train_data.iloc[:, :-1]
#y_train =  merged_train_data['group']
train_proba = best_model.predict(X_train).flatten()
train_auc = roc_auc_score(y_train, train_proba)
train_auc
train_risk_scores = pd.DataFrame({
    'SampleID': X_train.index,
    'RiskScore': train_proba,
    'Diagnosis': y_train
    })
train_risk_scores.to_csv(os.path.join(outdir, 'DL_train_risk_scores.csv'), index=False)

################################### testing risk score and AUC
X_test = merged_test_data.iloc[:, :-1]
y_test =  merged_test_data['group']
test_proba = best_model.predict(X_test).flatten()
test_auc = roc_auc_score(y_test, test_proba)  
test_auc
test_risk_scores = pd.DataFrame({
    'SampleID': X_test.index,
    'RiskScore': test_proba,
    'Diagnosis': y_test
    })
test_risk_scores.to_csv(os.path.join(outdir, 'DL_test_risk_scores.csv'), index=False)
#################################3 AUC
with open(os.path.join(outdir, "DL_auc_results.txt"), "w") as f:
    f.write(f"Test AUC: {test_auc:.4f}\n")
    f.write(f"Train AUC: {train_auc:.4f}\n")

######################################## save model
best_model.save(os.path.join(outdir, 'DL_best_dl_model.h5'))

####################### best_params
with open(os.path.join(outdir, 'DL_best_params.txt'), 'w') as f:
    f.write(f"Best Parameters:\n")
    for k, v in best_params.items():
        f.write(f"{k}: {v}\n")
		
def compute_feature_importance(model, X, y):
    # 假设我们关心的是第一个评估指标（通常是损失值）
    baseline_score = model.evaluate(X, y, verbose=0)[0]  # 提取损失值
    feature_importance = []
    
    for col in X.columns:
        X_permuted = X.copy()
        # 使用 .values 确保操作 NumPy 数组
        X_permuted[col] = np.random.permutation(X_permuted[col].values)
        permuted_score = model.evaluate(X_permuted, y, verbose=0)[0]  # 提取损失值
        # 确保数值类型一致
        importance = baseline_score - permuted_score
        feature_importance.append(importance)
    
    return feature_importance


feature_importance = compute_feature_importance(best_model, X_train, y_train)
pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importance
}).to_csv(os.path.join(outdir, 'DL_feature_importance.csv'), index=False)
