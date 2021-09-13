import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RF    # 随机森林
from sklearn.model_selection import KFold


def evaluate_model(data, pred, y):
    # 7 模型评估
    # 7.1 精确率、召回率以及综合两者的F1值
    from sklearn.metrics import precision_score, recall_score, f1_score    # 导入精确率、召回率、F1值等评价指标
    scoreDf = pd.DataFrame(columns=['LR', 'SVC', 'RandomForest', 'AdaBoost', 'XGBoost'])
    for i in range(5):
        r = recall_score(y, pred[i])
        p = precision_score(y, pred[i])
        f1 = f1_score(y, pred[i])
        scoreDf.iloc[:, i] = pd.Series([r, p, f1])

    scoreDf.index = ['Recall', 'Precision', 'F1-score']
    print(scoreDf)

    # 7.2 特征重要性
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    y_pred = np.zeros(len(y))    # 初始化y_pred数组
    clf = RF()

    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]    # 划分数据集
        clf.fit(X_train, y_train)    # 模型训练
        y_pred[test_index] = clf.predict(X_test)    # 模型预测

    feature_importances = pd.DataFrame(clf.feature_importances_,
                                       index=data.columns.drop(['Churn']),
                                       columns=['importance']).sort_values('importance', ascending=False)
    print(feature_importances)    # 查看特征重要性


# 8.2结合模型
# 预测客户流失的概率值
def prob_cv(X, y, classifier, **kwargs):
    """
    :param X: 特征
    :param y: 目标变量
    :param classifier: 分类器
    :param **kwargs: 参数
    :return: 预测结果
    """
    kf = KFold(n_splits=5, random_state=0,shuffle=True)
    y_pred = np.zeros(len(y))
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        clf = classifier(**kwargs)
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict_proba(X_test)[:, 1]  # 注：此处预测的是概率值
    return y_pred
