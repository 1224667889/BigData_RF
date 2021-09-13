import numpy as np
import pandas as pd
import utils
from utils import Data
from sklearn.ensemble import RandomForestClassifier as RF    # 随机森林


# 读取数据
data_csv = utils.get_data('./datasets/Telco-Customer-Churn.csv')
# 构建数据对象
data = Data(data_csv)
data.get_dup_num()      # 查看是否有重复值
# 缺失值处理
data.get_isnull()       # 查看缺失值 && 查看TotalCharges的缺失值
data.data2numeric()     # 将数据转为float类型
# 缺失值填充
# full_type == 0 >> 用0填充   full_type == 1 >> 用MonthlyCharges的数值填充
data.full_data(full_type=0)
data.full_data(full_type=1)
# 异常值处理
data.show_describe(show=True)     # 输出异常值 && 展示箱型图


if __name__ == '__main__':
    # 4.1流失客户占比
    data.draw_left_client_rate(show=True)
    # 4.2基本特征对客户流失影响
    data.draw_base_attrs_power(show=True)
    # 4.3业务特征对客户流失影响
    data.draw_task_attrs_power(show=True)
    # 4.4合约特征对客户流失的影响
    data.draw_contract_attrs_power(show=True)

    # 5.1特征提取工程
    data.StandardScaler(show=True)

    # 6.1类别不平衡问题处理  && 返回预测值
    X, y, pred = data.get_predict()

    # 7 模型评估
    utils.evaluate_model(data.data, pred, y)

    # 8 结合模型
    # 8.2 预测客户流失的概率值
    prob = utils.prob_cv(X, y, RF)    # 预测概率值
    prob = np.round(prob, 1)    # 对预测出的概率值保留一位小数，便于分组观察

    # 8.3 合并预测值和真实值
    probDf = pd.DataFrame(prob)
    churnDf = pd.DataFrame(y)
    df1 = pd.concat([probDf, churnDf], axis=1)
    df1.columns = ['prob', 'churn']
    df1 = df1[:7043]    # 只取原始数据集的7043条样本进行决策
    print(df1.head(10))

    # 8.4 分组计算每种预测概率值所对应的真实流失率
    group = df1.groupby(['prob'])
    cnt = group.count()    # 每种概率值对应的样本数
    true_prob = group.sum() / group.count()    # 真实流失率
    df2 = pd.concat([cnt, true_prob], axis=1).reset_index()
    df2.columns = ['prob', 'cnt', 'true_prob']
    print(df2)
