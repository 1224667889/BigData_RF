import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors    # k近邻算法


class Data:
    def __init__(self, data):
        self.data = data

    def get_dup_num(self):
        # 查看是否有重复值
        dupNum = self.data.shape[0] - self.data.drop_duplicates().shape[0]
        print("数据集中有%s列重复值" % dupNum)
        # >>> 数据集中有0列重复值

    def get_isnull(self):
        # 查看缺失值
        print(self.data.isnull().any())
        print(self.data[self.data['TotalCharges'] == ' '])

    def data2numeric(self):
        # 将数据转为float类型
        # convert_numeric如果为True，则尝试强制转换为数字，不可转换的变为NaN
        self.data['TotalCharges'] = self.data['TotalCharges'].apply(pd.to_numeric, errors='coerce')
        print("此时TotalCharges是否已经转换为浮点型：", self.data['TotalCharges'].dtype == 'float')
        print("此时TotalCharges存在%s行缺失样本。" % self.data['TotalCharges'].isnull().sum())

    def full_data(self, full_type=0):
        if full_type == 0:
            # 固定值填充
            fnDf = self.data['TotalCharges'].fillna(0).to_frame()
            print("如果采用固定值填充方法还存在%s行缺失样本。" % fnDf['TotalCharges'].isnull().sum())
        elif full_type == 1:
            # 用MonthlyCharges的数值填充TotalCharges的缺失值
            self.data['TotalCharges'] = self.data['TotalCharges'].fillna(self.data['MonthlyCharges'])
            print("处理后缺失值", self.data[self.data['tenure'] == 0][['MonthlyCharges', 'TotalCharges']])

    def show_describe(self, show=False):
        print(self.data.describe())
        # 箱型图观察异常值情况

        # 分析百分比特征
        fig = plt.figure(figsize=(15, 6))  # 建立图像
        # tenure特征
        ax1 = fig.add_subplot(311)  # 子图1
        list1 = list(self.data['tenure'])
        ax1.boxplot(list1, vert=False, showmeans=True,
                    flierprops={"marker": "o", "markerfacecolor": "steelblue"})
        ax1.set_title('tenure')
        # MonthlyCharges特征
        ax2 = fig.add_subplot(312)  # 子图2
        list2 = list(self.data['MonthlyCharges'])
        ax2.boxplot(list2, vert=False, showmeans=True,
                    flierprops={"marker": "o", "markerfacecolor": "steelblue"})
        ax2.set_title('MonthlyCharges')
        # TotalCharges
        ax3 = fig.add_subplot(313)  # 子图3
        list3 = list(self.data['TotalCharges'])
        ax3.boxplot(list3, vert=False, showmeans=True,
                    flierprops={"marker": "o", "markerfacecolor": "steelblue"})
        ax3.set_title('TotalCharges')
        plt.tight_layout(pad=1.5)  # 设置子图之间的间距
        if show:
            plt.show()  # 展示箱型图

    def draw_left_client_rate(self, show=False):
        # 4.1流失客户占比
        # 观察是否存在类别不平衡现象
        p = self.data['Churn'].value_counts()  # 目标变量正负样本的分布
        plt.figure(figsize=(10, 6))  # 构建图像
        # 绘制饼图并调整字体大小
        patches, l_text, p_text = plt.pie(p, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))
        # l_text是饼图对着文字大小，p_text是饼图内文字大小
        for t in p_text:
            t.set_size(15)
        for t in l_text:
            t.set_size(15)
        if show:
            plt.show()  # 展示图像

    def draw_base_attrs_power(self, show=False):
        # 4.2基本特征对客户流失影响
        # 4.2.1 性别、是否老年人、是否有配偶、是否有家属等特征对客户流失的影响
        baseCols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
        for i in baseCols:
            cnt = pd.crosstab(self.data[i], self.data['Churn'])  # 构建特征与目标变量的列联表
            cnt.plot.bar(stacked=True)  # 绘制堆叠条形图，便于观察不同特征值流失的占比情况

        # 4.2.2 观察流失率与入网月数的关系
        # 折线图
        groupDf = self.data[['tenure', 'Churn']]    # 只需要用到两列数据
        groupDf['Churn'] = groupDf['Churn'].map({'Yes': 1, 'No': 0})    # 将正负样本目标变量改为1和0方便计算
        pctDf = groupDf.groupby(['tenure']).sum() / groupDf.groupby(['tenure']).count()    # 计算不同入网月数对应的流失率
        pctDf = pctDf.reset_index()    # 将索引变成列
        plt.figure(figsize=(10, 5))
        plt.plot(pctDf['tenure'], pctDf['Churn'], label='Churn percentage')    # 绘制折线图
        plt.legend()    # 显示图例
        pctDf.head()
        if show:
            plt.show()  # 展示图像

    def draw_task_attrs_power(self, show=False):
        # 4.3业务特征对客户流失影响
        # 4.3.1电话业务
        posDf = self.data[self.data['PhoneService'] == 'Yes']
        negDf = self.data[self.data['PhoneService'] == 'No']
        fig = plt.figure(figsize=(10, 4))   # 建立图像
        ax1 = fig.add_subplot(121)
        p1 = posDf['Churn'].value_counts()
        ax1.pie(p1, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))
        ax1.set_title('Churn of (PhoneService = Yes)')
        ax2 = fig.add_subplot(122)
        p2 = negDf['Churn'].value_counts()
        ax2.pie(p2, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))
        ax2.set_title('Churn of (PhoneService = No)')
        plt.tight_layout(pad=0.5)    # 设置子图之间的间距

        # 4.3.2多线业务
        df1 = self.data[self.data['MultipleLines'] == 'Yes']
        df2 = self.data[self.data['MultipleLines'] == 'No']
        df3 = self.data[self.data['MultipleLines'] == 'No phone service']
        fig = plt.figure(figsize=(15, 4))   # 建立图像
        ax1 = fig.add_subplot(131)
        p1 = df1['Churn'].value_counts()
        ax1.pie(p1, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))
        ax1.set_title('Churn of (MultipleLines = Yes)')
        ax2 = fig.add_subplot(132)
        p2 = df2['Churn'].value_counts()
        ax2.pie(p2, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))
        ax2.set_title('Churn of (MultipleLines = No)')
        ax3 = fig.add_subplot(133)
        p3 = df3['Churn'].value_counts()
        ax3.pie(p3, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))
        ax3.set_title('Churn of (MultipleLines = No phone service)')
        plt.tight_layout(pad=0.5)    # 设置子图之间的间距

        # 4.3.3互联网业务
        cnt = pd.crosstab(self.data['InternetService'], self.data['Churn'])    # 构建特征与目标变量的列联表
        cnt.plot.barh(stacked=True, figsize=(15, 6))    # 绘制堆叠条形图，便于观察不同特征值流失的占比情况

        # 4.3.4与互联网相关的业务
        internetCols = \
            ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        for i in internetCols:
            df1 = self.data[self.data[i] == 'Yes']
        df2 = self.data[self.data[i] == 'No']
        df3 = self.data[self.data[i] == 'No internet service']
        fig = plt.figure(figsize=(10, 3))  # 建立图像
        plt.title(i)
        ax1 = fig.add_subplot(131)
        p1 = df1['Churn'].value_counts()
        ax1.pie(p1, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))  # 开通业务
        ax2 = fig.add_subplot(132)
        p2 = df2['Churn'].value_counts()
        ax2.pie(p2, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))  # 未开通业务
        ax3 = fig.add_subplot(133)
        p3 = df3['Churn'].value_counts()
        ax3.pie(p3, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))  # 未开通互联网业务
        plt.tight_layout()  # 设置子图之间的间距
        if show:
            plt.show()  # 展示图像

    def draw_contract_attrs_power(self, show=False):
        # 4.4合约特征对客户流失的影响
        # 4.4.1合约期限
        df1 = self.data[self.data['Contract'] == 'Month-to-month']
        df2 = self.data[self.data['Contract'] == 'One year']
        df3 = self.data[self.data['Contract'] == 'Two year']
        fig = plt.figure(figsize=(15, 4))   # 建立图像
        ax1 = fig.add_subplot(131)
        p1 = df1['Churn'].value_counts()
        ax1.pie(p1, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))
        ax1.set_title('Churn of (Contract = Month-to-month)')
        ax2 = fig.add_subplot(132)
        p2 = df2['Churn'].value_counts()
        ax2.pie(p2, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))
        ax2.set_title('Churn of (Contract = One year)')
        ax3 = fig.add_subplot(133)
        p3 = df3['Churn'].value_counts()
        ax3.pie(p3, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))
        ax3.set_title('Churn of (Contract = Two year)')
        plt.tight_layout(pad=0.5)    # 设置子图之间的间距

        # 4.4.2是否采用电子结算
        df1 = self.data[self.data['PaperlessBilling'] == 'Yes']
        df2 = self.data[self.data['PaperlessBilling'] == 'No']
        fig = plt.figure(figsize=(10, 4))   # 建立图像
        ax1 = fig.add_subplot(121)
        p1 = df1['Churn'].value_counts()
        ax1.pie(p1, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))
        ax1.set_title('Churn of (PaperlessBilling = Yes)')
        ax2 = fig.add_subplot(122)
        p2 = df2['Churn'].value_counts()
        ax2.pie(p2, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))
        ax2.set_title('Churn of (PaperlessBilling = No)')
        plt.tight_layout(pad=0.5)    # 设置子图之间的间距

        # 4.4.3付款方式
        df1 = self.data[self.data['PaymentMethod'] == 'Bank transfer (automatic)']    # 银行转账（自动）
        df2 = self.data[self.data['PaymentMethod'] == 'Credit card (automatic)']    # 信用卡（自动）
        df3 = self.data[self.data['PaymentMethod'] == 'Electronic check']    # 电子支票
        df4 = self.data[self.data['PaymentMethod'] == 'Mailed check']    # 邮寄支票
        fig = plt.figure(figsize=(10, 8))   # 建立图像
        ax1 = fig.add_subplot(221)
        p1 = df1['Churn'].value_counts()
        ax1.pie(p1, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))
        ax1.set_title('Churn of (PaymentMethod = Bank transfer')
        ax2 = fig.add_subplot(222)
        p2 = df2['Churn'].value_counts()
        ax2.pie(p2, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))
        ax2.set_title('Churn of (PaymentMethod = Credit card)')
        ax3 = fig.add_subplot(223)
        p3 = df3['Churn'].value_counts()
        ax3.pie(p3, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))
        ax3.set_title('Churn of (PaymentMethod = Electronic check)')
        ax4 = fig.add_subplot(224)
        p4 = df4['Churn'].value_counts()
        ax4.pie(p4, labels=['No', 'Yes'], autopct='%1.2f%%', explode=(0, 0.1))
        ax4.set_title('Churn of (PaymentMethod = Mailed check)')
        plt.tight_layout(pad=0.5)    # 设置子图之间的间距

        # 4.4.4每月费用核密度估计图
        plt.figure(figsize=(10, 5))    # 构建图像
        negDf = self.data[self.data['Churn'] == 'No']
        sns.distplot(negDf['MonthlyCharges'], hist=False, label='No')
        posDf = self.data[self.data['Churn'] == 'Yes']
        sns.distplot(posDf['MonthlyCharges'], hist=False, label='Yes')

        # 4.4.5总费用核密度估计图
        plt.figure(figsize=(10, 5))    # 构建图像

        negDf = self.data[self.data['Churn'] == 'No']
        sns.distplot(negDf['TotalCharges'], hist=False, label='No')
        posDf = self.data[self.data['Churn'] == 'Yes']
        sns.distplot(posDf['TotalCharges'], hist=False, label='Yes')
        if show:
            plt.show()  # 展示图像

    def StandardScaler(self, show=False):
        # 5.1特征提取工程
        # 5.1.1数值特征标准化
        from sklearn.preprocessing import StandardScaler    # 导入标准化库

        scaler = StandardScaler()
        self.data[['tenure']] = scaler.fit_transform(self.data[['tenure']])
        self.data[['MonthlyCharges']] = scaler.fit_transform(self.data[['MonthlyCharges']])
        self.data[['TotalCharges']] = scaler.fit_transform(self.data[['TotalCharges']])
        print(self.data[['tenure', 'MonthlyCharges', 'TotalCharges']].head())    # 观察此时的数值特征

        # 5.1.2类别特征编码
        # 首先将部分特征值进行合并
        self.data.loc[self.data['MultipleLines'] == 'No phone service', 'MultipleLines'] = 'No'
        internetCols = \
            ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        for i in internetCols:
            self.data.loc[self.data[i] == 'No internet service', i] = 'No'
        print("MultipleLines特征还有%d条样本的值为 'No phone service'" %
              self.data[self.data['MultipleLines'] == 'No phone service'].shape[0])
        print("OnlineSecurity特征还有%d条样本的值为 'No internet service'" %
              self.data[self.data['OnlineSecurity'] == 'No internet service'].shape[0])

        # 部分类别特征只有两类取值，可以直接用0、1代替；
        # 另外，可视化过程中发现有四列特征对结果影响可以忽略，后续直接删除。
        # 选择特征值为"Yes"和 "No" 的列名
        encodeCols = list(self.data.columns[3: 17].drop(
            ['tenure', 'PhoneService', 'InternetService', 'StreamingTV', 'StreamingMovies', 'Contract']
        ))
        for i in encodeCols:
            self.data[i] = self.data[i].map({'Yes': 1, 'No': 0})    # 用1代替"Yes"，0代替 "No"
        # 顺便把目标变量也进行编码
        self.data['Churn'] = self.data['Churn'].map({'Yes': 1, 'No': 0})

        # 其他无序的类别特征采用独热编码
        onehotCols = ['InternetService', 'Contract', 'PaymentMethod']
        churnDf = self.data['Churn'].to_frame()    # 取出目标变量列，以便后续进行合并
        featureDf = self.data.drop(['Churn'], axis=1)    # 所有特征列

        for i in onehotCols:
            onehotDf = pd.get_dummies(featureDf[i], prefix=i)
            featureDf = pd.concat([featureDf, onehotDf], axis=1)    # 编码后特征拼接到去除目标变量的数据集中

        self.data = pd.concat([featureDf, churnDf], axis=1)    # 拼回目标变量，确保目标变量在最后一列
        self.data = self.data.drop(onehotCols, axis=1)    # 删除原特征列

        # 删去无用特征 'customerID'、'gender'、 'PhoneService'、'StreamingTV'和'StreamingMovies'
        self.data = self.data.drop(['customerID', 'gender', 'PhoneService', 'StreamingTV', 'StreamingMovies'], axis=1)
        print(self.data.head(10))        # 观察此时的数据集
        nu_fea = self.data[['tenure', 'MonthlyCharges', 'TotalCharges']]    # 选择连续型数值特征计算相关系数
        nu_fea = list(nu_fea)    # 特征名列表
        pearson_mat = self.data[nu_fea].corr(method='spearman')    # 计算皮尔逊相关系数矩阵
        plt.figure(figsize=(8, 8))      # 建立图像
        sns.heatmap(pearson_mat, square=True, annot=True, cmap="YlGnBu")    # 用热度图表示相关系数矩阵

        self.data = self.data.drop(['TotalCharges'], axis=1)
        print(self.data.head(10))            # 观察此时的数据集
        if show:
            plt.show()              # 展示热度图

    def get_predict(self):
        print(":::", self.data)
        # [7043 rows x 21 columns]
        # 6.1类别不平衡问题处理
        class Smote:
            def __init__(self, samples, N, k):
                self.n_samples, self.n_attrs = samples.shape
                self.N = N
                self.k = k
                self.samples = samples
                self.newindex = 0

            def over_sampling(self):
                N = int(self.N)
                self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
                # 1.对每个少数类样本均求其在所有少数类样本中的k近邻
                neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.samples)
                for i in range(len(self.samples)):
                    nnarray = neighbors.kneighbors(self.samples[i].reshape(1, -1), return_distance=False)[0]
                    self._populate(N, i, nnarray)
                return self.synthetic

            # 2.为每个少数类样本选择k个最近邻中的N个；3.并生成N个合成样本
            def _populate(self, N, i, nnarray):
                for j in range(N):
                    nn = random.randint(0, self.k - 1)
                    dif = self.samples[nnarray[nn]] - self.samples[i]
                    gap = random.random()
                    self.synthetic[self.newindex] = self.samples[i] + gap * dif
                    self.newindex += 1

        # 每个正样本用SMOTE方法随机生成两个新的样本
        posDf = self.data[self.data['Churn'] == 1].drop(['Churn'], axis=1)    # 共1869条正样本, 取其所有特征列
        posArray = posDf.values    # pd.DataFrame -> np.array, 以满足SMOTE方法的输入要求
        newPosArray = Smote(posArray, 2, 5).over_sampling()
        newPosDf = pd.DataFrame(newPosArray)    # np.array -> pd.DataFrame
        print(newPosDf.head(10))        # 观察此时的新样本

        # 调整为正样本在数据集中应有的格式
        newPosDf.columns = posDf.columns    # 还原特征名
        cateCols = list(newPosDf.columns.drop(['tenure', 'MonthlyCharges']))   # 提取离散特征名组成的列表
        for i in cateCols:
            newPosDf[i] = newPosDf[i].apply(lambda x: 1 if x >= 0.5 else 0)    # 将特征值变回0、1二元数值
        newPosDf['Churn'] = 1    # 添加目标变量列
        print(newPosDf.head(10))        # 观察此时的新样本
        print("原本的正样本有%d条" % posDf.shape[0])
        print("原本的负样本有%d条" % (self.data.shape[0] - posDf.shape[0]))

        # 构建类别平衡的数据集
        from sklearn.utils import shuffle
        newPosDf = newPosDf[:3305]    # 直接选取前3305条样本
        data = pd.concat([self.data, newPosDf])    # 竖向拼接
        # data = shuffle(data).reset_index(drop=True)
        print("此时数据集的规模为：", data.shape)

        # K折交叉验证代码
        # from sklearn.cross_validation import KFold
        from sklearn.model_selection import KFold

        def kFold_cv(X, y, classifier, **kwargs):
            """
            :param X: 特征
            :param y: 目标变量
            :param classifier: 分类器
            :param **kwargs: 参数
            :return: 预测结果
            """
            kf = KFold(n_splits=5, shuffle=True)
            y_pred = np.zeros(len(y))  # 初始化y_pred数组

            for train_index, test_index in kf.split(X):
                X_train = X[train_index]
                X_test = X[test_index]
                y_train = y[train_index]  # 划分数据集
                clf = classifier(**kwargs)
                clf.fit(X_train, y_train)  # 模型训练
                y_pred[test_index] = clf.predict(X_test)  # 模型预测
            return y_pred

        # 模型预测
        from sklearn.linear_model import LogisticRegression as LR    # 逻辑回归
        from sklearn.svm import SVC    # SVM
        from sklearn.ensemble import RandomForestClassifier as RF    # 随机森林
        from sklearn.ensemble import AdaBoostClassifier as Adaboost    # AdaBoost
        from xgboost import XGBClassifier as XGB    # XGBoost
        # X = data.iloc[:, :-1].as_matrix()
        X = data.iloc[:, :-1].iloc[:, :].values         # Kagging
        y = data.iloc[:, -1].values
        # 此处仅做演示，因此未进行调参过程
        lr_pred = kFold_cv(X, y, LR)
        svc_pred = kFold_cv(X, y, SVC)
        rf_pred = kFold_cv(X, y, RF)
        ada_pred = kFold_cv(X, y, Adaboost)
        xgb_pred = kFold_cv(X, y, XGB)
        pred = lr_pred, svc_pred, rf_pred, ada_pred, xgb_pred
        return X, y, pred
