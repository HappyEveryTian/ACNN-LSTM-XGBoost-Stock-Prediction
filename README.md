# 基于注意力机制的CNN-LSTM和XGBoost混合模型预测分析-股票预测

## Requirements

代码在Python3.7.4版本下运行，以下是各个依赖版本：
```
numpy==1.21.6
sklearn==0.21.3
statsmodels==0.10.1
pandas==1.2.1
tensorflow==2.1.0
keras==2.3.1
xgboost==1.5.0
matplotlib==3.5.3
tushare==1.4.19
```
你也可以根据requirements.txt进行快速安装。

## 模型预测
目前模型预测分析如下：

| 00003.HK |  ARIMA  |  LSTM   | ARIMA-LSTM | ARIMA-XGBoost | CNN-LSTM | CNN-LSTM-Attention | CNN-LSTM-Attention-XGBoost |
|:--------:|:-------:|:-------:|:----------:|:-------------:|:--------:|:------------------:|:--------------------------:|
|   MSE    | 0.00618 | 0.00933 |  0.00001   |    0.00676    | 0.02440  |      0.02231       |          0.00560           |
|   RMSE   | 0.07861 | 0.09661 |  0.00359   |    0.08047    | 0.15619  |      0.14935       |          0.07486           |
|   MAE    | 0.06024 | 0.06990 |  0.00282   |    0.06154    | 0.13604  |      0.12909       |          0.05650           |
|    R2    | 0.93695 | 0.90753 |  0.99987   |    0.93398    | 0.51541  |      0.55690       |          0.93840           |

| 601988.SH |  ARIMA  |  LSTM   | ARIMA-LSTM | ARIMA-XGBoost | CNN-LSTM | CNN-LSTM-Attention | CNN-LSTM-Attention-XGBoost |
|:---------:|:-------:|:-------:|:----------:|:-------------:|:--------:|:------------------:|:--------------------------:|
|    MSE    | 0.00027 | 0.00026 |  0.00027   |    0.00030    | 0.00883  |      0.00931       |          0.00020           |
|   RMSE    | 0.01633 | 0.01612 |  0.01633   |    0.01727    | 0.09399  |      0.09648       |          0.01430           |
|    MAE    | 0.01187 | 0.01182 |  0.01187   |    0.01248    | 0.07110  |      0.07505       |          0.01116           |
|    R2     | 0.84458 | 0.81704 |  0.84191   |    0.82613    | 0.77739  |      0.76540       |          0.89297           |