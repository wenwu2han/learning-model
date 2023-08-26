import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.arima import auto_arima


# 数据预处理
def pre_process_data(data):
    diff_data = data.diff().dropna()
    return diff_data


# 拟合ARIMA模型
def fit_ARIMA_model(data):
    model = auto_arima(data, start_p=1, start_q=1, max_p=5, max_q=5, d=None,
                       trace=True, error_action='ignore', suppress_warnings=True)
    return model


# 预测数据
def predict_data(model, steps):
    forecast = model.predict(n_periods=steps)
    return forecast


# 评估模型
def evaluate_model(train_data, test_data, model):
    history = [x for x in train_data]
    predictions = list()
    for t in range(len(test_data)):
        model_fit = model.fit(history)
        output = model_fit.forecast()
        yhat = output[0][0]
        predictions.append(yhat)
        obs = test_data[t]
        history.append(obs)
    mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
    mae = np.mean(np.abs(test_data - predictions))
    mse = np.mean((test_data - predictions) ** 2)
    return {'MAPE': mape, 'MAE': mae, 'MSE': mse}


if __name__ == "__main__":
    # 读取数据
    data = pd.read_csv('inventory_data.csv', index_col=0)
    train_data = data.iloc[:-5, :]
    test_data = data.iloc[-5:, :]

    # 数据预处理
    diff_data = pre_process_data(train_data)

    # 拟合ARIMA模型
    model = fit_ARIMA_model(diff_data.values)

    # 预测数据
    forecast = predict_data(model, len(test_data))

    # 评估模型
    eval_result = evaluate_model(diff_data.values, test_data.values, model)

    print('预测结果：', forecast)
    print('MAPE:', eval_result['MAPE'])
    print('MAE:', eval_result['MAE'])
    print('MSE:', eval_result['MSE'])