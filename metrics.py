import numpy as np


def RSE(pred, true):
    denom = np.sqrt(np.sum((true - true.mean()) ** 2)) + 1e-12
    return np.sqrt(np.sum((true - pred) ** 2)) / denom


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return (u / d).mean()


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    denom = np.maximum(np.abs(true), 1e-5)
    return np.mean(np.abs((pred - true) / denom))


def MSPE(pred, true):
    denom = np.maximum(np.abs(true), 1e-5)
    return np.mean(np.square((pred - true) / denom))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    return mae, mse, rmse, mape, mspe, rse, corr
