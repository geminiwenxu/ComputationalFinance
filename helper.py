# helper function: fit the RNN model
import time

import matplotlib.pyplot as plt
import numpy as np
from darts.metrics import mape, rmse, r2_score


def fit_it(model, train, val, flavor, covariates):
    t_start = time.perf_counter()
    print("\nbeginning the training of the {0} RNN:".format(flavor))

    res = model.fit(train,
                    future_covariates=covariates,
                    val_series=val,
                    val_future_covariates=covariates,
                    verbose=True)

    res_time = time.perf_counter() - t_start
    print("training of the {0} RNN has completed:".format(flavor), f'{res_time:.2f} sec')

    return res


# helper function: plot the predictions

def plot_fitted(pred, act, flavor):
    plt.figure(figsize=(12, 5))
    act.plot(label='actual')
    pred.plot(label='prediction')
    plt.title("RNN: {0} flavor".format(flavor) + ' | MAPE: {:.2f}%'.format(mape(pred, act)))
    plt.legend()
    plt.show()


# helper function: compute accuracy metrics

def accuracy_metrics(pred, act):
    act2 = act.slice_intersect(pred)
    pred2 = pred.slice_intersect(act2)
    resid = pred2 - act2
    sr = resid.pd_series()
    sa = act2.pd_series()
    sp = pred2.pd_series()
    res_mape = mape(pred2, act2)
    res_r2 = r2_score(pred2, act2)
    res_rmse = rmse(pred2, act2)
    res_pe = sr / sa
    n_act = len(act2)
    res_rmspe = np.sqrt(np.sum(res_pe ** 2) / n_act)  # root mean square percentage error
    res_std = np.std(sr)  # std error of the model = std deviation of the noise
    res_se = res_std / np.sqrt(n_act)  # std error in estimating the mean
    res_sefc = np.sqrt(res_std + res_se ** 2)  # std error of the forecast

    res_accuracy = {
        "MAPE": res_mape, "RMSPE": res_rmspe, "RMSE": res_rmse,
        "-R squared": -res_r2, "se": res_sefc}
    return res_accuracy
