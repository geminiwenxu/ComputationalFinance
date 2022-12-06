# %matplotlib inline


import matplotlib.pyplot as plt
import pandas as pd
from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.utils.statistics import check_seasonality
from darts.utils.timeseries_generation import datetime_attribute_timeseries




# set EPOCH to a low value like 3; for the real deal: 300
# 300 will take as much as 30 - 50 minutes of processing time
# set up, fit, run, plot, and evaluate the RNN model
# helper function: fit the RNN model
# helper function: plot the predictions

def prepare_data(FC_START):
    ## load data
    ts = AirPassengersDataset().load()

    series = ts
    df = ts.pd_dataframe()
    # print(df)
    plt.figure(100, figsize=(12, 5))
    # series.plot()
    # plt.show()

    # analyze its seasonality
    is_seasonal, periodicity = check_seasonality(ts, max_lag=240)
    dict_seas = {
        "is seasonal?": is_seasonal,
        "periodicity (months)": f'{periodicity:.1f}',
        "periodicity (~years)": f'{periodicity / 12:.1f}'}
    _ = [print(k, ":", v) for k, v in dict_seas.items()]

    # split training vs test dataset
    train, val = ts.split_after(pd.Timestamp(FC_START))

    # normalize the time series
    trf = Scaler()
    # fit the transformer to the training dataset
    train_trf = trf.fit_transform(train)
    # apply the transformer to the validation set and the complete series
    val_trf = trf.transform(val)
    ts_trf = trf.transform(ts)

    # create month and year covariate series
    year_series = datetime_attribute_timeseries(
        pd.date_range(start=series.start_time(),
                      freq=ts.freq_str,
                      periods=1000),
        attribute='year',
        one_hot=False)
    year_series = Scaler().fit_transform(year_series)

    month_series = datetime_attribute_timeseries(
        year_series,
        attribute='month',
        one_hot=True)

    covariates = year_series.stack(month_series)
    cov_train, cov_val = covariates.split_after(pd.Timestamp(FC_START))
    return ts_trf, train_trf, val_trf, cov_train, cov_val, covariates, periodicity
