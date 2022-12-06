import logging
import warnings

from prepare_data import prepare_data
from rnn import run_RNN

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    logging.disable(logging.CRITICAL)

    FC_N = 36  # forecast periods
    FC_STRIDE = 10
    FC_START = "19590101"  # period at which to split training and validation dataset
    EPOCH = 300  # for testing or debugging, rather than real forecasts,
    # run 3 different flavors of RNN on the time series:
    flavors = ["LSTM", "RNN"]

    # call the RNN model setup for each of the 3 RNN flavors
    ts_trf, train_trf, val_trf, cov_train, cov_val, covariates, periodicity = prepare_data(FC_START)
    res_flavors = [run_RNN(flv, ts_trf, train_trf, val_trf, covariates, periodicity, EPOCH,FC_N) for flv in flavors]
