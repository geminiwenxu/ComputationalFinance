from darts.models import RNNModel

from helper import fit_it, plot_fitted, accuracy_metrics


def run_RNN(flavor, ts, train, val, covariates, periodicity, EPOCH,FC_N):
    # set the model up
    model_RNN = RNNModel(
        model=flavor,
        model_name=flavor + str(" RNN"),
        input_chunk_length=periodicity,
        training_length=20,
        hidden_dim=20,
        batch_size=16,
        n_epochs=EPOCH,
        dropout=0,
        optimizer_kwargs={'lr': 1e-3},
        log_tensorboard=True,
        random_state=42,
        force_reset=True)

    if flavor == "RNN": flavor = "Vanilla"

    # fit the model
    fit_it(model_RNN, train, val, flavor, covariates)

    # compute N predictions
    pred = model_RNN.predict(n=FC_N, future_covariates=covariates)

    # plot predictions vs actual
    plot_fitted(pred, ts, flavor)

    # print accuracy metrics
    res_acc = accuracy_metrics(pred, ts)
    print(flavor + " : ")
    _ = [print(k, ":", f'{v:.4f}') for k, v in res_acc.items()]

    return [pred, res_acc]
