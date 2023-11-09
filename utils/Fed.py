import torch
import copy
import numpy as np

from .utils_FL import update_momentum_weights, flatten_weights, assign_weights


def FedAvg(w, n_k):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = torch.mul(w_avg[k], n_k[0])
        for i in range(1, len(w)):
            w_avg[k] = torch.add(w_avg[k], w[i][k], alpha=n_k[i])
        w_avg[k] = torch.div(w_avg[k], sum(n_k))
    return w_avg


def FedAvgM(w, w_glob, n_k, server_momentum, avgm_beta):
    w_avg = FedAvg(w, n_k)

    flat_glob, flat_avg = (
        flatten_weights(w_glob, from_dict=True),
        flatten_weights(w_avg, from_dict=True),
    )

    server_momentum, flat_avgm = update_momentum_weights(
        server_momentum, flat_glob, flat_avg, avgm_beta
    )

    # Update Global Server Model
    w_avgm = assign_weights(w_glob, flat_avgm)

    return w_avgm, server_momentum