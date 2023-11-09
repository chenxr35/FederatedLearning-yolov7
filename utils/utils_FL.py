import torch
from torch import nn


def update_momentum_weights(m_flat, dg_flat, new_flat, beta=0.9):
    m_flat_new = beta * m_flat + (dg_flat - new_flat)
    ag_flat = dg_flat - m_flat_new  # new_flat -> dg_flat

    return m_flat_new, ag_flat


def flatten_weights(model, from_dict=False, numpy_output=True):
    """
    Flattens a PyTorch model. i.e., concat all parameters as a single, large vector.
    :param model: PyTorch model
    :param numpy_output: should the output vector be casted as numpy array?
    :return: the flattened (vectorized) model parameters either as Numpy array or Torch tensors
    """
    all_params = []

    if from_dict:
        for param in model.values():
            all_params.append((param.clone().detach()).view(-1))

    else:
        for param in model.parameters():
            all_params.append((param.clone().detach()).view(-1))
    all_params = torch.cat(all_params)

    if numpy_output:
        return all_params.cpu().clone().detach().numpy()

    return all_params


def assign_weights(model_dict, weights):
    """
    Manually assigns `weights` of a Pytorch `model`.
    Note that weights is of vector form (i.e., 1D array or tensor).
    Usage: For implementation of Mode Connectivity SGD algorithm.
    :param model_dict: Pytorch model.
    :param weights: A flattened (i.e., 1D) weight vector.
    :return: The `model` updated with `weights`.
    """
    # The index keeps track of location of current weights that is being un-flattened.
    index = 0
    # just for safety, no grads should be transferred.
    with torch.no_grad():
        for param in model_dict.keys():
            # ignore batchnorm params
            if (
                "running_mean" in param
                or "running_var" in param
                or "num_batches_tracked" in param
            ):
                continue
            param_count = model_dict[param].numel()
            param_shape = model_dict[param].shape
            model_dict[param] = nn.Parameter(
                torch.from_numpy(
                    weights[index: index + param_count].reshape(param_shape)
                )
            )
            index += param_count
    return model_dict