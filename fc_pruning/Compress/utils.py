import torch_pruning as tp
import torch
import torch.nn as nn
import copy
import os
import sys



def get_weights(model):
    return [param.data for param in model.parameters()]


def get_last_layer(model):
    last_layer = None
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            last_layer = layer
    return last_layer



def create_channel_mask(pruned_model):
    mask_indices = []

    # Flatten the parameters of the pruned model
    flattened_params = [tensor.flatten() for tensor in pruned_model.parameters()]
    pruned_model_flat = torch.cat(flattened_params, dim=0)

    for idx, val in enumerate(pruned_model_flat):
        if val != 0:
            # If the value is not pruned, save the index
            mask_indices.append(idx)

    mask_indices = torch.tensor([mask_indices], dtype=torch.int32)

    return mask_indices

def soft_prune(model, imp, example_inputs, pruning_ratio=None, ignored_layers=None):
    # peforms pruning by just zeroing out the groped parameters
    pmodel = copy.deepcopy(model)

    pruner = tp.pruner.MetaPruner(
        pmodel,
        example_inputs,
        importance=imp,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers
    )

    for group in pruner.step(interactive=True):
        for dep, idxs in group:
            target_layer = dep.target.module
            pruning_fn = dep.handler
            if pruning_fn in [tp.prune_conv_in_channels, tp.prune_linear_in_channels]:
                target_layer.weight.data[:, idxs] *= 0
            elif pruning_fn in [tp.prune_conv_out_channels, tp.prune_linear_out_channels]:
                target_layer.weight.data[idxs] *= 0
                if target_layer.bias is not None:
                    target_layer.bias.data[idxs] *= 0
            elif pruning_fn in [tp.prune_batchnorm_out_channels]:
                target_layer.weight.data[idxs] *= 0
                target_layer.bias.data[idxs] *= 0


    # create mask with non pruned indices
    mask_client_list = create_channel_mask(pmodel)

    return pmodel, mask_client_list


def hard_prune(model, imp, example_inputs, pruning_ratio=None, ignored_layers=None):
    # actual pruning will be performed here by removing unnecessary weights
    pmodel = copy.deepcopy(model)

    pruner = tp.pruner.MetaPruner(
        pmodel,
        example_inputs,
        importance=imp,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
    )


    if isinstance(imp, tp.importance.TaylorImportance):
        # Taylor expansion requires gradients for importance estimation
        loss = pmodel(example_inputs).sum()  # a dummy loss for TaylorImportance
        loss.backward()  # before pruner.step()
        print(loss)
    pruner.step()

    return pmodel


def reconstruct_model(mask_idx, pruned_model_params, reference_model):
    pruned_idx = 0

    # use set to speed up iterating
    mask = set(mask_idx.tolist()[0])


    flattened_params_ref = [tensor.flatten() for tensor in get_weights(reference_model)]
    reference_model_flat = torch.cat(flattened_params_ref, dim=0)

    flattened_params = [tensor.flatten() for tensor in pruned_model_params]
    pruned_model_flat = torch.cat(flattened_params, dim=0)


    # Number of Parameters
    nr_param = len(reference_model_flat)

    # Reconstruct by setting all zero
    reconstructed_model_flat = torch.zeros_like(reference_model_flat)

    for idx in range(nr_param):
      if idx in mask:
        reconstructed_model_flat[idx] = pruned_model_flat[pruned_idx]
        pruned_idx += 1
      else:
        reconstructed_model_flat[idx] = reference_model_flat[idx]


    desired_shape = [param_tensor.shape for param_tensor in get_weights(reference_model)]

    reconstructed_model = copy.deepcopy(reference_model)
    # Assign the reconstructed parameters to the new model
    start = 0
    for param, shape in zip(reconstructed_model.parameters(), desired_shape):
        end = start + shape.numel()
        param.data = reconstructed_model_flat[start:end].reshape(shape)
        start = end


    return reconstructed_model



def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6  # size in MB
    os.remove('temp.p')
    return size