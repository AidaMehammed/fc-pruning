import torch
import torch_pruning as tp
import copy
import os
import torch.nn as nn
import torch.optim as optim

def get_weights(model):
    return [param.data for param in model.parameters()]

def set_weights(model, weights):
  for param, weight in zip(model.parameters(), weights):
        param.data = weight



def test(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = 100. * correct / total

        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')


def train(model, train_loader, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0


        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = out.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        print(f'Epoch [{epoch + 1}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Accuracy: {100 * train_accuracy:.2f}%, '
              )
    print('Finished Training')

def get_last_layer(model):
    last_layer = None
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            last_layer = layer
    return last_layer


def create_channel_mask(pruned_model):
    mask_list = []

    for param in pruned_model.parameters():
        mask = torch.ones_like(param)
        for idx, val in enumerate(param.view(-1)):
            if val == 0:
                mask.view(-1)[idx] = 0
        mask_list.append(mask)

    return mask_list


def soft_prune(model, imp, example_inputs, iterative_steps=None, pruning_ratio=None, ignored_layers=None):
    pmodel = copy.deepcopy(model)

    pruner = tp.pruner.MetaPruner(
        pmodel,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers
    )

    for i in range(iterative_steps):
        print('Applying soft pruning step')
        for group in pruner.step(interactive=True):
            #print(group)
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

    mask_client_list = create_channel_mask(pmodel)

    return pmodel, mask_client_list


def hard_prune_and_finetune(model, imp, example_inputs,learning_rate, iterative_steps=None, pruning_ratio=None, ignored_layers=None,
                            epochs=None, train_loader=None):
    pmodel = copy.deepcopy(model)

    pruner = tp.pruner.MagnitudePruner(
        pmodel,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
    )

    for i in range(iterative_steps):
        print('Applying hard pruning step')

        if isinstance(imp, tp.importance.TaylorImportance):
            # Taylor expansion requires gradients for importance estimation
            loss = pmodel(example_inputs).sum()  # a dummy loss for TaylorImportance
            loss.backward()  # before pruner.step()
            pruner.regularize(pmodel, loss)
            print(loss)
        pruner.step()


    # finetune model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            print(f'round:{epoch + 1}, loss:{loss.item()}')
    return pmodel




def reconstruct_model(pruned_model_params, binary_mask, reference_model):
    pruned_idx = 0

    if reference_model is not None:

        flattened_params_ref = [tensor.flatten() for tensor in get_weights(reference_model)]
        reference_model_flat = torch.cat(flattened_params_ref, dim=0)


    flattened_params = [tensor.flatten() for tensor in pruned_model_params]
    pruned_model_flat = torch.cat(flattened_params, dim=0)

    flattened_tensors = [param.view(-1) for param in binary_mask]
    binary_mask_flat = torch.cat(flattened_tensors, dim=0)

    # Number of Parameters
    nr_param = len(binary_mask_flat)

    # Reconstruct by setting all zero
    reconstructed_model_flat = torch.zeros_like(binary_mask_flat)

    for idx in range(nr_param):
      if binary_mask_flat[idx] == 1:

        reconstructed_model_flat[idx] = pruned_model_flat[pruned_idx]
        pruned_idx += 1
      else:
        if reference_model is not None:
            reconstructed_model_flat[idx] = reference_model_flat[idx]
        else:
            reconstructed_model_flat[idx]= 0.0

    desired_shape = [param_tensor.shape for param_tensor in binary_mask]

    reconstructed_model = []
    start = 0
    for shape in desired_shape:
        end = start + shape.numel()
        reconstructed_model.append(reconstructed_model_flat[start:end].reshape(shape))
        start = end

    return reconstructed_model


def average_weights(reconstructed_clients):
    # save desired_shape to reshape after averaging
    desired_shape = [param_tensor.shape for param_tensor in reconstructed_clients[0]]

    # flatten to average
    reconstructed_clients_flat = []
    for client in reconstructed_clients:
        flattened_params = [tensor.flatten() for tensor in client]
        reconstructed_client_flat = torch.cat(flattened_params, dim=0)
        reconstructed_clients_flat.append(reconstructed_client_flat)

    num_clients = len(reconstructed_clients_flat)
    total_weights = len(reconstructed_clients_flat[0])

    averaged_weights = []

    for i in range(total_weights):

        summed_weights = sum(weights[i] for weights in reconstructed_clients_flat)
        averaged_weight = summed_weights / num_clients
        averaged_weights.append(averaged_weight)


    print('Data has been averged')

    # Convert the list of tensors to a single tensor
    averaged_weights_tensor = torch.Tensor(averaged_weights)

    # reshape global model
    global_model = []
    start = 0
    for shape in desired_shape:
        end = start + shape.numel()
        global_model.append(averaged_weights_tensor[start:end].reshape(shape))
        start = end

    return global_model


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6  # size in MB
    os.remove('temp.p')
    return size


