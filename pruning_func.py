import torch.nn.utils.prune as prune
import torch
import torch_pruning as tp
import copy


def create_channel_mask(pruned_model):
    mask_list = []

    for param in pruned_model.parameters():
        # num_params = param.numel()
        mask = torch.ones_like(param)
        for idx, val in enumerate(param.view(-1)):
            if val == 0:
                mask.view(-1)[idx] = 0
        mask_list.append(mask)

    return mask_list

def soft_prune(model,imp, example_inputs, iterative_steps=None, pruning_ratio=None, ignored_layers=None):
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
        print('soft pruning-----------')
        # Soft Pruning
        for group in pruner.step(interactive=True):
            print(group)
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
def hard_prune_and_finetune(model,imp, example_inputs, iterative_steps=None, pruning_ratio=None, ignored_layers=None, epochs=None, train_loader= None):
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
        print('hard pruning-----------')

        if isinstance(imp, tp.importance.TaylorImportance):
            # Taylor expansion requires gradients for importance estimation
            loss = pmodel(example_inputs).sum()  # a dummy loss for TaylorImportance
            loss.backward()  # before pruner.step()
            pruner.regularize(pmodel, loss)
            print(loss)
        pruner.step()
    # finetune model
    optimizer = torch.optim.Adam(pmodel.parameters(), lr=0.0005)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = pmodel(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f'round:{epoch + 1}, loss:{loss.item()}')
    return pmodel

def local_structured_pruning(pruned_model, pruning_ratio):
  # can exchange to prune.ln_structured
  # or  prune.random_unstructured
  # or prune.l1_unstructured(module, name='weight', amount, importance_scores=None)

  for name, module in pruned_model.named_modules():
      # prune 20% of connections in all 2D-conv layers
      if isinstance(module, torch.nn.Conv2d):
          prune.random_structured(module, name='weight', amount=pruning_ratio, dim=0)
      # prune 20% of connections in all linear layers
      elif isinstance(module, torch.nn.Linear):
          prune.random_structured(module, name='weight', amount=pruning_ratio, dim=0)

  for module, _ in pruned_model:
      prune.remove(module, 'weight')


def global_unstructured_pruning(pruned_model, pruning_ratio):
  parameters_to_prune = (
    (pruned_model.conv1, 'weight'),
    (pruned_model.conv2, 'weight'),
    (pruned_model.fc1, 'weight'),
    (pruned_model.fc2, 'weight'),
    (pruned_model.fc3, 'weight'))


  prune.global_unstructured(
      parameters_to_prune,
      pruning_method=prune.L1Unstructured,
      amount=pruning_ratio)

  for module, _ in parameters_to_prune:
     prune.remove(module, 'weight')
