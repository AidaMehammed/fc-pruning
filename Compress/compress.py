from FeatureCloud.app.engine.app import AppState
from typing import TypedDict, Union, List, Type
from torch.utils.data import DataLoader
import torch
import torch_pruning as tp

class PruningType(TypedDict):
    model: torch.nn.Module
    reference_model: Union[List[torch.nn.Module], None]
    imp: Type[tp.pruner.importance.Importance]
    ignored_layers: Union[List[torch.nn.Module], None]
    train_loader: DataLoader
    epochs: int
    iterative_steps: int
    pruning_ratio: float
    learning_rate:float


class CompressAppState(AppState):

    def configure_pruning(self, iterative_steps: int = 0, epochs: int = 0, pruning_ratio : float = 0.5,learning_rate : float = 0.001, model: torch.nn.Module = None, reference_model: Union[List[torch.nn.Module]] = None,
                         imp: Type[tp.pruner.importance.Importance] = tp.importance.MagnitudeImportance(p=2), ex_input : torch.Tensor = None,
                         ignored_layers: Union[List[torch.nn.Module], None] = None , train_loader: DataLoader = None):

        '''
        Configures the pruning settings for your model.
        Parameters
        ----------
        model : nn.Module
            Your PyTorch model.
        reference_model : nn.Module
            Could be same model as your original model.
            If None is given it will fill out the pruned values with zeroes.
        imp : tp.importance
            Importance method for pruning.
        ex_input : torch.Tensor
            Example input tensor.
        ignored_layers : Union[List[nn.Module], None], optional
            List of layers to be ignored during pruning, mostly last layer.
        epochs : int
            Number of training epochs for pruning.
        iterative_steps : int
            Number of iterative pruning steps.
        pruning_ratio : float
            Pruning ratio, should be between 0 and 1. Default value is 0.5.
        train_loader : DataLoader
            DataLoader for training data.
        '''

        if self.load('default_pruning') is None:
            self.store('default_pruning', PruningType())

        default_pruning = self.load('default_pruning')

        updated_pruning = default_pruning.copy()

        updated_pruning['model'] = model
        updated_pruning['reference_model'] = reference_model
        updated_pruning['epochs'] = epochs
        updated_pruning['iterative_steps'] = iterative_steps
        updated_pruning['pruning_ratio'] = pruning_ratio
        updated_pruning['imp'] = imp
        updated_pruning['ex_input'] = ex_input
        updated_pruning['ignored_layers'] = ignored_layers
        updated_pruning['train_loader'] = train_loader
        updated_pruning['learning_rate_pr'] = learning_rate


        self.store('default_pruning', updated_pruning)




    def gather_data(self, use_pruning=True, **kwargs):
        '''
                Gathers data for federated learning, including pruning if enabled.

                Parameters
                ----------
                use_pruning : bool, optional
                    Flag to indicate whether to use pruning. Default is True.


                Returns
                -------
                data : list
                    List of data to be sent to the coordinator.
                '''
        data_with_mask_list = super().gather_data(**kwargs)

        if use_pruning:
            reference_model= self.load('reference_model')

            # extract data and binary_mask
            data = []
            binary_mask = []
            for data_with_mask in data_with_mask_list:
                data.append(data_with_mask[:-1])
                binary_mask.append(data_with_mask[-1])

            # reconstruct data after pruning
            reconstructed_clients = []
            for i in range(len(data)):
                reconstructed_model = pf.reconstruct_model(data[i], binary_mask[i][0], reference_model)
                reconstructed_clients.append(reconstructed_model)
            data = reconstructed_clients
        else:
            data = data_with_mask_list

        return data

    def send_data_to_coordinator(self, data, use_pruning= True, **kwargs):
        '''
            Sends data to the coordinator, including pruning if enabled.

            Parameters
            ----------
            data : list
                List of data to be sent to the coordinator.
            use_pruning : bool, optional
                Flag to indicate whether to use pruning. Default is True.

            Returns
            -------
            data_with_mask : list
                List of data with masks for pruning.
            '''
        if use_pruning:

            default_pruning = self.load('default_pruning')

            example_input = default_pruning['ex_input']
            ignored_layers = default_pruning['ignored_layers']
            pruning_ratio = default_pruning['pruning_ratio']
            model = data
            reference_model = default_pruning['reference_model']
            epochs = default_pruning['epochs']
            train_loader = default_pruning['train_loader']
            iterative_steps = default_pruning['iterative_steps']
            imp = default_pruning['imp']
            learning_rate_pr = default_pruning['learning_rate_pr']

            self.store('default_pruning', default_pruning)
            self.store('reference_model', reference_model)



            self.log('start pruning...')

            # exclude last_layer from pruning
            last_layer = pf.get_last_layer(model)


            if ignored_layers is None:
                ignored_layers = []
            if last_layer not in ignored_layers:
                ignored_layers.append(last_layer)



            self.log(f'Size of model before pruning: {pf.print_size_of_model(model)} MB')

            binary_mask = []

            model, mask_client_list = pf.soft_prune(model, imp, example_input, iterative_steps,
                                            pruning_ratio, ignored_layers)
            binary_mask.append(mask_client_list)
            model = pf.hard_prune_and_finetune(model, imp, example_input,learning_rate_pr, iterative_steps,
                                            pruning_ratio, ignored_layers, epochs,
                                            train_loader)


            self.log(f'Size of model after pruning: {pf.print_size_of_model(model)} MB')



            data_with_mask = pf.get_weights(model) + [binary_mask]

            super().send_data_to_coordinator(data_with_mask,**kwargs)
        else:
            super().send_data_to_coordinator(data, **kwargs)
            data_with_mask = data
        return data_with_mask


