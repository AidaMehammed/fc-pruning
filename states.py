
from FeatureCloud.app.engine.app import AppState, app_state, Role
from typing import TypedDict, Union, List, Type
from torch.utils.data import Dataset, DataLoader
import utils as pf
import torch
import torch_pruning as tp
import bios
import importlib.util


INPUT_DIR = '/mnt/input'
OUTPUT_DIR = '/mnt/output'



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



@app_state('initial')
class InitialState(CompressAppState):

    def register(self):
        # Register transition for local update
        self.register_transition('local_update',label="Broadcast initial weights")


    def run(self) :

        # Reading configuration file
        self.log('Reading configuration file ...')

        # Loading configuration from file
        config = bios.read(f'{INPUT_DIR}/config.yml')

        max_iterations = config['max_iter']
        self.store('iteration', 0)
        self.store('max_iterations', max_iterations)
        self.store('learning_rate', config['learning_rate'])
        self.store('learning_rate_pr', config['learning_rate_finetune'])
        self.store('epochs', config['epochs'])
        self.store('pruning_ratio', config['pruning_ratio'])
        self.store('iterative_steps', config['iterative_steps'])
        self.store('imp', eval(config['imp']))
        self.store('batch_size', config['batch_size'])

        shape_ex = config['example_input']
        shape = tuple(map(int, shape_ex.strip('()').split(', ')))
        ex_input = torch.randn(*shape)
        self.store('ex_input', ex_input)


        train_dataset_path = f"{INPUT_DIR}/{config['train_dataset']}"
        test_dataset_path = f"{INPUT_DIR}/{config['test_dataset']}"
        train_dataset = torch.load(train_dataset_path)
        test_dataset = torch.load(test_dataset_path)
        self.store('train_dataset', train_dataset)
        self.store('test_dataset', test_dataset)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.load('batch_size'), shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.load('batch_size'), shuffle=False)
        self.store('train_loader', train_loader)
        self.store('test_loader', test_loader)

        self.log('Done reading configuration.')

        # Loading and preparing initial model
        self.log('Preparing initial model ...')
        model_path = f"{INPUT_DIR}/{config['model']}"

        # Loading model from file
        spec = importlib.util.spec_from_file_location("model_module", model_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        model_class_name = config.get('model_class', 'Model')
        model = getattr(model_module, model_class_name)()#
        # Storing model
        self.store('model', model)



        self.log('Transition to local state ...')

        if self.is_coordinator:
            # Broadcasting initial weights to participants
            self.broadcast_data([pf.get_weights(model),False], send_to_self=False)
            self.store('received_data', pf.get_weights(model))

        return 'local_update'





@app_state('local_update', Role.BOTH)
class LocalUpdate(CompressAppState):
    def register(self):
        # Registering transitions for local update
        self.register_transition('aggregation', Role.COORDINATOR, label="Gather local models")
        self.register_transition('local_update', Role.PARTICIPANT, label="Wait for global model")
        self.register_transition('terminal',label="Terminate process")


    def run(self):
        # Running local update process

        iteration = self.load('iteration')
        self.log(f'ITERATION  {iteration}')
        model = self.load('model')
        stop_flag = False
        if self.is_coordinator:
            received_data = self.load('received_data')
        else:
            received_data, stop_flag = self.await_data(unwrap=True)
        self.log(len(received_data))

        if stop_flag:
            self.log('Stopping')
            return 'terminal'

        # Receive global model from coordinator
        self.log('Receive model from coordinator')
        pf.set_weights(model,received_data)

        # Receive dataframe
        train_loader = self.load('train_loader')
        it_steps = self.load('iterative_steps')
        epochs = self.load('epochs')
        pr = self.load('pruning_ratio')
        imp = self.load('imp')
        self.log(type(imp))
        ex_input = self.load('ex_input')
        ignored_layers = self.load('ignored_layers')
        learning_rate = self.load('learning_rate')
        learning_rate_pr = self.load('learning_rate_pr')
        test_loader = self.load('test_loader')



        # Train local model
        self.log('Training local model ...')
        pf.train(model, train_loader, epochs=epochs, learning_rate=learning_rate)
        # Pruning local model
        self.log('Pruning local model ...')

        self.configure_pruning(iterative_steps=it_steps, epochs=epochs, pruning_ratio=pr,learning_rate=learning_rate_pr,
                               model=model, reference_model=model, imp=imp,
                               ex_input=ex_input,
                               ignored_layers=ignored_layers, train_loader=train_loader)


        self.send_data_to_coordinator(model, True, use_smpc=False, use_dp=False)

        # Test pruned model
        pf.test(model, test_loader)

        iteration += 1
        self.store('iteration', iteration)

        if stop_flag:
            return 'terminal'

        if self.is_coordinator:
            return 'aggregation'

        else:
            return 'local_update'

@app_state('aggregation', Role.COORDINATOR)
class AggregateState(CompressAppState):

    def register(self):
        # Registering transitions for aggregation state
        self.register_transition('local_update', Role.COORDINATOR, label="Broadcast global model")
        self.register_transition('terminal', Role.COORDINATOR, label="Terminate process")

    def run(self) :
        # Running aggregation process
        self.log(f'Aggregating Data ...')
        # Gathering and averaging data
        data = self.gather_data(use_pruning=True, is_json=False, use_smpc=False, use_dp=False, memo=None)
        self.log(f'Averaging Data ...')
        global_averaged_weights = pf.average_weights(data)

        stop_flag = False
        if self.load('iteration') >= self.load('max_iterations'):
            stop_flag = True

        # Set averaged_weights as new global model
        self.store('received_data', global_averaged_weights)
        new_model= self.load('model')
        pf.set_weights(new_model, global_averaged_weights)

        # Broadcasting global model
        self.log('Broadcasting global model ...')
        self.broadcast_data([global_averaged_weights, stop_flag], send_to_self=False)


        if stop_flag:
            return 'terminal'

        return 'local_update'

