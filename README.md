# [FeatureCloud Pruning](https://github.com/AidaMehammed/FC_Pruning_App)
### Model Compression with Pruning

The FC Pruning package offers a streamlined approach to model compression using advanced pruning techniques. With support for federated learning frameworks and integration with the [Torch-Pruning](https://github.com/VainF/Torch-Pruning/tree/master) library, this package enables efficient distributed training, suitable for a wide range of machine learning tasks.
1. Train Local Model


    
2. Configure Quantization:
    - Set up initial quantization settings.
      ```python
        self.configure_pruning(pruning_ratio, model, reference_model, imp,ex_input,
                               ignored_layers)        
      ```



 

3. Send Data to Coordinator and perform Pruning:
    - Send the prepared model data to the coordinator.
   ```python
    self.send_data_to_coordinator(model, use_pruning=True, use_smpc=False, use_dp=False)
   ```