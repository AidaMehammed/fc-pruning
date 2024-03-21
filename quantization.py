# define self.quantize in model

import torch
import torch.nn as nn
import torch.nn.functional as F


def post_static_quant(model, device, train_loader, quantize=False, fbgemm=False):
    model.to(device)
    model.eval()

    if quantize:
        model.quantize = quantize

        # possible model fusion

        if fbgemm:
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        else:
            model.qconfig = torch.quantization.default_qconfig

        torch.quantization.prepare(model, inplace=True)
        model.eval()

        # Calibrate with the training data

        with torch.no_grad():
            for data, target in train_loader:
                model(data)

        torch.quantization.convert(model, inplace=True)

    print(model)


def apply_qat(client, global_weights):
    client.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    client.model.train()
    quantize = True
    client.model.quantize = quantize

    torch.quantization.prepare_qat(client.model, inplace=True)

    print("Training QAT Model...")
    updated_weights = client.train_model(global_weights, 2)

    quantized_model = torch.quantization.convert(client.model, inplace=True)
