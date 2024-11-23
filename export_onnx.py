""" Export model to onnx format to make it compatible for any device (For Deployment) """

import torch
import argparse

from collections import OrderedDict
from assets.model import LSTMModel


def main(args):
    model = LSTMModel(input_dim=1001,
                      embedding_dim=64,
                      hidden_dim=256,
                      output_dim=1,
                      num_layers=2,
                      dropout=.2)

    # Load trained model from checkpoint
    checkpoint = torch.load(args.checkpoint_path, weights_only=True, map_location="cpu")
    model_state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = OrderedDict()

    for k, v in model_state_dict.items():
        if k.startswith('_orig_mod.'):
            name = k.replace('_orig_mod.', '')
            new_state_dict[name] = v

    # Load state dictionaries into the model
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # Export the model to ONNX
    filepath = "model.onnx"
    input_sample = torch.randint(0, 1000, (1, 1000), dtype=torch.long)


    try:
        torch.onnx.export(
            model,
            input_sample,
            filepath,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f'Finished Model Conversion to ONNX format saved at {filepath}')

    except Exception as e:
        print(f"An error occurred during ONNX export: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX Conversion")
    parser.add_argument("-cp", "--checkpoint_path", type=str, required=True, help="File path to the PyTorch checkpoint model")
    args = parser.parse_args()
    main(args)