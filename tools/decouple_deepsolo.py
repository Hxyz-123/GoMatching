import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="decouple deepsolo backbone & transformer", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", type=str, help='path to origin deepsolo model')
    parser.add_argument("--output", type=str, help='output path')
    return parser.parse_args()

args = parse_args()
model = torch.load(args.input, map_location='cpu')
new_model = {'model': {}}
for k, v in model['model'].items():
    if 'detection_transformer.backbone' in k:
        value = v
        new_k = k.split('detection_transformer.')[-1]
        new_model['model'][new_k] = value
    else:
        new_model['model'][k] = v
torch.save(new_model, args.output)