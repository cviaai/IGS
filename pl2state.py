import torch
import argparse
from collections import OrderedDict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt',
        type=str,
        required=True
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True
    )
    args = parser.parse_args()
    state = torch.load(args.ckpt, map_location=torch.device('cpu'))['state_dict']
    new_state = OrderedDict()
    for k, v in state.items():
        new_state[k[4:]] = v
    torch.save(new_state, args.out)
