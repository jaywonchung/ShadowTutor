import argparse
from pathlib import Path

import torch


def mpiscript(host, env):
    filepath = Path.home() / 'mpi.sh'
    if host == 'server':
        with open(filepath, 'w') as f:
            f.write('source ~/anaconda3/etc/profile.d/conda.sh\n')
            f.write(f'conda activate {env}\n')
            f.write(f'cd {Path.cwd()}\n')
            f.write('python dod_server.py $@')
    elif host == 'client':
        with open(filepath, 'w') as f:
            f.write(f'source ~/{env}/bin/activate\n')
            f.write(f'cd {Path.cwd()}\n')
            f.write('python dod_client.py $@\n')
    
    print(f'Saved {filepath} for {host}!')

def amendckpt(file_path, save_path):
    # Load pretrained student model checkpoint (81 classes)
    state_dict = torch.load(file_path)
    if 'out_conv3.weight' not in state_dict:  # Directly from detectron2 training
        state_dict = state_dict['model']

    # map original class numbers to new class numbers:
    # ['person', 'bicycle', 'auto', 'bird', 'dog', 'horse', 'elephant', 'giraffe', 'background']
    w = state_dict['out_conv3.weight']
    state_dict['out_conv3.weight'] = torch.stack([
        w[0], w[1], torch.mean(w[[2,5,7]],0), w[14], w[16], w[17], w[20], w[23], w[80]
    ], dim=0)
    b = state_dict['out_conv3.bias']
    state_dict['out_conv3.bias'] = torch.stack([
        b[0], b[1], torch.mean(b[[2,5,7]],0), b[14], b[16], b[17], b[20], b[23], b[80]
    ])

    # Save amended checkpoint
    torch.save(state_dict, save_path)

    print(f'Saved amended checkpoint to {save_path}!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Generate bash script for mpi (~/mpi.sh)
    # OpenMPI will invoke this script to run ShadowTutor
    mpiscript_parser = subparsers.add_parser('mpiscript')
    mpiscript_parser.add_argument('--host', choices=['server', 'client'], required=True, help='What device are you on?')
    mpiscript_parser.add_argument('--env', type=str, required=True, 'What is the name of your python virtual environment?')

    # Ammend the final layer of the pretrained student model
    # to output logits for 9 classes for the LVS dataset
    # You probably don't need to do this yourself.
    # The amended checkpoint is already in the git repo.
    amendckpt_parser = subparsers.add_parser('amendckpt')
    amendckpt_parser.add_argument('--file-path', type=str, required=True, help='Path to pretrained checkpoint file')
    amendckpt_parser.add_argument('--save-path', type=str, required=True, help='Path to save amended checkpoint')

    args = parser.parse_args()
    
    if args.command == 'mpiscript':
        mpiscript(args.host, args.env)
    
    elif args.command == 'amendckpt':
        amendckpt(args.file_path, args.save_path)

    else:
        raise NotImplementedError
