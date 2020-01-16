

import argparse
from argparse import Namespace
import os
# workaround to unpickle olf model files
import sys
import time
from tqdm import tqdm

import numpy as np
import torch

import h5py as hf
import foundation as fd
from foundation import util

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

# %matplotlib tk
# import matplotlib.pyplot as plt



from mujoco_py import GlfwContext
GlfwContext(offscreen=True)

sys.path.append('a2c_ppo_acktr')


def main(argv=None):

    parser = argparse.ArgumentParser(description='record (images only) sequences with a trained policy')

    parser.add_argument(
        '--save-dir', type=str, default=None,
    )

    parser.add_argument('-n', '--num-batches', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-s', '--max-steps', type=int, default=1000)

    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--env-name',
        default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--policy-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--non-det',
        action='store_true',
        default=False,
        help='whether to use a non-deterministic policy')
    args = parser.parse_args(argv)

    args.det = not args.non_det


    policy_name = args.env_name + '.pt'

    if policy_name not in os.listdir(args.policy_dir):
        print('ERROR: could not find policy in provided policy-dir')
        found = [p for p in os.listdir(args.policy_dir) if '.pt' in p]
        if len(found):
            print('Policies must be saved in the provided dir using the template: [env_name].pt')
            print('Found {} policies for other environments though:')
            for f in found:
                print(f.split('.')[0])
            return 1

    env = make_vec_envs(
        args.env_name,
        args.seed + 1000,
        args.batch_size,
        None,
        None,
        device='cpu',
        allow_early_resets=False)

    # Get a render function
    # render_func = get_render_func(env)
    render_func = env.render

    # We need to use the same statistics for normalization as used in training
    actor_critic, ob_rms = torch.load(os.path.join(args.policy_dir, policy_name))

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    recurrent_hidden_states = torch.zeros(args.batch_size,
                                          actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(args.batch_size, 1)

    obs = env.reset()

    if args.env_name.find('Bullet') > -1:
        import pybullet as p

        torsoId = -1
        for i in range(p.getNumBodies()):
            if (p.getBodyInfo(i)[0].decode() == "torso"):
                torsoId = i

    if render_func is  None:
        print('ERROR: No rendering possible')
        return 1

    util.create_dir(args.save_dir)

    rgb = render_func('rgb_array')  # initial img

    for bidx in range(args.num_batches):

        print('Generating batch {}/{}'.format(bidx+1,args.num_batches))

        rgbs = []

        for step in tqdm(range(args.max_steps)):

            rgbs.append(rgb)

            with torch.no_grad():
                value, action, _, recurrent_hidden_states = actor_critic.act(
                    obs, recurrent_hidden_states, masks, deterministic=args.det)

            # Obser reward and next obs
            obs, reward, done, _ = env.step(action)
            masks.copy_(torch.from_numpy(~done).unsqueeze(1))

            if args.env_name.find('Bullet') > -1:
                if torsoId > -1:
                    distance = 5
                    yaw = 0
                    humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
                    p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

            rgb = render_func('rgb_array')

        save_path = os.path.join(args.save_dir, '{}_{}x{}_batch{}.h5'.format(args.env_name,
                                                                             args.batch_size, args.max_steps, bidx))
        print('Saving batch {}/{} to {}'.format(bidx+1, args.num_batches, save_path))

        with hf.File(save_path, 'w') as f:
            f.create_dataset('images', data=np.array(rgbs).T)


    return 0



if __name__ == '__main__':
    sys.exit(main())




