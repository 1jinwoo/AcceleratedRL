"""
output a .bk2 video of a (pre-trained model) agent playing a game
"""

import retro
import os, inspect
import pickle
from baselines.ppo2.model import Model
from baselines.common.vec_env import SubprocVecEnv
from baselines.common.retro_wrappers import make_retro, wrap_deepmind_retro
from baselines.common.policies import build_policy
from train_ppo import ppo2
from train_ppo.runner import Runner
import retro
import matplotlib.pyplot as plt
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

def main():
    def make_env():
        obs_type = retro.Observations.IMAGE  # retro.Observations.RAM
        env = retro.make(game='Pitfall-Atari2600', state=retro.State.DEFAULT, scenario='scenario', record='.', 
                         players=1, obs_type=obs_type)
        env = wrap_deepmind_retro(env)
        return env

    base_dirname = os.path.join(currentdir, "results")
    #dir_name = "pitfall_ppo2_rl_baseline1"
    dir_name = "test"
    dir_name = os.path.join(base_dirname, dir_name)
    load_path = os.path.join(dir_name, 'models/00390')

    venv = SubprocVecEnv([make_env] * 1) #Vectorized
    network = 'cnn'
    policy = build_policy(venv, network)
    nenvs = venv.num_envs  # Get the nb of env

    # Get state_space and action_space
    ob_space = venv.observation_space
    ac_space = venv.action_space

    # Instantiate the model object
    model_fn = Model
    nsteps=2048
    nbatch = nenvs * nsteps
    nminibatches= 4
    nbatch_train = nbatch // nminibatches
    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs,
                     nbatch_train=nbatch_train, nsteps=2048, ent_coef=0.0, vf_coef=0.5,  max_grad_norm=0.5)
    model.load(load_path)
    
    # Instantiate the runner object
    runner = Runner(env=venv, model=model, nsteps=nsteps, gamma=0.99, lam=0.95)

    # run the Runner and record video
    total_timesteps=int(1e4)
    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        print("progress: ", update, "/", nupdates)
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()

if __name__ == '__main__':
    main()