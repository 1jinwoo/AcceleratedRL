import retro
import os, inspect
import pickle

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
from baselines.common.vec_env import SubprocVecEnv
from baselines.common.retro_wrappers import make_retro, wrap_deepmind_retro
from baselines.common.policies import build_policy
from train_ppo import ppo2
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from train_ppo.runner import Runner
import retro

game = 'Pitfall-Atari2600'
state = retro.State.DEFAULT
scenario = 'scenario'
record = True
verbose = 1
quiet = 0
obs_type = 'image'
players = 1
dir_note = '_rl_baseline1'

def main():
    def make_env():
        obs_type = retro.Observations.IMAGE  # retro.Observations.RAM
        env = retro.make(game=game, state=state, scenario=scenario, record=record, players=players, obs_type=obs_type)
        # env = retro.make(game=game, state=state, scenario=scenario)
        env = wrap_deepmind_retro(env)
        return env

    base_dirname = os.path.join(currentdir, "results")

    dir_name = "pitfall_ppo2_rl_baseline1"
    dir_name = os.path.join(base_dirname, dir_name)
    load_path = os.path.join(dir_name, )

    venv = SubprocVecEnv([make_env] * 1)
    network = 'cnn'
    policy = build_policy(venv, network)
    # Get the nb of env
    nenvs = venv.num_envs

    # Get state_space and action_space
    ob_space = venv.observation_space
    ac_space = venv.action_space




    # Instantiate the model object (that creates act_model and train_model)
    from baselines.ppo2.model import Model
    model_fn = Model

    model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs)
    model.load(load_path)
    # Instantiate the runner object
    runner = Runner(env=venv, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    performance_fname = os.path.join(dir_name, "performance.p")
    with open(performance_fname, "wb") as f:
        pickle.dump(performance, f)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(performance["policy_loss"], label="policy_loss")
    plt.plot(performance["value_loss"], label="value_loss")
    plt.legend()
    plt.savefig(os.path.join(dir_name, "loss.jpg"), dpi=300)
    plt.close()
    plt.figure()
    plt.plot(performance["reward"], label="reward")
    plt.legend()
    plt.savefig(os.path.join(dir_name, "reward.jpg"), dpi=300)
    plt.close('all')


if __name__ == '__main__':
    main()