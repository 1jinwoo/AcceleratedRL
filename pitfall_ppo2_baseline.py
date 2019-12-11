"""
Train an agent using Proximal Policy Optimization from OpenAI Baselines
"""
import retro
import os, inspect
import pickle

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
from baselines.common.vec_env import SubprocVecEnv
from baselines.common.retro_wrappers import make_retro, wrap_deepmind_retro
from train_ppo import ppo2

game = 'Pitfall-Atari2600'
state = retro.State.DEFAULT
scenario = 'scenario'
record = False
verbose = 1
quiet = 0
obs_type = 'image'
players = 1
dir_note = '_rl_baseline'


def main():
    def make_env():
        obs_type = retro.Observations.IMAGE  # retro.Observations.RAM
        env = retro.make(game=game, state=state, scenario=scenario, record=record, players=players, obs_type=obs_type)
        # env = retro.make(game=game, state=state, scenario=scenario)
        env = wrap_deepmind_retro(env)
        return env

    base_dirname = os.path.join(currentdir, "results")

    if not os.path.exists(base_dirname):
        os.mkdir(base_dirname)
    dir_name = "pitfall_ppo2_"
    dir_name = os.path.join(base_dirname, dir_name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    venv = SubprocVecEnv([make_env] * 8)
    performance = ppo2.learn(
        network='cnn',
        env=venv,
        total_timesteps=int(1e5),
        nsteps=32,
        nminibatches=4,
        lam=0.95,
        gamma=0.99,
        noptepochs=16,
        log_interval=10,
        save_interval=5,
        ent_coef=.02,
        lr=lambda f: f * 3e-4,
        cliprange=0.2,
        base_path=dir_name
    )

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
