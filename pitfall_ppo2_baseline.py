"""
Train an agent using Proximal Policy Optimization from OpenAI Baselines
"""
import retro
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


def main():
    def make_env():
        obs_type = retro.Observations.IMAGE  # retro.Observations.RAM
        env = retro.make(game=game, state=state, scenario=scenario, record=record, players=players, obs_type=obs_type)
        # env = retro.make(game=game, state=state, scenario=scenario)
        env = wrap_deepmind_retro(env)
        return env

    venv = SubprocVecEnv([make_env] * 8)
    ppo2.learn(
        network='cnn',
        env=venv,
        total_timesteps=int(1e6),
        nsteps=128,
        nminibatches=4,
        lam=0.95,
        gamma=0.99,
        noptepochs=4,
        log_interval=1,
        ent_coef=.01,
        lr=lambda f: f * 2.5e-4,
        cliprange=0.1,
    )


if __name__ == '__main__':
    main()