import retro

game = 'Pitfall-Atari2600'
scenario = 'scenario'
record = '.'
verbose = 1
quiet = 0
obs_type = 'image'
players = 1

'''
parser.add_argument('--game', default=GAME, help='the name or path for the game to run')
parser.add_argument('--state', help='the initial state file to load, minus thes extension')
parser.add_argument('--scenario', '-s', default='scenario', help='the scenario file to load, minus the extension')
parser.add_argument('--record', '-r', action='store_true', help='record bk2 movies')
parser.add_argument('--verbose', '-v', action='count', default=1, help='increase verbosity (can be specified multiple times)')
parser.add_argument('--quiet', '-q', action='count', default=0, help='decrease verbosity (can be specified multiple times)')
parser.add_argument('--obs-type', '-o', default='image', choices=['image', 'ram'], help='the observation type, either `image` (default) or `ram`')
parser.add_argument('--players', '-p', type=int, default=1, help='number of players/agents (default: 1)')
args = parser.parse_args()
'''

obs_type = retro.Observations.IMAGE if obs_type == 'image' else retro.Observations.RAM
env = retro.make(game, retro.State.DEFAULT, scenario=scenario, record=record, players=players, obs_type=obs_type)
verbosity = verbose - quiet
try:
    while True:
        ob = env.reset()
        t = 0
        totrew = [0] * players
        while True:
            ac = env.action_space.sample()
            ob, rew, done, info = env.step(ac)
            t += 1
            if t % 10 == 0:
                if verbosity > 1:
                    infostr = ''
                    if info:
                        infostr = ', info: ' + ', '.join(['%s=%i' % (k, v) for k, v in info.items()])
                    print(('t=%i' % t) + infostr)
                env.render()
            if players == 1:
                rew = [rew]
            for i, r in enumerate(rew):
                totrew[i] += r
                if verbosity > 0:
                    if r > 0:
                        print('t=%i p=%i got reward: %g, current reward: %g' % (t, i, r, totrew[i]))
                    if r < 0:
                        print('t=%i p=%i got penalty: %g, current reward: %g' % (t, i, r, totrew[i]))
            if done:
                env.render()
                try:
                    if verbosity >= 0:
                        if players > 1:
                            print("done! total reward: time=%i, reward=%r" % (t, totrew))
                        else:
                            print("done! total reward: time=%i, reward=%d" % (t, totrew[0]))
                        input("press enter to continue")
                        print()
                    else:
                        input("")
                except EOFError:
                    exit(0)
                break
except KeyboardInterrupt:
    exit(0)