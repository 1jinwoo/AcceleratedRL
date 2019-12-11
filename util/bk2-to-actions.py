"""
Converts Pitfall's bk2 formatted retro.Movie objects into custom python list of player actions
Based on https://github.com/AurelianTactics/sonic_1_utilities/blob/master/retro_movie_transitions.py?fbclid=IwAR1hWkB7qiRMjohXqBZZ-RfeoXTnSY_ITBAADCC6MP8SajeD5mYXo8nKuNg
"""

import sys
import retro
import pickle

output_name = sys.argv[2]

movie_path = sys.argv[1]
output_path = f'../actions/{output_name}'

print(f'Loading movie from {movie_path}')

movie = retro.Movie(movie_path)
movie.step()

env = retro.make(
    game=movie.get_game(),
    state=None,
    # bk2s can contain any button presses, so allow everything
    use_restricted_actions=retro.Actions.ALL,
    players=movie.players,
)
env.initial_state = movie.get_state()
env.reset()

print('Reading actions from the .bk2')

keys_list = []
while movie.step():
    keys = []
    for p in range(movie.players):
        for i in range(env.num_buttons):
            keys.append(movie.get_key(i, p))
    keys_list.append(keys)
    env.step(keys)
#     env.render()  # Comment in only to see the game in action
    env.reset()
env.close()

print(f'Pickling a list of lenth {len(keys_list)}')
print(f'Saving to {output_path}')
pickle.dump(keys_list, open(output_path, 'wb'))