import os
import numpy as np
import itertools as it
import vizdoom as vzd 
from examples.python.learning_pytorch import preprocess, DuelQNet, DQNAgent
from time import sleep
import torch
from torch.autograd import Variable

# Misc parameters for running game
episodes_to_watch = 10
frame_repeat = 12
resolution = (30, 45)

# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
train_epochs = 1000
learning_steps_per_epoch = 2000
replay_memory_size = 10000
# NN learning settings
batch_size = 64
model_savefile = "./model-doom2.pth"

def get_q_values(state):
    state = torch.from_numpy(state).to('cuda:0')
    state = Variable(state)
    return model(state)

def get_best_action(state):
    q = get_q_values(state)
    m, index = torch.max(q, 1)
    action = index.data.cpu().numpy()[0]
    return action

game = vzd.DoomGame()

# Use other config file if you wish.
game.load_config(os.path.join(vzd.scenarios_path, "basic_doom2.cfg"))
game.set_episode_timeout(4200)

# Record episodes while playing in 320x240 resolution without HUD
game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
game.set_render_hud(False)

# Episodes can be recorder in any available mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR)
game.set_mode(vzd.Mode.ASYNC_PLAYER)

game.init()

game.add_available_game_variable(vzd.GameVariable.POSITION_X)
game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
game.add_available_game_variable(vzd.GameVariable.POSITION_Z)
game.add_available_game_variable(vzd.GameVariable.ANGLE)
game.add_available_game_variable(vzd.GameVariable.PITCH)
game.add_available_game_variable(vzd.GameVariable.ROLL)

actions_names = [	
    	"ATTACK",
        "USE",
        "MOVE_RIGHT",
        "MOVE_LEFT",
        "MOVE_BACKWARD",
        "MOVE_FORWARD",
        "TURN_RIGHT",
        "TURN_LEFT",
        "SELECT_NEXT_WEAPON",
        "SELECT_PREV_WEAPON"
        ]

n = game.get_available_buttons_size()
actions = [list(a) for a in it.product([0, 1], repeat=n)]

model = torch.load(model_savefile)

for idx in range(episodes_to_watch):
    if idx == episodes_to_watch - 1:
        game.set_window_visible(True)
    game.new_episode()
    while not game.is_episode_finished():
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            state = state.reshape([1, 1, resolution[0], resolution[1]])
            best_action_index = get_best_action(state)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])

            action_indexes = [index for index, value in enumerate(actions[best_action_index]) if value == 1]
            taken_action = []
            for index in action_indexes:
                taken_action.extend([actions_names[index]])
            game_variables = [variable for variable in game.get_available_game_variables()]
            tick = game.get_state().tic
            print(tick)
            print(game_variables)
            print(taken_action)
            print(game.get_last_reward())
            
            for _ in range(frame_repeat):
                game.advance_action()
                print(game.get_last_reward())
                # sleep(0.03)
        print(game.get_total_reward())