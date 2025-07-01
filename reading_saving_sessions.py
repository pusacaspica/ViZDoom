import os
import numpy as np
import itertools as it
import vizdoom as vzd 
from examples.python.learning_pytorch import preprocess, DuelQNet, DQNAgent
from time import sleep
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.image as img

class Tick:

    tick_count = -1
    position = (0, 0, 0)
    roll_pitch_angle = (0, 0, 0)
    health_armor = (0, 0)
    weapon_ammo = (0, 0)
    action = ['NOTHING']

    def __init__(self, tick_count, position_x, position_y, position_z, roll, pitch, angle, health, weapon, armor, ammo, action):
        self.tick_count = tick_count
        self.position = (position_x, position_y, position_z)
        self.roll_pitch_angle = (roll, pitch, angle)
        self.health_armor = (health, armor)
        self.weapon_ammo = (weapon, ammo)
        self.action = action

    def __str__(self):
        return str(self.tick_count) + ": HEALTH " + str(self.health_armor[0]) + ", ARMOR: " + str(self.health_armor[1]) + "; WEAPON " + str(self.weapon_ammo[0]) + " w/ AMMO " + str(self.weapon_ammo[1]) + "; ACTION " + str(self.action) + "; POSITION " + str(self.position) + ", ROTATION " + str(self.roll_pitch_angle)

# Misc parameters for running game
episodes_to_watch = 1
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
# Out File Names
# Play sessions should be nominated as playsession_e(number of episode)m(number of map)_test(number of test)_.png
out_file_prefix = "playsession_e"
out_file_extension = "_.txt"

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

game.add_available_game_variable(vzd.GameVariable.POSITION_X)
game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
game.add_available_game_variable(vzd.GameVariable.POSITION_Z)
game.add_available_game_variable(vzd.GameVariable.ANGLE)
game.add_available_game_variable(vzd.GameVariable.PITCH)
game.add_available_game_variable(vzd.GameVariable.ROLL)

game.init()

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
sessions = []

model = torch.load(model_savefile)

rpt = 0

bkgfig = img.imread("E:\\trabalhos\\Doutorado\\GAMEANALYTICS\\Doom_II_Map01-2683275591.png")

for idx in range(episodes_to_watch):
    game.new_episode(recording_file_path = f"episode_{idx}_{rpt}")
    while not game.is_episode_finished():
        session_ticks = []
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
        this_tick = Tick(tick, 
                        game.get_game_variable(vzd.GameVariable.POSITION_X),game.get_game_variable(vzd.GameVariable.POSITION_Y),game.get_game_variable(vzd.GameVariable.POSITION_Z),
                        game.get_game_variable(vzd.GameVariable.ROLL),game.get_game_variable(vzd.GameVariable.PITCH),game.get_game_variable(vzd.GameVariable.ANGLE),
                        game.get_game_variable(vzd.GameVariable.HEALTH),game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON),game.get_game_variable(vzd.GameVariable.ARMOR),game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO),
                        taken_action)
        session_ticks.append(this_tick)
        
        for _ in range(frame_repeat):
            game.advance_action()
            # sleep(0.03)
        if tick + frame_repeat >= 120:
            sessions.append(this_tick)
            out_file_name = out_file_prefix + str(idx) + "_" + str(rpt) + out_file_extension
            if os.path.exists(out_file_name):
                os.remove(out_file_name)
            with open(out_file_name, 'x', ) as file:
                for tick in session_ticks:
                    file.write(str(tick))

out_file_prefix = "playsession_e"
out_file_extension = "_.png"
out_file = out_file_prefix + str(idx) + out_file_extension
fig, ax = plt.subplots()
ax.imshow(bkgfig, extent = [0, len(bkgfig), 0, len(bkgfig[0])])
for idx, tick in enumerate(sessions):
    # for tick in session:
    ax.arrow(
        x = tick.position[0], y =  tick.position[1],
        dx = tick.position[0] * np.cos(tick.roll_pitch_angle[2]), dy = tick.position[1] * np.sin(tick.roll_pitch_angle[2]),      # Direction and length
    )
# Hide axis ticks if desired
ax.set_xticks([])
ax.set_yticks([])
# Save or display
plt.savefig('out_file', bbox_inches='tight', dpi=300)
plt.show()