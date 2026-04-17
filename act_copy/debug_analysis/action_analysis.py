import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import ast
import os
import argparse
from matplotlib.lines import Line2D

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Analyze Action Chunks Side-by-Side")
parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing CSVs")
args = parser.parse_args()

# --- CONFIGURATION ---
WINDOW_PAST = 50
CHUNKS_TO_SHOW = 32
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue (X), Orange (Y), Green (Z)
GRIP_COLOR = 'black'
PLAN_STYLE = '--'

# 1. Setup Data Discovery
target_folder = args.folder.strip()

def get_all_csvs(path):
    if not os.path.exists(path):
        return []
    # Get all CSVs in the directory, sorted alphabetically
    return sorted([f for f in os.listdir(path) if f.endswith('.csv')])

all_csv_files = get_all_csvs(target_folder)

if not all_csv_files:
    print(f"Error: No CSV files found in {target_folder}")
    exit()

def load_data(folder, filename):
    if not filename or filename == "No Files":
        return None
    path = os.path.join(folder, filename)
    df = pd.read_csv(path)
    df['action_chunk'] = df['action_chunk'].apply(ast.literal_eval).apply(np.array)
    df['final_action'] = df['final_action'].apply(ast.literal_eval).apply(np.array)
    return {
        'df': df, 
        'history': np.stack(df['final_action'].values),
        'name': filename, 
        'len': len(df), 
        'chunk_size': df['action_chunk'][0].shape[0]
    }

# Initial State: Load the first file for both sides by default
state = {
    "left": load_data(target_folder, all_csv_files[0]),
    "right": load_data(target_folder, all_csv_files[0])
}

# 2. Plotting Setup
fig, axs = plt.subplots(2, 2, figsize=(16, 9), sharex='col')
plt.subplots_adjust(bottom=0.15, top=0.88, hspace=0.2, wspace=0.15)
(ax_pos_l, ax_pos_r), (ax_grip_l, ax_grip_r) = axs

def draw_side(ax_pos, ax_grip, data, t):
    if data is None: 
        ax_pos.text(0.5, 0.5, "No Data Loaded", ha='center')
        return
    
    t = min(int(t), data['len'] - 1)
    df, history, chunk_size = data['df'], data['history'], data['chunk_size']
    ax_pos.clear(); ax_grip.clear()
    
    # Bolt Effect (Past Predictions)
    start_fan = max(0, t - CHUNKS_TO_SHOW)
    for past_t in range(start_fan, t + 1):
        past_chunk = df['action_chunk'][past_t]
        fut_rng = np.arange(past_t, past_t + chunk_size)
        alpha = 0.05 + (0.35 * (past_t - start_fan) / CHUNKS_TO_SHOW)
        for i in range(3):
            ax_pos.plot(fut_rng, past_chunk[:, i], color=COLORS[i], lw=0.7, alpha=alpha)
        ax_grip.plot(fut_rng, past_chunk[:, -1], color='gray', lw=0.7, alpha=alpha)

    # Executed Reality & Current Plan
    h_rng = np.arange(max(0, t - WINDOW_PAST), t + 1)
    executed = history[max(0, t - WINDOW_PAST) : t + 1]
    curr_plan = df['action_chunk'][t]
    
    for i in range(3):
        ax_pos.plot(h_rng, executed[:, i], color=COLORS[i], lw=3, label=f'Action {["X", "Y", "Z"][i]}' if t == 0 else "")
        ax_pos.plot(np.arange(t, t + chunk_size), curr_plan[:, i], color=COLORS[i], ls=PLAN_STYLE, lw=1.5)
        ax_pos.scatter(t, executed[-1, i], color=COLORS[i], s=70, edgecolors='white', zorder=5)

    # Plot plug position (if available in the data)
    if 'plug_x' in df.columns and 'plug_y' in df.columns and 'plug_z' in df.columns:
        plug_x = df['plug_x'].values[h_rng]
        plug_y = df['plug_y'].values[h_rng]
        plug_z = df['plug_z'].values[h_rng]
        
        for i, (plug_data, color) in enumerate([(plug_x, COLORS[0]), (plug_y, COLORS[1]), (plug_z, COLORS[2])]):
            ax_pos.plot(h_rng, plug_data, color=color, lw=2, alpha=0.5, linestyle=':', label=f'Plug {["X", "Y", "Z"][i]}')

    ax_grip.plot(h_rng, executed[:, -1], color=GRIP_COLOR, lw=2)
    ax_grip.scatter(t, executed[-1, -1], color='red', marker='X', s=100, zorder=5)
    
    # Title coloring based on filename
    title_color = 'green' if 'success' in data['name'].lower() else 'red'
    ax_pos.set_title(data['name'], fontsize=10, fontweight='bold', color=title_color)
    ax_pos.set_xlim(t - WINDOW_PAST, t + chunk_size + 5)
    ax_grip.set_ylim([-1.1, 1.1])
    ax_pos.grid(True, alpha=0.1); ax_grip.grid(True, alpha=0.1)

def update(val=None):
    t = slider.val
    draw_side(ax_pos_l, ax_grip_l, state["left"], t)
    draw_side(ax_pos_r, ax_grip_r, state["right"], t)
    fig.canvas.draw_idle()

# 3. LEGEND
legend_elements = [
    Line2D([0], [0], color=COLORS[0], lw=3, label='Action X'),
    Line2D([0], [0], color=COLORS[1], lw=3, label='Action Y'),
    Line2D([0], [0], color=COLORS[2], lw=3, label='Action Z'),
    Line2D([0], [0], color=COLORS[0], lw=2, alpha=0.5, linestyle=':', label='Plug X'),
    Line2D([0], [0], color=COLORS[1], lw=2, alpha=0.5, linestyle=':', label='Plug Y'),
    Line2D([0], [0], color=COLORS[2], lw=2, alpha=0.5, linestyle=':', label='Plug Z'),
    Line2D([0], [0], color=GRIP_COLOR, lw=2, label='Gripper'),
    Line2D([0], [0], color='black', ls=PLAN_STYLE, label='Current Plan'),
    Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markersize=10, label='Executed Step')
]
fig.legend(handles=legend_elements, loc='upper center', ncol=9, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.96))

# 4. Interactive Widgets
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
max_l = state['left']['len'] if state['left'] else 0
max_r = state['right']['len'] if state['right'] else 0
slider = Slider(ax_slider, 'Step', 0, max(max_l, max_r, 1)-1, valinit=0, valfmt='%d')
slider.on_changed(update)

# --- Menu Logic ---
ax_menu_l = plt.axes([0.05, 0.45, 0.22, 0.4], visible=False, facecolor='#f8f9fa')
ax_menu_r = plt.axes([0.73, 0.45, 0.22, 0.4], visible=False, facecolor='#f8f9fa')

# Both radios now get the full file list
radio_l = RadioButtons(ax_menu_l, all_csv_files)
radio_r = RadioButtons(ax_menu_r, all_csv_files)

def set_radio_visible(ax, visible):
    ax.set_visible(visible)
    for child in ax.get_children():
        child.set_visible(visible)

def toggle_menu(side):
    target_ax, other_ax = (ax_menu_l, ax_menu_r) if side == "left" else (ax_menu_r, ax_menu_l)
    set_radio_visible(other_ax, False)
    new_state = not target_ax.get_visible()
    set_radio_visible(target_ax, new_state)
    fig.canvas.draw_idle()

def select_l(label):
    state["left"] = load_data(target_folder, label)
    set_radio_visible(ax_menu_l, False)
    update()

def select_r(label):
    state["right"] = load_data(target_folder, label)
    set_radio_visible(ax_menu_r, False)
    update()

radio_l.on_clicked(select_l)
radio_r.on_clicked(select_r)

# Buttons (renamed for general use)
ax_btn_l = plt.axes([0.05, 0.9, 0.12, 0.04])
btn_l = Button(ax_btn_l, 'Select Left', color='#e3f2fd')
btn_l.on_clicked(lambda x: toggle_menu("left"))

ax_btn_r = plt.axes([0.83, 0.9, 0.12, 0.04])
btn_r = Button(ax_btn_r, 'Select Right', color='#f5f5f5')
btn_r.on_clicked(lambda x: toggle_menu("right"))

# Startup setup
set_radio_visible(ax_menu_l, False)
set_radio_visible(ax_menu_r, False)
update()
plt.show()