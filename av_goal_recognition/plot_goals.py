import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import lanelet2

import imageio
import map_vis_lanelet2
from tracks_import import read_from_csv
from goal_recognition import ScenarioConfig, GoalDetector


parser = argparse.ArgumentParser(description='create a plot of lanelets and goals')
parser.add_argument('--dataset', type=str, default='ind')
parser.add_argument('--map', type=str, default='heckstrasse')
args = parser.parse_args()

# create a figure
fig, axes = plt.subplots(1, 1, figsize=(12, 12))

# load and draw the lanelet2 map
lat_origin = 50.779081
lon_origin = 6.164788
print("Loading map...")
lanelet_map_file = '../inD_LaneletMaps/heckstrasse.osm'
projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(lat_origin, lon_origin))
laneletmap, _ = lanelet2.io.loadRobust(lanelet_map_file, projector)
print('loaded map')
map_vis_lanelet2.draw_lanelet_map(laneletmap, axes)

# import tracks and draw
recording_number = '30'
tracks, static_info, meta_info = read_from_csv(
    '../inD-dataset/data/{}_tracks.csv'.format(recording_number),
    '../inD-dataset/data/{}_tracksMeta.csv'.format(recording_number),
    '../inD-dataset/data/{}_recordingMeta.csv'.format(recording_number))

# for idx, track in enumerate(tracks):
#     if static_info[idx]['class'] == 'car':
#         plt.plot(track['xCenter'], track['yCenter'], zorder=11)
#         #break
track_idx = 0
track = tracks[track_idx]
plt.plot(track['xCenter'], track['yCenter'], 'yo', zorder=11, markersize=2)

# get the lanelet of the vehicles initial position

initial_pos = (tracks[0]['xCenter'][0], tracks[0]['yCenter'][0])

background = imageio.imread('../inD-dataset/data/{}_background.png'.format(recording_number))
rescale_factor = meta_info['orthoPxToMeter'] * 12

target_shape = (int(background.shape[1] * rescale_factor),
                int(background.shape[0] * rescale_factor))

extent = (0, int(background.shape[1] * rescale_factor),
          -int(background.shape[0] * rescale_factor), 0)

background_resized = cv2.resize(background, target_shape)


goal_locations = [(17.40, -4.97),
                  (75.18, -56.65),


                  (62.47, -17.54)]

goal_dist_threshold = 1.5
goal = None
for i in range(static_info[track_idx]['numFrames']):
    point = np.array([track['xCenter'][i], track['yCenter'][i]])
    for goal_idx, loc in enumerate(goal_locations):
        dist = np.linalg.norm(point - loc)
        if dist < goal_dist_threshold:
            goal = goal_idx
            break


print('goal is G{}'.format(goal))


plt.plot(*zip(*goal_locations), 'ro', zorder=12, markersize=20)
ax = plt.gca()

for i in range(len(goal_locations)):
    label = 'G{}'.format(i)
    ax.annotate(label, goal_locations[i], zorder=12, color='white')

plt.imshow(background, extent=extent)

# plot goal priors
map_meta = ScenarioConfig.load('scenario_config/heckstrasse.json')
goal_detector = GoalDetector()
agent_goals = goal_detector.get_agents_goals_ind(tracks, static_info, meta_info, map_meta)

num_agents_for_goal = {}
total_agents_with_goals = 0
for goal_idx, goal in enumerate(map_meta.goals):
    num_agents = sum([(goal in ag) for ag in agent_goals.values()])
    num_agents_for_goal[goal_idx] = num_agents
    total_agents_with_goals += num_agents

total_agents = len(agent_goals)

height = np.array(list(num_agents_for_goal.values()) + [total_agents - total_agents_with_goals])
height = height / total_agents
labels = ['G0', 'G1', 'G2', 'Unknown']
plt.figure()
x = range(4)
plt.bar(x, height, tick_label=labels)
plt.ylabel('Fraction of total agents')

plt.show()
