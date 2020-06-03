import numpy as np
from av_goal_recognition.decision_tree import Node, ThresholdDecision, BinaryDecision


"""
al features
path_to_goal_length
in_correct_lane
speed
acceleration
angle_in_lane
angle_to_goal
"""

scenario_trees = {'heckstrasse':
                 {0: Node(0.5, ThresholdDecision(3 / 8 * np.pi, 'angle_to_goal',
                          Node(0.5, ThresholdDecision(34.5, 'path_to_goal_length',
                               Node(0.5),
                               Node(0.5, ThresholdDecision(np.pi/18, 'angle_in_lane',
                                    Node(0.7),
                                    Node(0.4, ThresholdDecision(-np.pi/18, 'angle_in_lane',
                                         Node(0.5),
                                         Node(0.3)
                                         ))
                                    ))
                               )),
                          Node(0.5, ThresholdDecision(10, 'speed',
                               Node(0.2),
                               Node(0.8)
                               ))
                          )),

                  1: Node(0.5, ThresholdDecision(-3 / 5 * np.pi, 'angle_to_goal',
                          Node(0.5, ThresholdDecision(-np.pi / 4, 'angle_to_goal',
                               Node(0.5, BinaryDecision('in_correct_lane',
                                    Node(0.8),
                                    Node(0.2)
                                    )),
                               Node(0.5, ThresholdDecision(56, 'path_to_goal_length',
                                    Node(0.5),
                                    Node(0.5, ThresholdDecision(np.pi / 18, 'angle_in_lane',
                                         Node(0.7),
                                         Node(0.4, ThresholdDecision(-np.pi / 18, 'angle_in_lane',
                                              Node(0.5),
                                              Node(0.3)
                                              ))
                                         ))
                                    ))
                               )),
                          Node(0.5))),

                  2: Node(0.5, ThresholdDecision(0, 'angle_to_goal',
                          Node(0.5, ThresholdDecision(30, 'path_to_goal_length',
                               Node(0.5),
                               Node(0.5, ThresholdDecision(10, 'speed',
                                    Node(0.2),
                                    Node(0.8)
                                    ))
                               )),
                          Node(0.5, BinaryDecision('in_correct_lane',
                                           Node(0.9),
                                           Node(0.1)
                               ))
                          ))
                 }
                 }
