import numpy as np
from av_goal_recognition.decision_tree import Node, ThresholdDecision, BinaryDecision

scenario_trees = {'heckstrasse':
                 {0: Node(0.5, ThresholdDecision(3 / 8 * np.pi, 'angle_to_goal',
                          Node(0.5, ThresholdDecision(2, 'speed',
                               Node(0.5, ThresholdDecision(np.pi/18, 'angle_in_lane',
                                    Node(0.7),
                                    Node(0.4, ThresholdDecision(-np.pi/18, 'angle_in_lane',
                                         Node(0.5),
                                         Node(0.3)
                                         ))
                                    )),
                               Node(0.4, ThresholdDecision(30, 'oncoming_vehicle_dist',
                                    Node(0.3),
                                    Node(0.7)
                                    ))
                               )),
                          Node(0.5, ThresholdDecision(10, 'speed',
                               Node(0.8),
                               Node(0.3, ThresholdDecision(10, 'vehicle_in_front_speed',
                                    Node(0.8),
                                    Node(0.5, ThresholdDecision(30, 'vehicle_in_front_dist',
                                         Node(0.8), Node(0.2)))
                                    ))
                               ))
                          )),

                  1: Node(0.5, ThresholdDecision(-3 / 5 * np.pi, 'angle_to_goal',
                          Node(0.5, ThresholdDecision(-np.pi / 4, 'angle_to_goal',
                               Node(0.5, BinaryDecision('in_correct_lane',
                                    Node(0.8),
                                    Node(0.2)
                                    )),
                               Node(0.5, ThresholdDecision(2, 'speed',
                                    Node(0.5, ThresholdDecision(np.pi/18, 'angle_in_lane',
                                         Node(0.7),
                                         Node(0.4, ThresholdDecision(-np.pi/18, 'angle_in_lane',
                                              Node(0.5),
                                              Node(0.3)
                                              ))
                                         )),
                                    Node(0.4, ThresholdDecision(30, 'oncoming_vehicle_dist',
                                         Node(0.3),
                                         Node(0.7)
                                         ))
                                    ))
                               )),
                          Node(0.5, ThresholdDecision(10, 'speed',
                               Node(0.2),
                               Node(0.8)
                               ))
                          )),

                  2: Node(0.5, ThresholdDecision(0, 'angle_to_goal',
                          Node(0.5, ThresholdDecision(10, 'speed',
                               Node(0.2),
                               Node(0.8, ThresholdDecision(10, 'vehicle_in_front_speed',
                                    Node(0.8),
                                    Node(0.5, ThresholdDecision(30, 'vehicle_in_front_dist',
                                         Node(0.8),
                                         Node(0.5)
                                         ))
                                    ))
                               )),
                          Node(0.5, BinaryDecision('in_correct_lane',
                               Node(0.9),
                               Node(0.1)
                               ))
                          ))
                 }
                 }
