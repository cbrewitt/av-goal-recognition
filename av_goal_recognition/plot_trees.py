import pydot

from av_goal_recognition.handcrafted_trees import scenario_trees
from av_goal_recognition.base import get_img_dir


def build_pydot_tree(graph, root, idx='R'):
    node = pydot.Node(idx, label=str(root))
    graph.add_node(node)
    if root.decision is not None:
        true_child = build_pydot_tree(graph, root.decision.true_child, idx + 'T')
        false_child = build_pydot_tree(graph, root.decision.false_child, idx + 'F')
        graph.add_edge(pydot.Edge(node, true_child, label='T'))
        graph.add_edge(pydot.Edge(node, false_child, label='F'))
    return node


scenario_name = 'heckstrasse'

for goal_idx, root in scenario_trees[scenario_name].items():
    graph = pydot.Dot(graph_type='digraph')
    build_pydot_tree(graph, root)
    graph.write_png(get_img_dir() + 'handcrafted_tree_{}_G{}.png'.format(scenario_name, goal_idx))
