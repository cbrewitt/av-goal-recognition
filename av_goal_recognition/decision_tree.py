import pickle

import pydot
from sklearn.tree import _tree


class Node:
    def __init__(self, value, decision=None):
        self.value = value
        self.decision = decision

    def traverse(self, features):
        current_node = self
        while current_node.decision is not None:
            current_node = current_node.decision.select_child(features)
        return current_node.value

    def __str__(self):
        text = ''
        text += '{0:.3f}\n'.format(self.value)
        if self.decision is not None:
            text += str(self.decision)
        return text

    @classmethod
    def from_sklearn(cls, input_tree, feature_types):
        # based on:
        # https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree

        tree_ = input_tree.tree_
        feature_names = [*feature_types]
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        def recurse(node):
            value = tree_.value[node][0][1] / tree_.value[node].sum()
            out_node = Node(value)
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                true_child = recurse(tree_.children_right[node])
                false_child = recurse(tree_.children_left[node])
                if feature_types[name] == 'scalar':
                    out_node.decision = ThresholdDecision(threshold, name, true_child, false_child)
                elif feature_types[name] == 'binary':
                    out_node.decision = BinaryDecision(name, true_child, false_child)
                else:
                    raise ValueError('invalid feature type')
            return out_node

        return recurse(0)

    def pydot_tree(self):
        graph = pydot.Dot(graph_type='digraph')

        def recurse(graph, root, idx='R'):
            node = pydot.Node(idx, label=str(root))
            graph.add_node(node)
            if root.decision is not None:
                true_child = recurse(graph, root.decision.true_child, idx + 'T')
                false_child = recurse(graph, root.decision.false_child, idx + 'F')
                true_weight = root.decision.true_child.value / root.value
                false_weight = root.decision.false_child.value / root.value
                graph.add_edge(pydot.Edge(node, true_child, label='T: {:.2f}'.format(true_weight)))
                graph.add_edge(pydot.Edge(node, false_child, label='F: {:.2f}'.format(false_weight)))
            return node

        recurse(graph, self)
        return graph

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


class Decision:

    def __init__(self, feature_name, true_child, false_child):
        self.feature_name = feature_name
        self.true_child = true_child
        self.false_child = false_child

    def rule(self, features):
        raise NotImplementedError

    def select_child(self, features):
        if self.rule(features):
            return self.true_child
        else:
            return self.false_child


class BinaryDecision(Decision):

    def rule(self, features):
        return features[self.feature_name]

    def __str__(self):
        return self.feature_name + '\n'


class ThresholdDecision(Decision):

    def __init__(self, threshold, *args):
        super().__init__(*args)
        self.threshold = threshold

    def rule(self, features):
        return features[self.feature_name] > self.threshold

    def __str__(self):
        return '{} > {:.2f}\n'.format(self.feature_name, self.threshold)

