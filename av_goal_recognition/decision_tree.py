

class Node:
    def __init__(self, value, decision=None):
        self.value = value
        self.decision = decision

    def traverse(self, features):
        current_node = self
        while current_node.decision is not None:
            current_node = current_node.decision.select_child(features)
        return current_node.value


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


class ThresholdDecision(Decision):

    def __init__(self, threshold, *args):
        super().__init__(*args)
        self.threshold = threshold

    def rule(self, features):
        return features[self.feature_name] > self.threshold


tree = Node(0.5)
tree.decision = ThresholdDecision(0, 'angle_to_goal', Node(0.5), Node(0.5))
tree.decision.true_child.decision = BinaryDecision('in_correct_lane', Node(0.9), Node(0.1))
tree.decision.false_child.decision = ThresholdDecision(30, 'path_to_goal_length', Node(0.5), Node(0.5))
tree.decision.false_child.decision.false_child.decision = ThresholdDecision(10, 'speed', Node(0.2), Node(0.8))

print(tree.traverse({'angle_to_goal': -1, 'in_correct_lane': True, 'path_to_goal_length': 20, 'speed': 15}))