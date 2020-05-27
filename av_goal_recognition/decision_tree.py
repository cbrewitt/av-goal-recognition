

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
