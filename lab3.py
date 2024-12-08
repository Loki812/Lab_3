from dataclasses import dataclass
from typing import List
from collections import Counter
import math
import sys
import pickle


# for this project each label will either be nl or en.
# each value will be a 15 word sentance in dutch or english
@dataclass
class Example:
    label: str
    value: str

class Node:
    def __init__(self, feature=None, left=None, right=None, value=None) -> None:
       self.feature = feature
       self.left = left
       self.right = right
       self.value = value


class  DecisionTree:

    def __init__(self, features, max_depth=None) -> None:
        self.max_depth = max_depth
        self.features = features
        self.root = None

    def entropy(self, labels: List[str], weights: List[float] = None) -> float:
        """
        Returns the given entropy for any split of data
        """
        if weights is None:
            weights = [1] * len(labels)

        total_weight = sum(weights)
        weighted_counts = Counter()

        for label, weight in zip(labels, weights):
            weighted_counts[label] += weight

        probs = [w_count / total_weight for w_count in weighted_counts.values()]
        e = 0
        for p in probs:
            if p > 0:
                e -= p * math.log2(p)
        return e
    
    def find_best_split(self, examples: List[Example], features: List[str], weights: List[float]) -> str:
        """
        Finds the best feature to use to reduce the entrophy of a given subset of examples
        """
        best_feature = None
        best_gain = -1
        labels = [e.label for e in examples]
        current_impurity = self.entropy(labels, weights)

        for feature in features:
            # makes the feature a standalone word so substrings do not count "become" as containing "me"
            left_examples = []
            right_examples = []
            left_weights = []
            right_weights = []
            for i, example in enumerate(examples):
                e = example.value.replace("\n", "")
                e = e.split(" ")

                if feature in e:
                    left_examples.append(example)
                    left_weights.append(weights[i])
                else:
                    right_examples.append(example)
                    right_weights.append(weights[i])

            weighted_impurity = (
                ((sum(left_weights) / sum(weights)) * self.entropy([e.label for e in left_examples], left_weights)) +
                ((sum(right_weights) / sum(weights)) * self.entropy([e.label for e in right_examples], right_weights))
            )

            info_gain = current_impurity - weighted_impurity

            if info_gain > best_gain:
                best_feature = feature
                best_gain = info_gain
        
        return best_feature
    

    def build_tree(self, examples: List[Example], features: List[str], weights: List[float], depth: int = 0) -> Node:
        
        if depth >= self.max_depth or len(examples) == 1 or len(features) == 0:
            value = self.find_most_common([e.label for e in examples], weights)
            return Node(value=value) # creates a leaf node what will has a value of the most common training examples
        
        best_feature = self.find_best_split(examples, features, weights)

        left_examples = []
        right_examples = []
        left_weights = []
        right_weights = []
        for i, example in enumerate(examples):
            e = example.value.replace("\n", "")
            e = e.split(" ")

            if best_feature in e:
                left_examples.append(example)
                left_weights.append(weights[i])
            else:
                right_examples.append(example)
                right_weights.append(weights[i])

        if len(left_examples) == 0:
            return Node(value=right_examples[0].label)
        if len(right_examples) == 0:
            return Node(value=left_examples[0].label)

        child_features = [f for f in features if f != best_feature]
        left_child = self.build_tree(left_examples, child_features, left_weights, depth + 1)
        right_child = self.build_tree(right_examples, child_features, right_weights, depth + 1)

        return Node(best_feature, left_child, right_child) 


    def find_most_common(self, labels: List[str], weights: List[float]) -> str:
        """
        Finds the most common label taking into account the weight of each example
        """
        weighted_counts = Counter()
        for label, weight in zip(labels, weights):
            weighted_counts[label] += weight

        return max(weighted_counts, key=weighted_counts.get)


    def fit_tree(self, examples, weights: List[float] = None):
        if weights is None:
            weights = [1] * len(examples)
        self.root = self.build_tree(examples, self.features, weights)

    def traverse(self, x: str, node: Node) -> str:

        if node.value is not None:
            return node.value
        
        split = x.replace("\n", "").split(" ")

        if node.feature in split:
            return self.traverse(x, node.left)
        else:
            return self.traverse(x, node.right)
        
    def predict(self, X: List[str]):
        if self.root is not None:
            return [self.traverse(x, self.root) for x in X]

class Adaboost:

    def __init__(self, features: List[str], n_estimators=20) -> None:
        """
        Initializes the adaboost algorithm, the number of features is the max number of 
        decision stumps we can have
        """
        self.features = features
        self.model_weights = []
        self.stumps: List[DecisionTree] = []
        self.n_estimators = n_estimators
    
    def fit(self, examples: List[Example]):

        base_weight = 1 / len(examples)
        weights = [base_weight] * len(examples)


        for _ in range(self.n_estimators):
            stump = DecisionTree(self.features, max_depth=1)
            stump.fit_tree(examples, weights)
            pred_labels = stump.predict(X=[e.value for e in examples])
            act_labels = [e.label for e in examples]

            # calculating error
            error = 0
            for i in range(0,len(pred_labels)):
               if pred_labels[i] != act_labels[i]:
                   error += weights[i]
            error = error / sum(weights)
            if error == 0:
                error = 1e-10
            elif error >= 1:
                error = 1 - 1e-10

            delta = (1 - error) / (error + 1e-10)
            model_weight = 0.5 * math.log(delta)

            # modify weights for examples
            for i in range(0,len(pred_labels)):
                if pred_labels[i] == act_labels[i]:
                    weights[i] *= math.exp(-model_weight)
                else:
                    weights[i] *= math.exp(model_weight)
            
            # normalize the weights
            new_sum = sum(weights)
            weights = [w / new_sum for w in weights]

            self.stumps.append(stump)
            self.model_weights.append(model_weight)

    def predict(self, X: List[str]) -> List[str]:
        
        ne_prob_predictions = [0] * len(X)
        en_prob_predictions = [0] * len(X)

        for i, model in enumerate(self.stumps):
            pred = model.predict(X)

            for k in range(0,len(X)):
                if pred[k] == 'en':
                    en_prob_predictions[k] += self.model_weights[i]
                else:
                    ne_prob_predictions[k] += self.model_weights[i]
        
        # make final predictions based off of which class has higher 
        # sum of models ("voting")
        final_pred = ["en" if en_prob_predictions[i] > ne_prob_predictions[i] else "nl" for i in range(len(X))]
        
        return final_pred

def build_examples(file_name: str) -> List[Example]:
    examples = []
    with open(file_name, "r") as file:
        line = file.readline()
        while line != "":
            sep = line.split("|")
            examples.append(Example(sep[0], sep[1]))
            line = file.readline()
    return examples
        
def build_features(file_name: str) -> List[str]:
    features = []
    with open(file_name, "r") as file:
        line = file.readline()
        while line != "":
            line = line.replace("\n", "")
            features.append(line)
            line = file.readline()
    return features

def get_accuracy(examples: List[Example], pred: List[str]):
    num_succ = 0
    for i in range(0,len(pred)):
        if examples[i].label == pred[i]:
            num_succ += 1
    
    return num_succ / len(pred)

def main():
    MAX_DEPTH = 13 
    N_ESTIMATORS = 23
    
    features = build_features(sys.argv[3])

    if sys.argv[1] == "train":
        examples = build_examples(sys.argv[2])
        if sys.argv[5] == "dt":
            model = DecisionTree(features, MAX_DEPTH)
            model.fit_tree(examples)
            with open(sys.argv[4], "wb") as file:
                pickle.dump(model, file)
        else:
            model = Adaboost(features, N_ESTIMATORS)
            model.fit(examples)
            with open(sys.argv[4], "wb") as file:
                pickle.dump(model, file)
    elif sys.argv[1] == "predict":
        X = []
        with open(sys.argv[2], "r") as file:
            for line in file.readlines():
                line = line.lower()
                X.append(line)
        with open(sys.argv[4], "rb") as file:
            model = pickle.load(file)
            pred = model.predict(X)
        for p in pred:
            print(p)
        

if __name__ == "__main__":
    main()

