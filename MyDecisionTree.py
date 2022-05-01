from platform import node
from posixpath import split
from webbrowser import MacOSX
import numpy as np

class Tree_node:
    """
    Data structure for nodes in the decision-tree
    """
    def __init__(self,):
        self.feature = None # index of the selected feature (for non-leaf node)
        self.label = -1 # class label (for leaf node), -1 means the node is not a leaf node
        self.left_child = None # left child node
        self.right_child = None # right child node

class Decision_tree:
    """
    Decision tree with binary features
    """
    def __init__(self,min_entropy):
        self.min_entropy = min_entropy
        self.root = None


    def fit(self,train_x,train_y):
        # construct the decision-tree with recursion
        self.root = self.generate_tree(train_x,train_y)


    def predict(self,test_x):
        # iterate through all samples
        cur_node = Tree_node()
        
        prediction = np.zeros([len(test_x),]).astype('int') # placeholder
        for i in range(len(test_x)):
            cur_node = self.root

            for j in range(len(test_x[0])):

                if test_x[i][j] == 1 :
                    if cur_node.right_child != None:
                        cur_node = cur_node.right_child
                    else:
                        prediction[i] = cur_node.label
                        break

                elif test_x[i][j] == 0:
                    if cur_node.left_child != None:
                        cur_node = cur_node.left_child
                    else:
                        prediction[i] = cur_node.label
                        break
                    
        return prediction


    def generate_tree(self,data,label):
        # initialize the current tree node
        cur_node = Tree_node()

        # compute the node entropy
        node_entropy = self.compute_node_entropy(label)

        # determine if the current node is a leaf node based on minimum node entropy 
        # (if yes, find the corresponding class label with majority voting and exit the current recursion)
        if(node_entropy <= self.min_entropy):
            label_list = label.tolist()
            majority_vote = np.argmax(np.bincount(label_list))        #find mode of labels for cur_node
            cur_node.label = majority_vote
            return cur_node

        # select the feature that will best split the current non-leaf node
        selected_feature = self.select_feature(data,label)
        cur_node.feature = selected_feature

        # split the data based on the selected feature and start the next level of recursion
 
        right_branch =  data[data[:,selected_feature] == 1]
        right_labels = label[data[:,selected_feature] == 1]
        left_branch  =  data[data[:,selected_feature] == 0]
        left_labels  = label[data[:,selected_feature] == 0]

        cur_node.right_child = self.generate_tree(right_branch,right_labels)
        cur_node.left_child  = self.generate_tree(left_branch,left_labels)
        return cur_node


    def select_feature(self,data,label):
        # iterate through all features and compute their corresponding entropy
        best_feat = -1
        split_entropies = []

        for i in range(len(data[0])):

            # compute the entropy of splitting based on the selected features
            left = label[data[:,i] == 0]
            right = label[data[:,i] == 1]

            split_entropies.append(self.compute_split_entropy(left ,right))
            # select the feature with minimum entropy
        
        best_feat = np.argmin(split_entropies)
        
        return best_feat


    def compute_split_entropy(self,left_y,right_y):
        # compute the entropy of a potential split (with compute_node_entropy function), left_y and right_y
        #  are labels for the two branches
        split_entropy = -1 # placeholder
        
        left_node_entropy = self.compute_node_entropy(left_y)
        right_node_entropy = self.compute_node_entropy(right_y)
        
        num_of_labels = left_y.size + right_y.size
        split_entropy = (left_y.size / num_of_labels) * left_node_entropy +(right_y.size / num_of_labels) * right_node_entropy
        return split_entropy


    def compute_node_entropy(self,label):
        # compute the entropy of a tree node 
        # (add 1e-15 inside the log2 when computing the entropy to prevent numerical issue)
        node_entropy = 0 # placeholder
        label_occurences = np.bincount(label)
        for j in range(len(label_occurences)):
            cur_label_occurence = label_occurences[j]
            node_entropy += (cur_label_occurence/len(label)) * np.log2((cur_label_occurence /len(label)) + 1e-15)
        node_entropy = node_entropy * -1
        return node_entropy
