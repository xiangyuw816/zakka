def insert(root, node):
    if not root:
        root = node
    else:
        if root.val < node.val:
            if root.right is None:
                root.right = node
            else:
                insert(root.right, node)
        else:
            if root.left is None:
                root.left = node
            else:
                insert(root.left, node)
    return root


def inorder(root):
    if not root:
        pass  # exist
    inorder(root.left)
    # do something
    inorder(root.right)


def inOrder(root):
    # http://www.geeksforgeeks.org/inorder-tree-traversal-without-recursion/
    # Set current to root of binary tree
    current = root
    s = []  # initialze stack
    done = 0

    while (not done):
        # Reach the left most Node of the current Node
        if current is not None:
            # Place pointer to a tree node on the stack
            # before traversing the node's left subtree
            s.append(current)
            current = current.left

            # BackTrack from the empty subtree and visit the Node
        # at the top of the stack; however, if the stack is
        # empty you are done
        else:
            if (len(s) > 0):
                current = s.pop()
                print current.data,

                # We have visited the node and its left
                # subtree. Now, it's right subtree's turn
                current = current.right
            else:
                done = 1


"""315. Count of Smaller Numbers After Self
# insert numbers into BST and use leftTreeSize as the result"""


class BinarySearchTreeNode(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.count = 1
        self.leftTreeSize = 0


class BinarySearchTree(object):
    def __init__(self):
        self.root = None

    # insert val and return num_smaller
    def insert(self, val, root):
        if not root:
            # the last element will have 0 as final result
            self.root = BinarySearchTreeNode(val)
            return 0

        if val == root.val:
            # if the number is just the same as previous one, use the result of the previous one
            root.count += 1
            return root.leftTreeSize

        if val < root.val:
            # if the number is smaller, add 1 for the root
            root.leftTreeSize += 1
            ## to insert the node: 1. if leaf, directly insert; 2. if not, continue with left subtree
            if not root.left:
                # no root.left means it's leaf, so it's ready to insert
                root.left = BinarySearchTreeNode(val)
                return 0  # also there's nothing smaller than val
            return self.insert(val, root.left)
        # if the number is larger: 1. if leaf, directly insert; 2. if not, continue with right subtree
        if not root.right:
            root.right = BinarySearchTreeNode(val)
            return root.count + root.leftTreeSize

        return root.count + root.leftTreeSize + self.insert(val, root.right)


class Solution(object):
    def countSmaller(self, nums):
        tree = BinarySearchTree()
        return [tree.insert(nums[i], tree.root) for i in range(len(nums) - 1, -1, -1)][::-1]


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


"""538. Convert BST to Greater Tree
  #  do a reverse inorder traverse"""


class Solution(object):
    def convertBST(self, root):
        # given node, generate the greater tree node (define node.val, node.left, and node.right)
        # new node.val = SUM (all nodes in right subtree) + node.val
        def generate_greater_tree(node):
            if not node: return None
            right = generate_greater_tree(node.right)
            self.sum += node.val
            new_node = TreeNode(self.sum)
            new_node.right = right
            new_node.left = generate_greater_tree(node.left)
            return new_node

        self.sum = 0
        return generate_greater_tree(root)


"""530. Minimum Absolute Difference in BST
# do an inorder traverse, then find min abs in nearby
"""


class Solution(object):
    def dfs(self, root):
        if not root:
            return []
        return self.dfs(root.left) + [root.val] + self.dfs(root.right)

    def getMinimumDifference(self, root):
        l = self.dfs(root)
        return min([abs(b - a) for a, b in zip(l, l[1:])])


"""108. Convert Sorted Array to Binary Search Tree
# balanced tree -> find root first"""


class Solution(object):
    def sortedArrayToBST(self, num):
        if not num: return None
        mid = len(num) // 2
        root = TreeNode(num[mid])
        root.left = self.sortedArrayToBST(num[:mid])
        root.right = self.sortedArrayToBST(num[mid + 1:])
        return root


"""270. Closest Binary Search Tree Value
# either in one subtree or at root"""


class Solution(object):
    def closestValue(self, root, target):
        subtree = root.right if root.val < target else root.left
        if not subtree:
            return root.val
        cl = self.closestValue(subtree, target)
        return root.val if abs(root.val - target) < abs(cl - target) else cl


"""235. Lowest Common Ancestor of a Binary Search Tree
# Just walk down from the whole tree's root
 as long as both p and q are in the same subtree
  (meaning their values are both smaller or both larger than root's)."""


class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        while (root.val - p.val) * (root.val - q.val) > 0:
            root = (root.left, root.right)[p.val > root.val]
        return root


"""230. Kth Smallest Element in a BST
# ct = countNodes(root.left)
## if countNodes < k, find in left subtree
## if countNodes > k, find in right subtree (root.right, k - ct)
# Naive method: do an inorder-traverse, return list[k-1]
"""


def kthSmallest(root, k):
    def count_nodes(root):
        if not root:
            return 0
        return 1 + count_nodes(root.left) + count_nodes(root.right)

    ct = count_nodes(root.left)
    if ct >= k:
        return kthSmallest(root.left, k)
    elif ct < k:
        return kthSmallest(root.right, k - ct - 1)
    else:
        return root


"""173. Binary Search Tree Iterator
# do an in-order traverse and pop"""

"""96. Unique Binary Search Trees
# G(n): the number of unique BST for a sequence of length n.
# F(i, n), 1 <= i <= n: the number of unique BST, where the number i is the root of BST, and the sequence ranges from 1 to n.
# F(i, n) = G(i-1) * G(n-i)
# G(n) = F(1, n) + F(2, n) + ... + F(n, n)"""


def num_trees(n):
    g = [0] * (n + 1)
    g[0] = g[1] = 1
    for i in range(2, n + 1):
        for j in range(1, i + 1):
            g[i] += g[j - 1] * g[i - j]
    return g[n]


"""99. Recover Binary Search Tree
# do an in-order traverse, if prevElement > node.val, firstElement
# after find firstElement, do the same thing for secElement"""


def recoverTree(root):
    element1, element2 = traverse(root, None, None)
    return element1, element2


def traverse(root, element1, element2):
    if not root:
        return
    traverse(root.left, element1, element2)
    # do some business
    if not element1 and prev_ele >= root.val:
        element1 = prev_ele
    if element1 and prev_ele >= root.val:
        element2 = root
    prev_ele = root
    traverse(root.right, element1, element2)


"""285. Inorder Successor in BST
# the smallest value that's larger than the val"""


def inorderSuccessor(self, root, p):
    succ = None
    while root:
        if p.val < root.val:
            succ = root
            root = root.left
        else:
            root = root.right
    return succ


"""220. Contains Duplicate III"""


# consecutive buckets covering the range of nums with each bucket a width of (t+1).
# If there are two item with difference <= t, one of the two will happen:
# same bucket OR neighbor bucket
# t: gap between nums; k: gap between index
def containsNearbyAlmostDuplicate(self, nums, k, t):
    if t < 0: return False
    n = len(nums)
    d = {}
    w = t + 1
    for i in xrange(n):
        m = nums[i] / w
        if m in d:
            return True
        if m - 1 in d and abs(nums[i] - d[m - 1]) < w:
            return True
        if m + 1 in d and abs(nums[i] - d[m + 1]) < w:
            return True
        d[m] = nums[i]
        if i >= k: del d[nums[i - k] / w]
    return False
