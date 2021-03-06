"""291. Word Pattern II"""
class solution():
    def wordPatternMatch(self, pattern, str):
        return self.dfs(pattern, str, {})

    def dfs(self, pattern, str, dict):
        # dict to store existing patterns {sub-pattern: sub-string}
        if len(pattern) == 0 and len(str) > 0:
            return False
        if len(pattern) == len(str) == 0:
            return True

        # try use pattern[0] to match with each str[:end]
        # reduce search range by len(str)-len(pattern)+2 i.e. the largest sub-string length
        for end in range(1, len(str)-len(pattern)+2): # i.e. only 1 char in pattern has > one char
            # try to add a new pattern
            if pattern[0] not in dict and str[:end] not in dict.values():
                dict[pattern[0]] = str[:end]
                if self.dfs(pattern[1:], str[end:], dict):
                    return True
                # back tracking: to remove
                del dict[pattern[0]]
            elif pattern[0] in dict and dict[pattern[0]] == str[:end]:
                if self.dfs(pattern[1:], str[end:], dict):
                    return True

        return False

    
"""290. Word Pattern"""
class Solution(object):
    def wordPattern(self, pattern, str):
        # if one unique p-word pair exists
        # i.e. ('abba', 'dog dog dog dog') ('aaaa', 'mi ao mi ao') not working
        s = str.split()
        t = pattern
        return len(set(zip(s,t))) == len(set(s)) == len(set(t)) and len(s) == len(t)
    
    
"""Maximum Size Subarray Sum Equals k"""
# continuous sub-array
def maxSubArrayLen(nums, k):
    ans, acc = 0, 0               # answer and the accumulative value of nums
    mp = {0:-1}                 #key is acc value, and value is the index
    for i in xrange(len(nums)):
        acc += nums[i]
        if acc not in mp:
            mp[acc] = i
        if acc-k in mp:
            print(i, mp, i-mp[acc-k])
            ans = max(ans, i-mp[acc-k])
    return ans


"""48. Rotate Image"""
'''* clockwise rotate
 * first reverse up to down, then swap the symmetry 
 * 1 2 3     7 8 9     7 4 1
 * 4 5 6  => 4 5 6  => 8 5 2
 * 7 8 9     1 2 3     9 6 3'''
a[::-1]
'''anticlockwise rotate
 * first reverse left to right, then swap the symmetry
 * 1 2 3     3 2 1     3 6 9
 * 4 5 6  => 6 5 4  => 2 5 8
 * 7 8 9     9 8 7     1 4 7'''
[x[::-1] for x in a]


"""366. Find Leaves of Binary Tree""" # --> need to test
# DFS: to find the order of each node
class Solution(object):
    def __init__(self):
        self.dic = collections.defaultdict(list)
        
    def findLeaves(self, root):
        def order(root):
            """1. Record root's level to dic; 2. return root's level
            dic: {level: [node.val]}"""
            if not root:
                return 0
            left = order(root.left, dic)
            right = order(root.right, dic)
            lev = max(left, right) + 1
            self.dic[lev] += root.val,
            return lev
        
        ret = []
        order(root)
        for i in range(1, len(self.dic) + 1):
            ret.append(self.dic[i])
        return ret
    
    
"""339. Nested List Weight Sum"""
# {level: [elements]}
# level: how many times until the element pop out
class Solution(object):
    def __init__(self):
        self.dic = defaultdict(list)
        
    def solution(self, ls):
        self.helper(ls, 1)
        res = 0
        for k, v in self.dic.items():
            res += sum(x*k for x in v)
        return res
        
    def helper(self, ls, level):
        # update ls to self.dic based on level
        while ls:
            temp = ls.pop(0)
            if not isinstance(temp, list):
                self.dic[level] += [temp]
            else:
                self.helper(temp, level+1)
   
# Method2: sum along the way
def depthSum(nestedList):
    def DFS(nestedList, depth):
        """sum over the nestedList given initial depth level"""
        temp_sum = 0
        for member in nestedList:
            if isinstance(member, int):
                temp_sum += member * depth
            else:
                temp_sum += DFS(member,depth+1)
        return temp_sum
    return DFS(nestedList,1)


"""459. Repeated Substring Pattern"""
# If we repeat the string, then SS=SpSpSpSp.
# Destroying first and the last pattern by removing each character, we generate a new S2=SxSpSpSy
def repeatedSubstringPattern(self, str):
    return str in (2 * str)[1:-1]


"""277. Find the Celebrity"""
# 1. find celebrity candidate, similar to find min
# 2. verify this candidate: if all(!s/he knows) = True and all knows candidate


"""380. Insert Delete GetRandom O(1)"""
# list [vals] , dic {val: index in list}
## when remove a value, copy the last value to that position (also update dict) and list.pop()
import random
class RandomizedSet(object):

    def __init__(self):
        self.nums, self.pos = [], {}
        
    def insert(self, val):
        if val not in self.pos:
            self.nums.append(val)
            self.pos[val] = len(self.nums) - 1
            return True
        return False
        

    def remove(self, val):
        if val in self.pos:
            idx, last = self.pos[val], self.nums[-1]
            self.nums[idx], self.pos[last] = last, idx
            self.nums.pop(); self.pos.pop(val, 0)
            return True
        return False
            
    def getRandom(self):
        return self.nums[random.randint(0, len(self.nums) - 1)]
    

"""254. Factor Combinations"""
# split n to [i, n/i] and then further split n/i
# use i to control duplicates i.e. non-descending
# if no i: for n = 16, [2,2,4] and [4,2,2] will both occur
def getFactors(n):
    def factor(n, i, path, res):
    # split n by >=i and append it with path to res
        while i * i <= n:
            if n % i == 0:
                res.append(path + [i, n//i])
                factor(n//i, i, path+[i], res)
            i += 1
        return res
    return factor(n, 2, [], [])


"""249	Group Shifted Strings"""
# use %26 to get same pattern for 'zab', 'abc'
from collections import defaultdict
def groupStrings(strings):
    dic = defaultdict(list)
    for s in strings:
        dic[tuple((ord(c) - ord(s[0])) % 26 for c in s)].append(s)
    return dic.values()


"""36. Valid Sudoku"""
class Solution(object):
    def isValidSudoku(self, board):
        # use a set to maintain those num already shown up with format:
        # num + ' in row' + i, num + ' in col' + j, num + ' in block (i/3,j/3)'
        records = set()
        for i in range(len(board)):
            for j in range(len(board)):
                num = board[i][j]
                if num != '.':
                    if '{} in row {}'.format(num, i) in records or '{} in col {}'.format(num, j) in records or '{} in block ({}, {})'.format(num, i//3,j//3) in records:
                        return False
                    
                    records.add('{} in row {}'.format(num, i))
                    records.add('{} in col {}'.format(num, j))
                    records.add('{} in block ({}, {})'.format(num, i//3,j//3))
                    
        return True


"""54	Spiral Matrix"""
# use nrow != ncol as test case
class Solution(object):
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if not matrix:
            return []
        res = []
        row_begin, row_end = 0, len(matrix) - 1
        col_begin, col_end = 0, len(matrix[0]) - 1
        while row_begin <= row_end and col_begin <= col_end:
            # traverse to right
            for i in range(col_begin, col_end+1):
                res.append(matrix[row_begin][i])
            row_begin += 1
            # traverse to bottom
            for j in range(row_begin, row_end+1):
                res.append(matrix[j][col_end])
            col_end -= 1
            # traverse to left
            for i in range(col_end, col_begin-1, -1):
                res.append(matrix[row_end][i])
            row_end -= 1
            # traverse to upper
            for j in range(row_end, row_begin-1, -1):
                res.append(matrix[j][col_begin])
            col_begin += 1
        return res
                
        
"""49. Group Anagrams"""
class Solution(object):
    def groupAnagrams(self, strs):
        # Anagrams are the same when sorted, so use sorted string as the key
        # can improve the time complexity by using sorting algo like counting sort
        dic = collections.defaultdict(list)
        for s in strs:
            dic[''.join(sorted(s))].append(s)
            
        return dic.values()
        

"""251. Flatten 2D Vector"""
# no need for iterators, just update row,col index
# !! important case: [[],[],[-1]], if encounter [], need to continue to check --> while loop
class Vector2D(object):
    def __init__(self, vec2d):
        self.row = 0
        self.col = 0
        self.vec = vec2d

    def next(self):
        # The first time to call next should return the 1st element
        if self.hasNext:
            res = self.vec[self.row][self.col]
            self.col += 1
            return res

    def hasNext(self):
        """
        if hasNext: update index else return False
        - as long as row is not out of range, we need to continue checking
        - if found one valid index, just return to break the loop
        """
        while self.row < len(self.vec):
            if self.col < len(self.vec[self.row]):
                return True
            
            self.row += 1
            self.col = 0
            
        return False

    
"""325. Maximum Size Subarray Sum Equals k"""
# mp: {accumulate_value: left-most index}
def maxSubArrayLen(self, nums, k):
    ans, acc = 0, 0 # answer and the accumulative value of nums
    mp = {0:-1} # initialize 0! helpful when e.g. [1,-1]
    for i in xrange(len(nums)):
        acc += nums[i]
        if acc not in mp: # don't need to update value of dict, because we need the max size i.e. start from left-most
            mp[acc] = i 
        if acc-k in mp:
            ans = max(ans, i-mp[acc-k]) # sum(nums[acc-k+1, i]) = k
    return ans


"""Largest Sum Contiguous Subarray"""
# current_max: still possible when accu_sum >= 0
## ? greedy: a local max and global max
def maxSubArraySum(a,size):
    max_so_far =a[0]
    curr_max = a[0]
     
    for i in range(1,size):
        curr_max = max(a[i], curr_max + a[i])
        max_so_far = max(max_so_far,curr_max)
         
    return max_so_far

"""450. Delete Node in a BST"""
# use the smallest value in right sub-tree to replace the value to be deleted
def deleteNode(root, key):
    if not root:
        return root
    
    if root.val > key:
        root.left = deleteNode(root.left, key)
    elif root.val < key:
        root.right = deleteNode(root.right, key)
    else: # found the node to be deleted
        if not root.left: # if either sub-tree not exist, return the other subtree
            return root.right
        elif not root.right:
            return root.left
        root.val = get_min(root.right)
        root.right = deleteNode(root.right, root.val)
        
def get_min(root):
    # find the min in a tree i.e. left-most leaf
    m = root.val
    while root.left:
        root = root.left
        m = root.val
    return m


"""2. Add Two Numbers"""
# Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
# Output: 7 -> 0 -> 8
# i.e. 342 + 465 = 708
def addTwoNumbers(self, l1, l2):
    carry = 0
    root = n = ListNode(0)
    while l1 or l2 or carry:
        v1 = v2 = 0 # fillna
        if l1:
            v1 = l1.val
            l1 = l1.next
        if l2:
            v2 = l2.val
            l2 = l2.next
        carry, val = divmod(v1+v2+carry, 10)
        n.next = ListNode(val)
        n = n.next
        
    return root.next


"""155. Min Stack"""
# when `push`, use (val, current_min) to update the stack
class MinStack:
    def __init__(self):
        self.q = []

    def push(self, x):
        curMin = self.getMin()
        if not curMin or x < curMin:
            curMin = x
        self.q.append((x, curMin));

    def pop(self):
        self.q.pop()

    def top(self):
        if not self.q:
            return None
        return self.q[-1][0]

    def getMin(self):
        if not self.q:
            return None
        return self.q[-1][1]
        

"""206. Reverse Linked List"""
def reverseList(self, head):
    return self.helper(head, None)

def helper(self, node, prev):
    # point node.next to prev
    if not node: # return the beginning of the linkedList
        return prev
    after = node.next
    node.next = prev
    return self.helper(after, node)


"""658. Find K Closest Elements"""
class Solution(object):
# to find the start point of sub-array
    def findClosestElements(self, arr, k, x):
        l, r = 0, len(arr) - k
        while l < r:
            mid = (l+r)//2
            # mid+k is the start of right half
            if x - arr[mid] > arr[mid+k] - x:
                l = mid + 1
            else:
                r = mid
        return arr[l:l+k]
    

"""360. Sort Transformed Array"""
def sortTransformedArray(self, nums, a, b, c):
    # two pointers to traverse from two ends
    # if a>0: the max should be from two ends (half sorted)
    nums = [x*x*a + x*b + c for x in nums]
    ret = [0] * len(nums)
    p1, p2 = 0, len(nums) - 1
    i, step = (p1, 1) if a < 0 else (p2, -1)
    while p1 <= p2:
        if nums[p1] * -step > nums[p2] * -step:
            ret[i] = nums[p1]
            p1 += 1
        else:
            ret[i] = nums[p2]
            p2 -=1
        i += step
    return ret
