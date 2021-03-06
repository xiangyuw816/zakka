"""	101	Symmetric Tree"""
class Solution:
  def isSymmetric(self, root):
    if root is None:
      return True
    else:
      return self.isMirror(root.left, root.right)

  def isMirror(self, left, right):
    if left is None and right is None:
      return True
    if left is None or right is None:
      return False

    if left.val == right.val:
      outPair = self.isMirror(left.left, right.right)
      inPiar = self.isMirror(left.right, right.left)
      return outPair and inPiar
    else:
      return False

"""26. Remove Duplicates from Sorted Array"""
class Solution:
# Use newTail to keep unique index, only +=1 when the value is different from the tail
    def removeDuplicates(self, A):
        if not A:
            return 0

        newTail = 0

        for i in range(1, len(A)):
            if A[i] != A[newTail]:
                newTail += 1
                A[newTail] = A[i]

        return newTail + 1
      
"""27. Remove Element"""
def removeElement(self, nums, val):       
        nextPos = 0
        for num in nums:
            if num != val:
                nums[nextPos] = num
                nextPos += 1
        return nextPos

"""118. Pascal's Triangle"""
# actually 1st and last elemenet of each row are both 1
# res.append([1]*(i+1))
class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        if numRows == 0:
            return []
        res = [[1]]
        for i in range(1, numRows):
            res.append([])
            for j in range(i+1):
                res[i].append((res[i-1][j-1] if j>0 else 0) + (res[i-1][j] if j<i else 0))
        return res
      
"""242. Valid Anagram"""
# sort two string and compare: sorted(s) == sorted(t)
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) != len(t):
            return False

        alpha = {}
        beta = {}
        for c in s:
            alpha[c] = alpha.get(c, 0) + 1
        for c in t:
            beta[c] = beta.get(c, 0) + 1
        return alpha == beta
      
"""463. Island Perimeter"""
# count different 1/0
class Solution(object):
    def islandPerimeter(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if not grid:
            return 0

        def sum_adjacent(i, j):
            adjacent = (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1),
            res = 0
            for x, y in adjacent:
                if x < 0 or y < 0 or x == len(grid) or y == len(grid[0]) or grid[x][y] == 0:
                    res += 1
            return res

        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    count += sum_adjacent(i, j)
        return count

"""1. Two Sum"""
# 3 sums: loop over num and target = -num.
class Solution:
    def twoSum(self, nums, target):
        # dict = {'diff till target': index}
        map = {}
        for i in range(len(num)):
            if num[i] not in map:
                map[target - num[i]] = i + 1
            else:
                return map[num[i]], i + 1

        return -1, -1
      
""""Sliding window problems / substring"""
# https://discuss.leetcode.com/topic/68976/sliding-window-algorithm-template-to-solve-all-the-leetcode-substring-search-problem
## move end pointer (one while loop) and if hit the target (one while loop) move begin pointer
## quick template
def solution(string, target):
    # initialize
    begin = end = 0
    res = [] # depends
    # CONCERNER CASE
    # if target
    still_need_dict = {}
    for char in still_need_dict:
        still_need_dict[char] += 1
    counter = len(still_need_dict) # map size
        
    while end < len(string):
        c = string[end]
        if c in still_need_dict.keys():
            still_need_dict[c] -= 1 # can be NEGATIVE
            if still_need_dict[c] == 0:
                counter -= 1
                
        while counter == 0: # when satisfy the condition, move begin pointer
            temp_c = string[begin]
            if c in still_need_dict.keys():
                still_need_dict[temp_c] += 1
                if still_need_dict[temp_c] > 0: # !!! condition is >0
                    counter += 1
                    
            # update result
            # pass
            begin += 1
            
"""438. Find All Anagrams in a String"""
# counter = len(map)
# to update result: end - begin == len(target)
class Solution(object):
    # e.g input("cbaebabacd","abc") return [0, 6]
    def findAnagrams(self, string, target):
        result = []#  or int to save results
        if len(target)>len(string):
            return result
        
        # Initialize a hashmap to save characters of the target substring
        ## {character: frequence of the characters}
        still_need_dict = {}
        for c in target:
            still_need_dict[c] = still_need_dict.get(c, 0) + 1
        # maintain a counter to check whether sub-string matches the target string
        counter = len(still_need_dict) # must be map size, for maybe dups in char
        
        # two pointers: begin: left of the window; end: right of the window
        begin = end = 0
        head = 0 # different use case
        while end < len(string):
            # add end char
            c = string[end]
            if c in still_need_dict.keys():
                still_need_dict[c] -= 1 # plus / minus one
                if still_need_dict[c] == 0:
                    counter -= 1 # modify counter according to requirements
            
            end += 1
            
            # remove begin char
            while counter == 0: # different situations
            # keep move begin pointer for there's nothing qualified
                tempc = string[begin]
                if tempc in still_need_dict.keys():
                    still_need_dict[tempc] += 1
                    if still_need_dict[tempc] > 0:
                        counter += 1 # modify count for different situations
                
                #### update result
                if end-begin == len(target): # different situations
                    result.append(begin)
    
                begin += 1
                
        return result
      
"""76. Minimum Window Substring"""
# use head & smallest_len to tract result
# to update result: end - begin < smallest_len
class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        if len(s)<len(t):
            return ''
        head = 0
        smallest_len = len(s)+1
        still_need_dict = {} # what's still needed in order to get T
        for char in t:
            still_need_dict[char] = still_need_dict.get(char, 0) + 1
        counter = len(still_need_dict)
        end = 0
        begin = 0 # each time move end pointer, update still_need_dict
        # when still_need_dict empty: move begin pointer & update
        # after finish move: compare current end-begin with len
        # use head and len to maintain the result
        while end < len(s):
            c = s[end]
            if c in still_need_dict.keys():
                still_need_dict[c] -= 1
                if still_need_dict[c] == 0:# it can be negative
                    counter -= 1
            end += 1
            
            while not counter:
                temp_c = s[begin]
                if temp_c in still_need_dict.keys():
                    still_need_dict[temp_c] += 1
                    if still_need_dict[temp_c]>0:
                        counter += 1

                if end - begin < smallest_len:
                    head = begin
                    smallest_len = end - begin

                begin += 1
                
        if smallest_len > len(s):
            return ''
        return s[head: head+smallest_len]
      
"""3. Longest Substring Without Repeating Characters"""
# counter: num of duplicated chars
# map: track char frequency in the sub-string
    def lengthOfLongestSubstring(self, s):
        # move end pointer and update counter
        # if counter > 0: move begin pointer until no dups in s[begin:end]
        map = {}
        begin = 0
        end = 0
        counter = 0
        d = 0 # final result
        while end < len(s):
            c = s[end]
            map[c] = map.get(c, 0) + 1
            if map[c] > 1:# if there's dup char
                counter += 1
            end += 1
            
            while counter>0:
                temp_c = s[begin]
                if map[temp_c] > 1:
                    counter -= 1
                map[temp_c] -= 1
                begin += 1
            # update result only when there's no dups at all    
            d = max(d, end - begin)
            
        return d
      
"""159. Longest Substring with At Most Two Distinct Characters"""
# counter: num of distinct chars
# begin++ needs to be inside while loop
# condition: when counter > 2, to move begin pointer
class Solution(object):
    def lengthOfLongestSubstringTwoDistinct(self, s):
        """
        :type s: str
        :rtype: int
        """
        d = 0
        begin = 0
        end = 0
        tf = {}
        counter = 0
        # tf: track tf of sub-string
        # counter: num of distinct chars
        while end < len(s):
            c = s[end]
            if tf.get(c, 0) == 0:
                counter += 1
            tf[c] = tf.get(c, 0) + 1
            end += 1
            
            while counter > 2:
                temp_c = s[begin]
                if tf[temp_c] == 1:
                    counter -= 1
                tf[temp_c] -= 1
                begin += 1
                
            d = max(d, end-begin)
            
        return d
      
"""102. Binary Tree Level Order Traversal"""
# Use level to store *tree nodes* of current level
def levelOrder(self, root):
    if not root:
        return []
    ans, level = [], [root]
    while level:
        # put values of nodes in level to answer
        ans.append([node.val for node in level])
        temp = []
        # Update level: put all non-null sub-nodes of levels into level
        for node in level:
            temp.extend([node.left, node.right])
        level = [leaf for leaf in temp if leaf]
    return ans

"""11. Container With Most Water"""
# two pointers
# TARGET: min(a[i], a[j]) * abs(i - j)
# every time move the one with smaller value
class Solution(object):
    def maxArea(self, height):
        left, right = 0, len(height) - 1
        ans = 0
        while left < right:
            if height[left] < height[right]:
                area = height[left] * (right - left)
                left += 1
            else:
                area = height[right] * (right - left)
                right -= 1
            ans = max(ans, area) 
        return ans
      
"""4. Median of Two Sorted Arrays"""
# Use a more general function as helper
class Solution:
    def findMedianSortedArrays(self, A, B):
        n = len(A) + len(B)
        if n % 2 == 1:
            return self.findKth(A, B, n / 2 + 1)
        else:
            smaller = self.findKth(A, B, n / 2)
            bigger = self.findKth(A, B, n / 2 + 1)
            return (smaller + bigger) / 2.0

    def findKth(self, A, B, k):
      # find the kth element in [A, B]
      # divide & conquer
        if len(A) == 0:
            return B[k - 1] # k starts from 1 while index starts from 0
        if len(B) == 0:
            return A[k - 1]
        if k == 1:
            return min(A[0], B[0])
        
        a = A[k / 2 - 1] if len(A) >= k / 2 else None
        b = B[k / 2 - 1] if len(B) >= k / 2 else None
        
        # len(B) < k/2 OR a < b --> no need to try A[:k/2]
        if b is None or (a is not None and a < b):
            return self.findKth(A[k / 2:], B, k - k / 2)
        return self.findKth(A, B[k / 2:], k - k / 2)
      
"""153. Find Minimum in Rotated Sorted Array"""
# use two pointers to find the rotated part
# sorted array --> skip part of the array
class Solution(object):
    def findMin(self, nums):
        l = 0
        r = len(nums)-1
        while l<r:
            # sub-array is not rotated
            if nums[l] < nums[r]:
                return nums[l]
              
            mid = (l+r)//2
            if nums[mid] > nums[r]:
                l = mid + 1
            else:
                r = mid
                
        return nums[l]

"""DP"""
# one method to formulate DP problem:
## helper: the target value when sth ends at position i.

"""the longest increasing sequence"""
# f[n]: if the sequence *ends at n*, the longest sequence possible
# f[n] = max(f[i-1] + 1 for i in range(n) if array[i-1] < array[n])
## the list inside max can be empty
def lis(ls):
    res = [0]*len(ls)
    for i in range(len(ls)):
        temp = [res[j] for j in range(i) if ls[j]<ls[i]]
        res[i] = max(temp) + 1 if temp else 1
    return max(res)


# Maximum Sum Increasing Subsequence
# f[n] = max(f[i-1] + array[n] for i in range(n) if array[i-1] < array[n])
## the list inside max can be empty
def lis(ls):
    res = [0]*len(ls)
    for i in range(len(ls)):
        temp = [res[j] for j in range(i) if ls[j]<ls[i]]
        res[i] = max(temp) + ls[i] if temp else ls[i]
    return max(res)

  
  
"""139. Word Break"""
# as long as ok[j] AND ok[j:i] in dict
def wordBreak(self, s, words):
    ok = [True]
    for i in range(1, len(s)+1):
        ok += any(ok[j] and s[j:i] in words for j in range(i)),
    return ok[-1]

  
"""70. Climbing Stairs"""
# how many *distinct* ways can you climb to the top
# link function: f(n) = f(n-1) + f(n-2)
# f(1) = 1, f(2) = 2  
# Top down + memorization (dictionary)  
def __init__(self):
    self.dic = {1:1, 2:2}
    
def climbStairs(self, n):
    if n not in self.dic:
        self.dic[n] = self.climbStairs(n-1) + self.climbStairs(n-2)
    return self.dic[n]
  

"""152. Maximum Product Subarray"""
# max/min product for the current number is either the current number itself
# or the max/min by the previous number times the current one
class Solution:
    def maxProduct(self, nums):
        f, g = [], []
        f.append(nums[0])
        g.append(nums[0])
        for i in xrange(1, len(nums)):
            f.append(max(f[i-1]*nums[i], g[i-1]*nums[i], nums[i]))
            g.append(min(f[i-1]*nums[i], g[i-1]*nums[i], nums[i]))
        m = f[0]
        for i in xrange(1, len(f)): m = max(m, f[i])
        return m
      
      
"""152. Maximum Product Subarray"""
# two negatives treated as one positive
# only two patterns: AbCd or aAbCd
def maxProduct(nums):
    front_prod = 1
    back_prod = 1
    res = 0
    n = len(nums)
    for i in range(n):
        front_prod *= nums[i]
        back_prod *= nums[n-i-1]
        res = max(res, max(front_prod, back_prod))
        front_prod = 1 if front_prod == 0 else front_prod
        back_prod = 1 if front_prod == 0 else back_prod
    return res

def maxProduct(nums):
    # imax, imin: current max/min that ends at i
    # on adding new i: if num[i] > 0, num[i]*imax else num[i]*imin or maybe itself
    res = nums[0]
    imax = imin = nums[0]
    for i in range(1, len(nums)):
        if nums[i] < 0:
            imax, imin = imin, imax
        
        imax = max(nums[i], num[i]*imax)
        imin = min(nums[i], num[i]*imin)

        res =max(res, imax)
    return res
  

"""120. Triangle"""      
# go from bottom to top
# only need a list of minlen, because after calculation the result of lower paths does not matter
class Solution(object):
    def minimumTotal(self, triangle):
        n = len(triangle)
        minlen = triangle[n-1][:]# the minimum sum from current node to its children-path
        for layer in range(n-2, -1, -1): # iterate by layer
            for i in range(layer+1): # for each node in this layer
                # find the smaller of two children and sum the current value
                minlen[i] = min(minlen[i], minlen[i+1])+triangle[layer][i]
                
        return minlen[0]
      
      
"""6. ZigZag Conversion"""
class Solution(object):
    def convert(self, s, numRows):
      # Corner Case
        if numRows == 1 or len(s) < numRows:
            return s
        # initialize a list of strings to stand for elements in each row  
        index = 0
        down = True # control the direction to move index (+1 or -1)
        L=['']*numRows

        for char in s:
            L[index]+=char
            if index == numRows-1:
                down = False
            if index == 0:
                down = True
            index = index + 1 if down else index - 1

        return ''.join(L)

      
"""Magical vowls"""
# possible resutls are 5, so f(n) = [m1, m2, m3, m4, m5]
# stands for the possible outputs use 'a', 'e', ..., 'u' as end string with total length of n.
# f(1) = [1,1,1,1,1]
# f(n) = [f(n-1)1*num..., ..., ... ]


from collections import Counter
"""Counting Sort"""
def get_topK(text, K):
    words_freq = Counter(text.lower().split())
    if not words_freq:
        return

    # create a list to store words and use index as the frequency
    n = max(words_freq.values())
    freq_list = [[] for _ in range(n)] # [[]]*n will create n copy of [] —> modified same time
    for k,v in words_freq.items():
        freq_list[v-1].append(k)

    # get the topK words by traverse the list from the right side
    res = []
    freq = n
    while len(res) < k and freq_list:
        words = freq_list.pop()
        # no need to check if words == []. If empty, will automatically skip when extend
        res.extend([(x, freq) for x in sorted(words)[:K-len(res)]])
        freq -= 1

    for word, freq in res:
        print('{0} {1}'.format(word, freq))

        
"""547. Friend Circles"""
# Union Find: finding circles in un-directed graph
## Find: determine which subset an element is in (can be used to check if two elements belong to the same subset)
## Union: join two subsets into a single subset

# different from num_islands e.g. when M[0, len(M)-1] = 1
class Solution(object):
    def findCircleNum(self, M):
        """
        :type M: List[List[int]]
        :rtype: int
        """
        ds = DisjointSet()

        for i in range(len(M)):
            ds.make_set(i)

        for i in range(len(M)):
            for j in range(i, len(M)): # diagonal, no need to start j from 0
                if M[i][j] == 1:
                    ds.union(i, j)

        return ds.num_sets

class Node(object):
    def __init__(self, data, parent=None, rank=0):
        self.data = data # the actual data element
        self.parent = parent # pointer from child to parent
        self.rank = rank # depth of the tree

class DisjointSet(object):
    def __init__(self):
        self.map = {} # map element to its node
        self.num_sets = 0

    def make_set(self, data):
      """Initialization: only one element in a set
        and parent points to itself, rank = 0
      """
        node = Node(data)
        node.parent = node
        self.map[data] = node
        self.num_sets += 1

    def union(self, data1, data2):
      """Merge two sets into one
        1. find the root node of both element (i.e. find_set: root will be the one point to itself)
        2. compare rank: make parent the higher ranked one --> keep rank as small as possible
        3. update parent rank (rank of non-root does not matter): if same rank: root rank +1 else max(rank)
        4. path compression: make all non-root nodes point to root node directly (i.e. find_set_until)
      """
        node1 = self.map[data1]
        node2 = self.map[data2]

        parent1 = self.find_set_util(node1)
        parent2 = self.find_set_util(node2)

        if parent1.data == parent2.data:
            return

        if parent1.rank >= parent2.rank:
            if parent1.rank == parent2.rank:
                parent1.rank += 1
            parent2.parent = parent1
        else:
            parent1.parent = parent2

        self.num_sets -= 1


    def find_set(self, data):
      """Return an identity of a set which is usually an element in a set which acts as a rep of that set
          e.g 1,2,3 belong to the same set and 1 is the rep, so find_set(1),find_set(2),find_set(3) will all return 1.
      """
        return self.find_set_util(self.map[data])

    def find_set_util(self, node):
      """Path compression: make all non-root nodes point to root node directly"""
        parent = node.parent
        if not parent:
            return parent

        # along the way to find root, update the parent of all elements along the way to the root node.
        node.parent = self.find_set_util(node.parent) # path compression
        return node.parent


# BFS: stack could be [(x,y, string, step)] -- not limited to only pos!!!
"""200. Number of Islands"""
# BFS: no need for visited Instead: change every visited to 0
class Solution:
    def numIslands(self, grid):
        if not grid:
            return 0
    
        num_islands = 0
        nrow, ncol = len(grid), len(grid[0])
        for i in range(nrow):
            for j in range(ncol):
                q = collections.deque([])
                if grid[i][j]:
                    grid[i][j] = 0
                    num_islands += 1
                    q.append((i,j))
    
                while q:
                    m,n = q.popleft()
                    # no need to check (m,n), already did outside the loop
                    for d1,d2 in [(0,1),(0,-1),(1,0),(-1,0)]:
                        if 0<=m+d1<nrow and 0<=n+d2<ncol and grid[m+d1][n+d2]:
                            grid[m+d1][n+d2] = 0
                            q.append((m+d1, n+d2))
    
        return num_islands

"""130. Surrounded Regions"""
# 1. find all O that's connected to the boarder and label as 'S'
# 2. change all 'S' to 'O' and the rest as 'X'
class Solution:
    def surroundedRegions(self, board):
        # corner case!!
        if not board:
            return []
          
        # find all boarder elements
        nrow = len(board)
        ncol = len(board[0])
        if nrow == 1:
            boarder = [(0,j) for j in range(ncol)]
        elif ncol == 1:
            boarder = [(i,0) for i in range(nrow)]
        else:
            boarder = [(i,j) for j in (0, ncol-1) for i in range(nrow)]
            boarder.extend([(i,j) for i in (0, nrow-1) for j in range(1,ncol-1)])
        
        while boarder:
            i,j = boarder.pop()
            if 0<=i<nrow and 0<=j<ncol and board[i][j] == 'O':
                board[i][j] = 'S'
                boarder.extend([(i,j+1), (i,j-1), (i+1,j), (i-1,j)])
                
        for i in range(nrow):
            for j in range(ncol):
                board[i][j] = 'O' if board[i][j] == 'S' else 'X'
                
        return board
      
"""261. Graph Valid Tree"""
class Solution:
# Valid tree: 1) no cycle; 2) all connected
    # BFS
    def validTree(self, n, edges):
        # given n nodes and no repeat, valid tree should have n-1 edges
        if len(edges) != n - 1:
            return False

        neighbors = collections.defaultdict(list)
        for u, v in edges:
            neighbors[u].append(v)
            neighbors[v].append(u)

        visited = {}
        queue = [0]
        visited[0] = True
        while queue:
            cur = queue.pop()
            visited[cur] = True
            for node in neighbors[cur]:
                if node not in visited:
                    visited[node] = True
                    queue.append(node)

        # all connected
        return len(visited) == n

"""207. Course Schedule"""      
class Solution:
    # BFS with toplogical sorting
    # http://www.cnblogs.com/grandyang/p/4484571.html
    # put all nodes with 0 indegree into queue
    # iterate over queue and update those connected nodes with degree--
    # after all iteration, degree list should be 0
    def canFinish(self, numCourses, prerequisites):
        zeroInDegree = collections.deque([])
        # initialize a list of store the indegree of each node
        degree = [0] * numCourses
        for pre in prerequisites:
            degree[pre[1]] += 1
        # initialize queue
        for i in range(len(degree)):
            if degree[i] == 0:
                zeroInDegree.append(i)
                
        if not zeroInDegree:
            return False
        
        while zeroInDegree:
            course = zeroInDegree.popleft()
            # go over prerequisites to update indegree
            for i,j in prerequisites:
                if i == course:
                    degree[j] -= 1
                    if degree[j] == 0:
                        zeroInDegree.append(j)
        
        return sum(degree) == 0

      
"""257. Binary Tree Paths"""
# bfs + queue
def binaryTreePaths2(self, root):
    if not root:
        return []
    res, queue = [], collections.deque([(root, "")])
    while queue:
        node, ls = queue.popleft()
        if not node.left and not node.right:
            res.append(ls+str(node.val))
        if node.left:
            queue.append((node.left, ls+str(node.val)+"->"))
        if node.right:
            queue.append((node.right, ls+str(node.val)+"->"))
    return res
    
    
"""Backtracking & recursion DFS Template
def helper(sth, path, res)
# recursive call: declare exit condition
# only focus on current step -> update one element in sth to path

def solutioin(whole_sth):
    res = []
    helper(whole_sth, [], res)
    return res
"""    
# dfs recursively
def binaryTreePaths(self, root):
    if not root:
        return []
    res = []
    self.dfs(root, "", res)
    return res
# if need to update the result, then make res as one of the input param of the dfs recursive function.
def dfs(self, root, ls, res):
    if not root.left and not root.right:
        res.append(ls+str(root.val))
    if root.left:
        self.dfs(root.left, ls+str(root.val)+"->", res)
    if root.right:
        self.dfs(root.right, ls+str(root.val)+"->", res)
        
"""131. Palindrome Partitioning"""
def partition(self, s):
    res = []
    self.dfs(s, [], res)
    return res

def dfs(self, s, path, res):
  # s: string to be patitioned, path: already patitioned result, res: to put find result
  # function: partition s and append it to path, and finally append path to res.
    if not s:
        res.append(path)
        return
    for i in range(1, len(s)+1):
        if self.isPal(s[:i]):
            self.dfs(s[i:], path+[s[:i]], res)
    
def isPal(self, s):
    return s == s[::-1]

"""17. Letter Combinations of a Phone Number"""
def letterCombinations(self, digits):
    if not digits:
        return []
    dic = {"2":"abc", "3":"def", "4":"ghi", "5":"jkl", "6":"mno", "7":"pqrs", "8":"tuv", "9":"wxyz"}
    res = []
    self.dfs(digits, dic, 0, "", res)
    return res
    
# replace digits[index:] with the dictionary value and add to path    
def dfs(self, digits, dic, index, path, res):
    if len(path) == len(digits):
        res.append(path)
        return 
    for i in xrange(index, len(digits)):
        for j in dic[digits[i]]:
            # move index and update path
            self.dfs(digits, dic, i+1, path+j, res)
         
"""22. Generate Parentheses"""
# generate all possible pairs of valid ()
## another method: DP.
class Solution:
    # append #l '(' and #r ')' to item and append it to res.
    def helpler(self, l, r, item, res):
        if r < l: # num ')' inserted > num '(' inserted
            return # invalid - skip
        if l == 0 and r == 0: # all '('')' inserted
            res.append(item)
        if l > 0:
            self.helpler(l - 1, r, item + '(', res)
        if r > 0:
            self.helpler(l, r - 1, item + ')', res)
    
    def generateParenthesis(self, n):
        if n == 0:
            return []
        res = []
        self.helpler(n, n, '', res)
        return res
   
"""46. Permutations"""  
class Solution(object):
    def permute(self, nums):
        res = []
        self.helper(nums, [], res)
        return res
      
    # permutate elements and add them to path, and append res.
    def helper(self, elements, path, res):
        if not elements:
            res.append(path)
            return
        for i in range(len(elements)):
            self.helper(elements[:i]+elements[i+1:], path+[elements[i]], res)

            
"""89. Gray Code"""
# backtracking with iteration
## from f(n-1) to f(n): only need to [x+j for j in ('0','1') for x in f(n-1)]
def solution(n):
    rs=[0]
    for i in range(n):
        size=len(rs)
        for k in range(size-1, -1, -1):
            # a<<b: add b*0 to the right of a
            # |: or operator -> decimal
            rs.append(rs[k]|1<<i)
    return rs

""" 387. First Unique Character in a String""""
# go over list to obtain tf dict
# go over the string to return the first index with frequency equal to 1.
# As to find the 1st element, so just iterate from beginning to end!
def sol(string):
    tf = {}
    for i in range(len(string)):
        tf[string[i]] = tf.get(string[i], 0) + 1
    
    for i in range(len(string)):
        if tf[string[i]] == 1:
            return i
    return -1

  
"""Find intersections between several lists"""
# e.g. [1,2,3,2,5], [1,2,2], [2,3,2,6] --> [2,2,3]
# Use the list with least length to go over other lists.
def solution(a1,a2,a3):
    res = []
    # suppose a2 is the one with smallest length
    tf1 = Counter(a1) # build tf hashtable
    for term in tf2.keys():
        t = min(tf1.get(term,0), tf2[term], tf3.get(term, 0))
        if t > 0:
            res.extend([term]*t)
    
    return res

 
"""133. Clone Graph"""
class Solution:
    def cloneGraph(self, node):
        root = node
        if node is None:
            return node
            
        # BFS to traverse the graph and get all nodes.
        nodes = self.getNodes(node)
        
        # copy nodes, store the old->new mapping information in a hash map
        mapping = {}
        for node in nodes:
            mapping[node] = UndirectedGraphNode(node.label)
        
        # copy neighbors(edges)
        for node in nodes:
            new_node = mapping[node]
            for neighbor in node.neighbors:
                new_neighbor = mapping[neighbor]
                new_node.neighbors.append(new_neighbor)
        
        return mapping[root]
        
    def getNodes(self, node):
        q = collections.deque([node])
        result = set([node])
        while q:
            head = q.popleft()
            for neighbor in head.neighbors:
                if neighbor not in result: # prune
                    result.add(neighbor)
                    q.append(neighbor)
        return result

      
"""238. Product of Array Except Self"""      
def productExceptSelf(nums):
    # times twice: 1st - times every element before and then times every element after
    p = 1
    n = len(nums)
    output = []
    for i in range(n):
        output.append(p)
        p *= nums[i]
    p = 1
    for i in range(n-1,-1,-1):
        output[i] = output[i] * p
        p *= nums[i]
    return output

  
"""75. Sort 4 Colors"""
# [0,i) [i, j) [j, k) are 0s, 1s and 2s sorted in place for [0,k)
# suppose already sorted, now add a new 1 -> j,k both needs to add 1.
def solution(nums):
    # [0,i), [i,j), [j,k), [k:] - 0,1,2,3
    i = j = k = 0
    for ind in range(len(nums)):
        v = nums[ind]
        nums[ind] = 3
        if v < 3:
            nums[k] = 2
            k += 1
        if v < 2:
            nums[j] = 1
            j += 1
        if v == 0:
            nums[i] = 0
            i += 1
    return nums

  
"""41. First Missing Positive"""  
def swap(nums, i, j):
    temp = nums[i]
    nums[i] = nums[j]
    nums[j] = temp
def firstMissingPositive(nums):
    """If num > len(nums): there should be a number that's smaller than n that's missing"""
    i = 0
    while i < len(nums):
        # situations to ignore num
        if nums[i] == i+1 or nums[i] <= 0 or nums[i] > len(nums):
            i += 1
        # for a valid num, put nums[i] to its correct position: i.e. for num 5 swap it with num[4]
        elif nums[i] != nums[nums[i]-1]:
            swap(nums, i, nums[i]-1)
        else:
            i += 1
        
    for i in range(len(nums)):
        if nums[i] != i + 1:
            return i + 1
    return n + 1

  
"""151. Reverse Words in a String"""
# reverse the whole string and then reverse each word  
class solution:
    def reverseWords(self, s):
        """s should be a list"""
        # Reverse the whole string and then reverse word by word
        self.reverse(s, 0, len(s) - 1)

        beg = 0
        # by checking empty space
        for i in xrange(len(s)):
            if s[i] == ' ':
                self.reverse(s, beg, i-1)
                beg = i + 1
        # reverse the last word
        self.reverse(s, beg, len(s)-1)

    def reverse(self, s, start, end):
        while start < end:
            s[start], s[end] = s[end], s[start]
            start += 1
            end -= 1
            
            
"""227. Basic Calculator II"""
def calculate(s):
    # use stack to collect the result of each sign
    if not s:
        return "0"
    stack, num, sign = [], 0, "+"
    for i in xrange(len(s)):
        if s[i].isdigit(): # get number
            num = num*10+ord(s[i])-ord("0")
        # get signs
        if (not s[i].isdigit() and not s[i].isspace()) or i == len(s)-1:
            if sign == "-":
                stack.append(-num)
            elif sign == "+":
                stack.append(num)
            elif sign == "*":
                stack.append(stack.pop()*num)
            else: # /
                tmp = stack.pop()
                if tmp//num < 0 and tmp%num != 0:
                    stack.append(tmp//num+1)
                else:
                    stack.append(tmp//num)
            sign = s[i]
            num = 0
    return sum(stack)

  
"""Binary Tree Maximum Path Sum"""
# two situations: 1. node.val + one side; 2. node.val + 2 sides
class Solution(object):
    def maxPathSum(self, root):
        def dfs(node):  # returns: max one side path sum, max path sum
            l = r = 0
            ls = rs = None
            if node.left:
                l, ls = dfs(node.left)
                l = max(l, 0)
            if node.right:
                r, rs = dfs(node.right)
                r = max(r, 0)
            return node.val + max(l, r), max(node.val + l + r, ls, rs)
          
        if root:
            return dfs(root)[1]
        return 0

      
"""128. Longest Consecutive Sequence"""
# keep track of the sequence length and store that in the boundary points of the sequence
def longestConsecutive(nums):
    res = 0
    dic = {}
    for n in nums:
        if n not in dic.keys():
            # update keys
            left = dic.get(n-1, 0)
            right = dic.get(n+1, 0)
            total = left + right+ 1
            dic[n] = total
            # update max
            res = max(res, total)
            # extend length to boundaries
            dic[n-left] = total
            dic[n+right] = total
    return res

  
"""Find 5 consecutive 0s in a list"""
# every time when it's not 0, reset ct.
ct = 0
for i in elements:
    if i == 0:
        ct += 1
    else:
        ct = 0
        
"""161. One Edit Distance"""
# 3 possibilities: 1) equal length: replace one char; 2) remove one char from the longer string.
## only need to check the first char that's different!!
def isOneEditDistance(s, t):
    n = min(len(s), len(t))
    for i in range(n):
        if s[i] != t[i]:
            if len(s) == len(t): # situation 1) i.e. the rest part are the same
                return s[i+1:] == t[i+1:]
            elif len(s) > len(t): # i.e. remove one char
                return s[i+1:] == t[i:]
            else:
                return s[i:] == t[i+1:]
    # already has s[:n] == t[:n], now only possibility is one more char left
    return abs(len(s) - len(t)) == 1
  
# when it's about group together:
# one way is to use hashtap and then return dict.values().


"""88. Merge Sorted Array"""
# sort from the end to beginning
# use m,n as ct ~ how many un-sorted left
def merge(self, nums1, m, nums2, n):
        while m > 0 and n > 0:
            if nums1[m-1] >= nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m -= 1
            else:
                nums1[m+n-1] = nums2[n-1]
                n -= 1
        if n > 0:
            nums1[:n] = nums2[:n]
