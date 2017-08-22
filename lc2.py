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
class Solution(object):
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
                tempc = string[begin]##
                if tempc in still_need_dict.keys():
                    still_need_dict[tempc] += 1
                    if still_need_dict[tempc] > 0:
                        counter += 1 # modify count for different situations
                
                #### update result
                if end-begin == len(target): # different situations
                    result.append(begin)
    
                begin += 1
                
        return result

      
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

      
"""6. ZigZag Conversion"""
class Solution(object):
    def convert(self, s, numRows):
      # Concerner Case
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
