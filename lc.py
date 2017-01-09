# notes about leetcode questions

# 448
class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # use nums[i] as index to make nums[index] as negative
        # filter those positive nums and get the index
        for i in range(len(nums)):
            ind = abs(nums[i])-1
            nums[ind] = -abs(nums[ind])
            
        return [i+1 for i in range(len(nums)) if nums[i]>0]
      
# 162
class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # binary search
        left =0
        right=len(nums)-1
        while left<right:
            mid = (left+right)/2
            if nums[mid]>nums[mid+1] and nums[mid]>nums[mid-1]:
                return mid
            if nums[mid]<nums[mid+1]:
                left=mid+1
            else:
                right=mid
        return left if nums[left]>nums[right] else right

# 388
def lengthLongestPath(self, input):
    maxlen = 0
    path = []
    for line in input.splitlines():
        path[line.count('\t'):] = [line.lstrip('\t')]
        if '.' in line:
            maxlen = max(maxlen, sum(map(len, path)) + len(path) - 1)
    return maxlen

# 228 Summary Ranges
def summaryRanges(self, nums):
    ranges = []
    for n in nums:
        if not ranges or n > ranges[-1][-1] + 1:
            ranges += [],
        ranges[-1][1:] = n,
    return ['->'.join(map(str, r)) for r in ranges]

# 163 missing ranges
class Solution(object):
    def findMissingRanges(self, nums, lower, upper):
        """
        :type nums: List[int]
        :type lower: int
        :type upper: int
        :rtype: List[str]
        """
        # nums[i]-nums[i-1]==2: add nums[i]-1
        # nums[i]-nums[i-1]>2: add nums[i-1]+1 -> nums[i]-1
        ## use pre to present nums[i-1]
        pre=lower-1
        nums.append(upper+1)
        res=[]
        for i in nums:
            if(i==pre+2):
                res.append(str(i-1))
            elif(i>pre+2):
                res.append(str(pre+1) + '->' + str(i-1))
            pre=i
        return res
    
##    280. Wiggle Sort
# i even: num[i]<=num[i-1]
# i odd: num[i]>=num[i-1]
## if i even, we have num[i-1]>=num[i-2]
### if num[i]<=num[i-1], pass
### if num[i]>num[i-1], swap and still have num[i-1]>=num[i-2]
## exclusive or: (i%2) num[i]>num[i-1] - swap
class Solution(object):
    def wiggleSort(self, nums):
        for i in xrange(1, len(nums)):
            if (i % 2) ^ (nums[i] > nums[i - 1]):
                nums[i], nums[i - 1] = nums[i - 1], nums[i]

## 259. 3Sum Smaller
# first sort
# i:0 ++, j:k-1 --, k: range(len), i<j
# After sorting, if (i, j, k) is a valid triple, then (i, j-1, k), ..., (i, i+1, k) are also valid triples. count= j-i 
def threeSumSmaller(self, nums, target):
    nums.sort()
    count = 0
    for k in range(len(nums)):
        i, j = 0, k - 1
        while i < j:
            if nums[i] + nums[j] + nums[k] < target:
                count += j - i
                i += 1
            else:
                j -= 1
    return count

# 340. Longest Substring with At Most K Distinct Characters
# sliding window
# use dictionary d to store each character and its rightmost position.
# if if len(d) > k, delete the character with smallest pos

## low: begin of the string
class Solution(object):
    def lengthOfLongestSubstringKDistinct(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        # Use dictionary d to keep track of (character, location) pair,
        # where location is the rightmost location that the character appears at
        d = {}
        low, ret = 0, 0
        for i, c in enumerate(s):
            d[c] = i
            if len(d) > k:
                low = min(d.values())
                del d[s[low]]
                low += 1
            ret = max(i - low + 1, ret)
        return ret

# 415. add strings
# use izip_longest to tuples: respective digits
# c: larger than 10 --> carry
from itertools import izip_longest
class Solution(object):
    def addStrings(self, num1, num2):
        res, c = "", 0
        for (x, y) in izip_longest(num1[::-1], num2[::-1], fillvalue='0'):
            s = (int(x) + int(y) + c)
            d, c = s % 10, int(s / 10)
            res = str(d) + res

        if c > 0: res = str(c) + res

        return res

#    298. Binary Tree Longest Consecutive Sequence
## BFS: queue - popleft then add children in the end
from collections import deque
def longestConsecutive(self, root):
    if not root:
        return 0
    ans, dq = 0, deque([[root, 1]])
    while dq:
        node, length = dq.popleft()
        ans = max(ans, length)
        for child in [node.left, node.right]:
            if child:
                l = length + 1 if child.val == node.val + 1 else 1
                dq.append([child, l])
    return ans

## DFS: stack - pop
def longestConsecutive(self, root):
    if not root:
        return 0
    ans, stack = 0, [[root, 1]]
    while stack:
        node, length = stack.pop()
        ans = max(ans, length)
        for child in [node.left, node.right]:
            if child:
                l = length + 1 if child.val == node.val + 1 else 1
                stack.append([child, l])
    return ans

# 394. Decode String
## DFS, stack
# stack: ["string", num]
## digit -> num, [ -> append new stack & num='', ] -> pop() & add them to [-1]
class Solution(object):
    def decodeString(self, s):
        stack = []
        # to store final result
        stack.append(["", 1])
        num = ""
        for ch in s:
            if ch.isdigit():
              num += ch
            elif ch == '[':
                stack.append(["", int(num)])
                num = ""
            elif ch == ']':
                st, k = stack.pop()
                # actually only one [] left, consider case "3[a2[c]]" --> X stack[0][0] (put cc into 3a s.t. accaccacc)
                stack[-1][0] += st*k
            else:
                stack[-1][0] += ch
        return stack[0][0]

# 288. Unique Word Abbreviation
## [('c2e', set(['cake','cage'])), ('r1d', set(['red']))]
class ValidWordAbbr(object):    
    def __init__(self, dictionary):
        self.dic = collections.defaultdict(set)
        for s in dictionary:
            val = s
            if len(s) > 2:
                s = s[0]+str(len(s)-2)+s[-1]
            self.dic[s].add(val)

    def isUnique(self, word):
        val = word 
        if len(word) > 2:
            word = word[0]+str(len(word)-2)+word[-1]
        # if word abbreviation not in the dictionary, or word itself in the dictionary (word itself may 
        # appear multiple times in the dictionary, so it's better using set instead of list)
        return len(self.dic[word]) == 0 or (len(self.dic[word]) == 1 and val == list(self.dic[word])[0])
    
# 246. Strobogrammatic Number
def isStrobogrammatic(self, num):
    dic = {"0":"0", "1":"1", "6":"9", "8":"8", "9":"6"}
    l, r = 0, len(num)-1
    while l <= r:
        # num itself 
        if num[l] not in dic or dic[num[l]] != num[r]:
            return False
        l += 1
        r -= 1
    return True

# 50. Pow(x, n)
## recursive - pow(x,n/2)
## remember base case
class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n<0:
            return 1/self.myPow(x,-n)
        elif n==0:
            return 1
        else:
            res=self.myPow(x,n/2)
            res*=res
            if n%2:
                res*=x
            return res
