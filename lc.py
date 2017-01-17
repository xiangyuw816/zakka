# notes about leetcode questions

## string: two pointers, binary search
## matrix: dfs
## permutations: dfs

# 448. Find All Numbers Disappeared in an Array
## use nums[i] as index to make nums[index] as negative
## filter those positive nums and get the index
class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        for i in range(len(nums)):
            ind = abs(nums[i])-1
            nums[ind] = -abs(nums[ind])
            
        return [i+1 for i in range(len(nums)) if nums[i]>0]
      
# 162. Find Peak Element
## binary search
class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        left =0
        right=len(nums)-1
        while left<right:
            mid = (left+right)/2
            # mid as the peak element
            if nums[mid]>nums[mid+1] and nums[mid]>nums[mid-1]:
                return mid
            # increasing: peak element should be in the right part
            if nums[mid]<nums[mid+1]:
                left=mid+1
            else:
                right=mid
        # always non-decreasing/increasing
        return left if nums[left]>nums[right] else right

# 388. Longest Absolute File Path
def lengthLongestPath(self, input):
    maxlen = 0
    path = []
    for line in input.splitlines():
        path[line.count('\t'):] = [line.lstrip('\t')]
        if '.' in line:
            maxlen = max(maxlen, sum(map(len, path)) + len(path) - 1)
    return maxlen

# 228. Summary Ranges
def summaryRanges(self, nums):
    ranges = []
    for n in nums:
        # increment >1: add new list
        if not ranges or n > ranges[-1][-1] + 1:
            ranges += [],
        ranges[-1][1:] = n,
    return ['->'.join(map(str, r)) for r in ranges]

# 163. missing ranges
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
# if len(d) > k, delete the character with smallest pos

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
        # enumerate(s) = (index, character)
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

# 298. Binary Tree Longest Consecutive Sequence
## BFS: queue - popleft then add children in the end
## tree: [element, # consevative length so far] - structure
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
                # if consevative: add 1
                l = length + 1 if child.val == node.val + 1 else 1
                dq.append([child, l])
    return ans

# 394. Decode String
## DFS, stack
# stack: ["string", num]
## if digit: num
## if     [: push new stack & then num=''
## if     ]: pop() & add them to [-1]
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
        # use dictionary to store 'dictionary'
        self.dic = collections.defaultdict(set)
        for s in dictionary:
            val = s
            if len(s) > 2:
                # deal with abbr
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
# use a dictionary to pair
## two pointers
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

# 281. Zigzag Iterator
class ZigzagIterator(object):
#[1,2,3],[4,5]
## use queue: popleft the whole v1 and append back if v1 not empty.
    def __init__(self, v1, v2):
        self.q = collections.deque([x[::-1] for x in [v1, v2] if x])
        # [[3,2,1],[5,4]]
    
    def hasNext(self):
        return len(self.q) > 0
    
    def next(self):
        temp = self.q.popleft()# [3,2,1]
        res = temp.pop()#[1]
        # if temp has elements left, append it back
        if temp: self.q.append(temp)
        return res
        

# Your ZigzagIterator object will be instantiated and called as such:
# i, v = ZigzagIterator(v1, v2), []
# while i.hasNext(): v.append(i.next())

# 78. Subsets
# Iteratively
## push into: each element + nums[i]
## 0: []
## 1: [], [1]
## 2: [], [1], [ + 2], [1 + 2]
def subsets(self, nums):
    res = [[]]
    for num in sorted(nums):
        res += [item+[num] for item in res]
    return res

# 100. Same Tree
## do this recursively
### if node equal, then isSameTree(left) and isSameTree(right)
class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if p and q:
            return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        return p is q 

# 347. Top K Frequent Elements
import collections
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        # Use Counter to extract the top k frequent elements
        # most_common(k) return a list of tuples, where the first item of the tuple is the element,
        # and the second item of the tuple is the count
        # Thus, the built-in zip function could be used to extract the first item from the tuples
        return zip(*collections.Counter(nums).most_common(k))[0]
    
# 121. Best Time to Buy and Sell Stock
# maxCur: 
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        maxCur, maxSoFar=0,0
        for i in xrange(1,len(prices)):
            ### add to itself!!
            maxCur += prices[i]-prices[i-1]
            maxCur = max(0, maxCur)
            maxSoFar = max(maxCur, maxSoFar)
        return maxSoFar

# string -- all substring with length k (??backtracking)

# 332. Reconstruct Itinerary
## dictionary {'depart': arrival s}
## use dfs to try every sub itinerary
def findItinerary(self, tickets):
    d = defaultdict(list)
    for flight in tickets:
        d[flight[0]] += flight[1],
    self.route = ["JFK"]
    
    def dfs(start = 'JFK'):
        # used all flights
        if len(self.route) == len(tickets) + 1:
            return self.route
        # sort all arrivals
        myDsts = sorted(d[start])
        ## first append route, if not worked: pop route & add the flight back
        for dst in myDsts:
            d[start].remove(dst)
            self.route += dst,
            worked = dfs(dst)
            if worked:
                return worked
            self.route.pop()
            d[start] += dst,
    return dfs()

# 34. Search for a Range
## divide and conquer
# write a sub function to search between (lo, hi) --> recursively call this function
def searchRange(self, nums, target):
    def search(lo, hi):
        if nums[lo] == target == nums[hi]:
            return [lo, hi]# all elements are the target
        if nums[lo] <= target <= nums[hi]:
            mid = (lo + hi) / 2
            l, r = search(lo, mid), search(mid+1, hi)
            # [-1,2]+[3,4] = [-1,2,3,4]
            # -1 in l+r: one half does not contain target
            # else:      both contain, so union the two.
            return max(l, r) if -1 in l+r else [l[0], r[1]]
        return [-1, -1]# out of range
    return search(0, len(nums)-1)

# 412. Fizz Buzz
## if i%3: 'Fizz'
def fizzBuzz(self, n):
    return ['Fizz' * (not i % 3) + 'Buzz' * (not i % 5) or str(i) for i in range(1, n+1)]

# 242. Valid Anagram
## compare sorted()
## or use hashtable to store 'char': # ocurrances
def isAnagram1(self, s, t):
    dic1, dic2 = {}, {}
    for item in s:
        # return the vaule of key: dic.get(key, default return)
        dic1[item] = dic1.get(item, 0) + 1
    for item in t:
        dic2[item] = dic2.get(item, 0) + 1
    return dic1 == dic2

# 373. Find K Pairs with Smallest Sums
#### deal with certain length
## already sorted 2 lists
## use BFS to explore all elements, visited{} to store visted elements
## pop the smallest and append it to the result
class Solution(object):
    def kSmallestPairs(self, nums1, nums2, k):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :type k: int
        :rtype: List[List[int]]
        """
        import heapq
        ret = []
        if len(nums1) * len(nums2) > 0:
            # queue = [(sum, (loc,index))]
            queue = [(nums1[0] + nums2[0], (0, 0))]
            # dictionary to skip visited vertices
            visited = {}
            while len(ret) < k and queue:
                # heappop(): pop and return the smallest item
                _, (i, j) = heapq.heappop(queue)
                ret.append((nums1[i], nums2[j]))
                # add two new elements into the queue. (2 if's below)
                if j + 1 < len(nums2) and (i, j + 1) not in visited:
                        heapq.heappush(queue, (nums1[i] + nums2[j + 1], (i, j + 1)))
                        visited[(i, j + 1)] = 1
                if i + 1 < len(nums1) and (i + 1, j) not in visited:
                        heapq.heappush(queue, (nums1[i + 1] + nums2[j], (i + 1, j)))
                        visited[(i + 1, j)] = 1
        return ret

# 425. Word Squares
## try everyword for the 1st row e.g. ball
## then try fitting 2nd row starting with e.g. str1[2]
## then fit 3rd row str1[3] str2[2]
def wordSquares(self, words):
    n = len(words[0])
    # {'b': ['ball','back']}
    fulls = collections.defaultdict(list)
    for word in words:
        for i in range(n):
            # e.g. back
            # {'':'back','b': ['back'],...,'back':['back']}
            fulls[word[:i]].append(word)
    
    def build(square):
        if len(square) == n:
            squares.append(square)
            return
        # zip(*[(1,4),(2,5),(3,6)]) = [(1,2,3),(4,5,6)]
        for word in fulls[''.join(zip(*square)[len(square)])]:
            build(square + [word])
    squares = []
    for word in words:
        build([word])
    return squares

# 422. Valid Word Square
## transpose
def validWordSquare(self, words):
    return map(None, *words) == map(None, *map(None, *words))

# 359. Logger Rate Limiter
## dictionary {'message': fastest timestamp to print ~ e.g. print time+10}
class Logger(object):

    def __init__(self):
        self.ok = {}

    def shouldPrintMessage(self, timestamp, message):
        if timestamp < self.ok.get(message, 0):
            return False
        self.ok[message] = timestamp + 10
        return True
    
# ???284. Peeking Iterator
class PeekingIterator(object):
    def __init__(self, iterator):
        self.iter = iterator
        self.temp = self.iter.next() if self.iter.hasNext() else None
    # next iterator's next
    def peek(self):
        return self.temp
    # next iterator
    def next(self):
        ret = self.temp
        self.temp = self.iter.next() if self.iter.hasNext() else None
        return ret

    def hasNext(self):
        return self.temp is not None

# 341. Flatten Nested List Iterator??
## hasNext check the top element in the stack and iterate it
### if is integer: return it
##           list: pop & iterate all elements and push into stack
## stack (list, # elements) pair
class NestedIterator(object):
# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())
    def __init__(self, nestedList):
        self.stack = [[nestedList, 0]]

    def next(self):
        self.hasNext()
        nestedList, i = self.stack[-1]
        self.stack[-1][1] += 1
        return nestedList[i].getInteger()
            
    def hasNext(self):
        s = self.stack
        while s:
            nestedList, i = s[-1]
            if i == len(nestedList):
                s.pop()
            else:
                x = nestedList[i]
                if x.isInteger():
                    return True
                s[-1][1] += 1
                s.append([x.getList(), 0])
        return False
    
# 297. Serialize and Deserialize Binary Tree
## pre-order
class Codec:

    def serialize(self, root):
        # write sub function to append
        def doit(node):
            if node:
                vals.append(str(node.val))
                doit(node.left)
                doit(node.right)
            else:
                vals.append('#')
        vals = []
        doit(root)
        return ' '.join(vals)

    def deserialize(self, data):
        def doit():
            val = next(vals)
            if val == '#':
                return None
            node = TreeNode(int(val))
            node.left = doit()
            node.right = doit()
            return node
        vals = iter(data.split())
        return doit()

# 46. Permutations
# DFS
def permute(self, nums):
    res = []
    self.dfs(nums, [], res)
    return res
    
def dfs(self, nums, path, res):
    if not nums:
        res.append(path)
        # return # backtracking
    for i in xrange(len(nums)):
        self.dfs(nums[:i]+nums[i+1:], path+[nums[i]], res)
        
# 20. Valid Parentheses
## brackets must close in the right order
class Solution:
    # @return a boolean
    def isValid(self, s):
        stack = []
        dict = {"]":"[", "}":"{", ")":"("}
        for char in s:
            if char in dict.values():
                stack.append(char)
            elif char in dict.keys():
                if stack == [] or dict[char] != stack.pop():
                    return False
            else:
                return False
        return stack == []

#    338. Counting Bits
## dp as a subproblem
## dp[index] = dp[index - offset] + 1
class Solution(object):
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        offset, dp=1,[]
        dp.append(0)
        for i in xrange(1,num+1):
            if i==2*offset:
                offset*=2
            dp.append(dp[i-offset]+1)
        return dp

# 252. Meeting Rooms 
# sort first
    def canAttendMeetings(self, intervals):
    intervals.sort(key=lambda x: x.start)
    
    for i in range(1, len(intervals)):
        if intervals[i].start < intervals[i-1].end:
            return False
        
    return True

# 399. Evaluate Division
## (A/B)*(B/C)*(C/D) is like the path A->B->C->D
## use double index dictionary to create values
## use permutations to add the val indirectly.
def calcEquation(self, equations, values, queries):
    quot = collections.defaultdict(dict)
    # zip(equations, values): [(['a', 'b'], 2.0), (['b', 'c'], 3.0)]
    for (num, den), val in zip(equations, values):
        quot[num][num] = quot[den][den] = 1.0
        quot[num][den] = val
        quot[den][num] = 1 / val
    # permutations('ABCD', 2): A_4_2; combinations('ABCD', 2): C_4_2.
    # (a,c,b), (a,b,c)...
    for k, i, j in itertools.permutations(quot, 3):
        # add new path
        if k in quot[i] and j in quot[k]:
            quot[i][j] = quot[i][k] * quot[k][j]
    # answer does not exist: -1.0
    return [quot[num].get(den, -1.0) for num, den in queries]

# 165. Compare Version Numbers
## e.g. v1=1.2.3 v2=2.1
## write a for loop to compare from the 1st level to max level
## gap: the difference between corresponding version level
class Solution:
    def compareVersion(self, version1, version2):
        v1 = version1.split('.')
        v2 = version2.split('.')
        for i in range(max(len(v1), len(v2))):
            gap = (int(v1[i]) if i < len(v1) else 0) - (int(v2[i]) if i < len(v2) else 0)
            if gap != 0:
                return 1 if gap > 0 else -1
        return 0

# 482. License Key Formatting
# format: (len % K) - K - ... - K
## the same length --> better use while loop (no need to i*K)
class Solution(object):
    def licenseKeyFormatting(self, S, K):
        """
        :type S: str
        :type K: int
        :rtype: str
        """
        S = S.upper().replace('-','')
        size = len(S)
        s1 = K if size%K==0 else size%K
        res = S[:s1]
        while s1<size:
            res += '-'+S[s1:s1+K]
            s1 += K
        return res

# 418. Sentence Screen Fitting
## dynamic programming - find the sub problem --> no need to calculate more repeated cases.
# nextIndex: start from index i -> get the index at the end of the line
# time: start from index i -> the number of times we fit a whole sentence.
# res += time[start]
class Solution(object):
    def wordsTyping(self, sentence, rows, cols):
        """
        :type sentence: List[str]
        :type rows: int
        :type cols: int
        :rtype: int
        """
        nextIndex,time=[],[]
        for i in range(len(sentence)):
            cur,t=i,0
            curLen=0
            while curLen+len(sentence[cur])<=cols:
                curLen+=len(sentence[cur])+1
                cur+=1
                # end of the sentence: time++ & restart from beginning
                if cur==len(sentence):
                    cur=0
                    t+=1
            nextIndex.append(cur)
            time.append(t)
        
        cur,res=0,0    
        for i in range(rows):
            res+=time[cur]
            cur=nextIndex[cur]
        return cur

# 200. Number of Islands
# 289. Game of Life
# 351. Android Unlock Patterns
# 351. Android Unlock Patterns???

# 286. Walls and Gates
## BFS with matrix
## find a gate --> append its surroding with val=1
##               --> inside this queue: ### prune: out of range & dist already smaller
##                      -->   each val=min(val, new added from gate)
##                      --> extend its surroding with val+1
def wallsAndGates(self, rooms):
    if not rooms:
        return 
    r, c= len(rooms), len(rooms[0])
    for i in xrange(r):
        for j in xrange(c):
            if rooms[i][j] == 0:
                queue = collections.deque([(i+1, j, 1), (i-1, j, 1), (i, j+1, 1), (i, j-1, 1)])
                while queue:
                    x, y, val = queue.popleft()
                    # pruning
                    if x < 0 or x >= r or y < 0 or y >= c or rooms[x][y] <= val:
                        continue
                    rooms[x][y] = val
                    queue.extend([(x+1, y, val+1), (x-1, y, val+1), (x, y+1, val+1), (x, y-1, val+1)])

# 320. Generalized Abbreviation
## current element abbreviated or not: so total # of permutation = 2^n
## DFS: (deep) replace 0th element, then replace pos+1 from the previous one --> until no more elements can be replaced.
## BFS: (breadth) replace 0th, then 1st, 2nd...

# 345. Reverse Vowels of a String
## two pointers
class Solution(object):
    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """
        vowel = 'AEIOUaeiou'
        s = list(s)
        i,j = 0, len(s)-1
        while i<j:
            while s[i] not in vowel and i<j:
                i = i + 1
            while s[j] not in vowel and i<j:
                j = j - 1
            s[i], s[j] = s[j], s[i]
            i, j = i + 1, j - 1
        return ''.join(s)
    
# 247. Strobogrammatic Number II
## observations
## n%2: oddCandidate inside findStrobogrammatic(n-1)
## else: evenCandidate inside findStrobogrammatic(n-2)
class Solution(object):
    def findStrobogrammatic(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        evenCandidate=['11','88','00','69','96']
        oddCandidate=['1','8','0']
        if n==1:
            return oddCandidate
        if n==2:
            return ['11','88','69','96']
        if n%2:
            pre, midCandidate=self.findStrobogrammatic(n-1),oddCandidate
        else:
            pre, midCandidate=self.findStrobogrammatic(n-2),evenCandidate
        mid=(n-1)/2
        return [p[:mid]+c+p[mid:] for p in pre for c in midCandidate]
    
# 294. Flip Game II
## memoization: dictionary
## TRUE: has a '++' and after flip to '--', the new string is False.
## at least one such situation exist --> thus use any()
class Solution(object):
    def canWin(self, s):
        memo = {}
        def can(s):
            if s not in memo:
                # any(): return True when at least has one True
                memo[s] = any(s[i:i+2] == '++' and not can(s[:i] + '-' + s[i+2:])
                              for i in range(len(s)))
            return memo[s]
        return can(s)

# 389. Find the Difference
## dictionary: {'char': # occurances}
## 1st string: add one; 2nd string: minus one --> return char with 0 occurance.
class Solution(object):
    """
    dictionary
    """
    def findTheDifference(self, s, t):
        dic = {}
        for ch in s:
            dic[ch] = dic.get(ch, 0) + 1
        for ch in t:
            if dic.get(ch, 0) == 0:
                return ch
            else:
                dic[ch] -= 1

# 360. Sort Transformed Array
## a>0: max should be in either side
## a<0: min should be in either side
## use two pointers!! to compare.

# 276. Paint Fence
## n==1: k
## n==2: same: k*1; diff: k*(k-1)
####### think as to print same/diff --> consistent
## n==3: given pre same: k*1*(k-1) - diff
##       given pre diff: k*(k-1) *(1 + k-1) - both same&diff
## ... to print same: pre.diff
## ... to print diff: (pre.diff+pre.same)*(k-1)
    if n == 0:
        return 0
    if n == 1:
        return k
    same, dif = k, k*(k-1)
    for i in range(3, n+1):
        same, dif = dif, (same+dif)*(k-1)
    return same + dif
