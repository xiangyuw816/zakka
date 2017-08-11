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
