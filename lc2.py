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
