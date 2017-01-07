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
