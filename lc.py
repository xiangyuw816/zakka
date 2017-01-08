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
