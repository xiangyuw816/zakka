"""243. Shortest Word Distance"""
# Assume that words = ["practice", "makes", "perfect", "coding", "makes"]
# Given word1 = “coding”, word2 = “practice”, return 3
# iterate over words, every time when find an index match word1/word2, compare min(min, abs(i1-i2))


"""624. Maximum Distance in Arrays"""
# Already sorted! distance defined as |a-b|
# iterate over arrays: max_dist = max(max_dist, prev_array[0]-current_arry[-1], prev_array[-1]-current_arry[0]
def maxDistance(self, arrays):
    res, curMin, curMax = 0, arrays[0][0], arrays[0][-1]
    for a in arrays[1:]:
        res = max(res, a[-1]-curMin, curMax-a[0])
        curMin, curMax = min(curMin, a[0]), max(curMax, a[-1])
    return res


"""256. Paint House"""
# only need to maintain one list of costs that painted in three colors respectively.
def minCost(self, costs):
    prev = [0] * 3
    for now in costs:
        prev = [now[i] + min(prev[:i] + prev[i+1:]) for i in range(3)]
    return min(prev)


"""276. Paint Fence"""
# Rule: no more than two adjacent fence posts have the same color
## break into: if paint the same or paint diff
## same, dif = dif, (same+dif)*(k-1)


"""55. Jump Game"""
# iterate and update the max index that's reachable
def canJump(self, A):
    if not A:
        return False

    maxjump = 0 # at current index, the max jump we can make
    for i in range(len(A)):
        maxjump = max(maxjump-1,A[i]) # previous maxjump need to -1 because now we move forward one step
        if maxjump==0:
            break

    return i==len(A)-1

def jump(self, nums):
    # to find the min step to reach the final index
    # start, end as current range of reachable
    # then update to the fastest index can be reached in 1 step
    n, start, end, step = len(nums), 0, 0, 0
    while end < n - 1:
        step += 1
        maxend = end + 1
        for i in range(start, end + 1):
            if i + nums[i] >= n - 1:
                return step
            maxend = max(maxend, i + nums[i])
        start, end = end + 1, maxend
        
    return step
