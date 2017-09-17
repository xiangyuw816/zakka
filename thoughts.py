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
