"""291. Word Pattern II"""
class solution():
    def wordPatternMatch(self, pattern, str):
        return self.dfs(pattern, str, {})

    def dfs(self, pattern, str, dict):
        # dict to store existing patterns {sub-pattern: sub-string}
        if len(pattern) == 0 and len(str) > 0:
            return False
        if len(pattern) == len(str) == 0:
            return True

        # try use pattern[0] to match with each str[:end]
        # reduce search range by len(str)-len(pattern)+2 i.e. the largest sub-string length
        for end in range(1, len(str)-len(pattern)+2): # i.e. only 1 char in pattern has > one char
            # try to add a new pattern
            if pattern[0] not in dict and str[:end] not in dict.values():
                dict[pattern[0]] = str[:end]
                if self.dfs(pattern[1:], str[end:], dict):
                    return True
                # back tracking: to remove
                del dict[pattern[0]]
            elif pattern[0] in dict and dict[pattern[0]] == str[:end]:
                if self.dfs(pattern[1:], str[end:], dict):
                    return True

        return False

    
"""Maximum Size Subarray Sum Equals k"""
# continuous sub-array
def maxSubArrayLen(nums, k):
    ans, acc = 0, 0               # answer and the accumulative value of nums
    mp = {0:-1}                 #key is acc value, and value is the index
    for i in xrange(len(nums)):
        acc += nums[i]
        if acc not in mp:
            mp[acc] = i
        if acc-k in mp:
            print(i, mp, i-mp[acc-k])
            ans = max(ans, i-mp[acc-k])
    return ans
