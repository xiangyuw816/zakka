# Particle filter
# http://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/

"""Huffman coding"""
## 1. sort symbols by their frequency (freq table)
## 2. select 2 symbols with least freq (f1, f2) to create 2 nodes: S1, S2
## 3. combine S1 and S2 to create S12/freq(f1+f2)
## 4. remove S1, S2 to create S12 in freq table and sorted
## 5. repeat step2-4 until only one char left --> a binary tree
## 6. assign value 0 to left sub-tree and 1 to right sub-tree
## 7. traver the tree to get symbols encoding i.e. most common symbols are using less bits to encode

"""146. LRU Cache"""

"""Toplogical sorting"""
# Description: In Docker, building an image has dependencies. An image can only be built once its dependency is built
(If the dependency is from outside, then the image can be built immediately). Sometimes, engineers make mistakes by 
forming a cycle dependency of local images. In this case, ignore the cycle and all the images depending on this cycle.
Input is vector of pair of images (image, its dependency). Output the order of images to be built in order. 
##Example: 
``` Example 1: {{"master", "ubuntu"}, {"numpy", "master"}, {"tensorflow", "numpy"}} 
Output: master, numpy, tensorflow 
Example 2: {{"python", "numpy"}, {"numpy", "python"}, {"tensorflow", "ubuntu"}} 
Output: tensorflow 
Example 3: {{"b", "c"}, {"c", "d"}, {"a", "b"}, {"d", "e"}, {"e","c"}, {"f", "g"}} Ouput: f ``` 

"""44. Wildcard Matching"""
class Solution:
# ?: match a single char; *: match anything
    def isMatch(self, s, p):
        s_cur = 0;
        p_cur= 0;
        match = 0;
        star = -1;
        while s_cur<len(s):
            if p_cur< len(p) and (s[s_cur]==p[p_cur] or p[p_cur]=='?'):
                s_cur = s_cur + 1
                p_cur = p_cur + 1
            elif p_cur<len(p) and p[p_cur]=='*':
                match = s_cur
                star = p_cur
                p_cur = p_cur+1
            elif (star!=-1):
                p_cur = star+1
                match = match+1
                s_cur = match
            else:
                return False
        while p_cur<len(p) and p[p_cur]=='*':
            p_cur = p_cur+1
             
        if p_cur==len(p):
            return True
        else:
            return False
                 
