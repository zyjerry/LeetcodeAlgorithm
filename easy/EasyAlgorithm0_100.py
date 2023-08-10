"""
    力扣算法题Python实践：https://leetcode.cn/problemset/algorithms/，可用于中学编程教学
    DATE        AUTHOR        CONTENTS
    2023-08-10  Jerry Chang   Create
"""

import math

class EasyAlgorithm:

    """    构造函数，什么都不做    """
    def __init__(self):
        print('Hello World!')


    """
    1. 两数之和：给定一个整数数组 nums 和一个整数目标值 target，
       请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
       你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
       你可以按任意顺序返回答案。
       进阶：你可以想出一个时间复杂度小于 O(n2) 的算法吗？
       标签：数组
       https://leetcode.cn/problems/two-sum/
    """
    def findTwoNumbers_1(self, nums=[1, 3, 5, 7, 9], target=10):
        i = 0
        j = 1
        flag = 0

        # 方法1：最简单的，双重循环，判断2个数之和是否等于目标数，如是，返回并终止循环，时间复杂度：O(n!)
        print('方法一：双重循环')
        for i in range(len(nums)):
            j = i + 1
            while j < len(nums):
                if ((nums[i] + nums[j]) == target):
                    print('nums[', i, ']+nums[', j, '] = ', nums[i], '+', nums[j], '=', target)
                    flag = 1
                j = j + 1
        if (flag == 0):
            print('数组中没有符合条件的两个数字之和为', nums)

        # 方法2：把数组放入HashMap中，key为数值，value为数组下标，循环n次判断两数差是否在keys中即可，时间复杂度O(n)
        print('方法二：哈希表')
        flag = 0
        numDict = {}
        for i in range(len(nums)):
            if (target - nums[i]) in numDict:
                print('nums[', i, ']+nums[', numDict[target - nums[i]], '] = ', nums[i], '+', target - nums[i], '=', target)
                flag = 1
            else:
                numDict[nums[i]] = i
        if (flag == 0):
            print('数组中没有符合条件的两个数字之和为', target)

        return 0


    """
    9. 回文数：给你一个整数 x ，如果 x 是一个回文整数，返回 true ；否则，返回 false 。
       回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。例如，121 是回文，而 123 不是。
       标签：数学
       进阶：你能不将整数转为字符串来解决这个问题吗？
    """
    def palindromicNumber_9(self, num):

        # 方法1：最笨办法，先把整数转成字符串，再一个一个翻转，最后比较，时间复杂度O(n)
        str1 = str(num)
        str2 = ''
        l = len(str1)
        for i in range(l):
            str2 = str2 + str1[l-i-1]
        if (str1==str2):
            print(str1,'是回文数。')
        else:
            print(str1,'不是回文数。')

        # 方法2：进阶，利用对数函数获取数字长度，再两头同时比较，时间复杂度O(n/2)
        l = math.ceil(math.log10(num))
        h = math.floor(l/2)
        flag = 0
        print('数字长度：', l, '数字长度的一半：', h)
        for i in range(h):
            leftnum = math.floor(num/(10**(l-i-1))) % 10
            rightnum = (math.floor((num/(10**i)))) % 10
            print('leftnum:', leftnum, 'rightnum', rightnum)
            if (leftnum != rightnum):
                flag = 1
                print(num, '不是回文数。')
                break
        if (flag == 0):
            print(num, '是回文数。')

        return 0


    """
    13. 罗马数字转整数：https://leetcode.cn/problems/roman-to-integer/
    """
    def romanToInteger_13(self, romanstr):

        return 0


    """
    14. 最长公共前缀：编写一个函数来查找字符串数组中的最长公共前缀。如果不存在公共前缀，返回空字符串 ""。
        https://leetcode.cn/problems/longest-common-prefix/
        标签：字符串，字典树
    """
    def longestCommonPrefix_14(self, strList=[]):

        str = ''
        # 先获取字符串中长度最小的那个长度
        minLen = len(strList[0])
        for i in range(len(strList)):
            if (minLen > len(strList[i])):
                minLen = len(strList[i])

        # 方法1：从最前边开始，双重循环比较
        tempStr = ''
        for i in range(minLen):
            flag = 0
            tempStr = strList[0][i]
            for j in range(len(strList)):
                if (tempStr != strList[j][i]):
                    flag = 1
                    break
            if (flag == 0):
                str = str + tempStr
            else:
                break
        print('最长公共前缀：', str)

        # 方法2：从最后面开始，双重循环比较
        j = minLen
        while (j >= 0):
            tempStr = strList[0][0:j]
            flag = 0
            for i in range(len(strList)):
                if (strList[i][0:j] != tempStr):
                    j = j-1
                    flag = 1
                    break
            if (flag == 0):
                break
        if (flag == 1):
            tempStr = ''
        print('最长公共前缀：', tempStr)

        return 0





if __name__ == "__main__":
    ea = EasyAlgorithm()
    #ea.findTwoNumbers_1([1, 3, 5, 7, 9], 10)
    #ea.findTwoNumbers_1([1, 3, 5, 8], 10)
    #ea.palindromicNumber_9(1234543)
    #ea.palindromicNumber_9(123456789)
    ea.longestCommonPrefix_14(["flower", "flow", "flight"])
    ea.longestCommonPrefix_14(["dog", "racecar", "car"])