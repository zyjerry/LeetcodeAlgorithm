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

    def findTwoNumbers_1(self, nums=[1, 3, 5, 7, 9], target=10) -> int:
        i = 0
        j = 1
        flag = 0

        # 方法1：最简单的，双重循环，判断2个数之和是否等于目标数，如是，返回并终止循环，时间复杂度：O(n!)
        print('方法一：双重循环')
        for i in range(len(nums)):
            j = i + 1
            while j < len(nums):
                if (nums[i] + nums[j]) == target:
                    print('nums[', i, ']+nums[', j, '] = ', nums[i], '+', nums[j], '=', target)
                    flag = 1
                j = j + 1
        if flag == 0:
            print('数组中没有符合条件的两个数字之和为', nums)

        # 方法2：把数组放入HashMap中，key为数值，value为数组下标，循环n次判断两数差是否在keys中即可，时间复杂度O(n)
        print('方法二：哈希表')
        flag = 0
        numDict = {}
        for i in range(len(nums)):
            if (target - nums[i]) in numDict:
                print('nums[', i, ']+nums[', numDict[target - nums[i]], '] = ', nums[i], '+', target - nums[i], '=',
                      target)
                flag = 1
            else:
                numDict[nums[i]] = i
        if flag == 0:
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
            str2 = str2 + str1[l - i - 1]
        if str1 == str2:
            print(str1, '是回文数。')
        else:
            print(str1, '不是回文数。')

        # 方法2：进阶，利用对数函数获取数字长度，再两头同时比较，时间复杂度O(n/2)
        l = math.ceil(math.log10(num))
        h = math.floor(l / 2)
        flag = 0
        print('数字长度：', l, '数字长度的一半：', h)
        for i in range(h):
            leftnum = math.floor(num / (10 ** (l - i - 1))) % 10
            rightnum = (math.floor((num / (10 ** i)))) % 10
            print('leftnum:', leftnum, 'rightnum', rightnum)
            if leftnum != rightnum:
                flag = 1
                print(num, '不是回文数。')
                break
        if flag == 0:
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
            if minLen > len(strList[i]):
                minLen = len(strList[i])

        # 方法1：从最前边开始，双重循环比较
        tempStr = ''
        for i in range(minLen):
            flag = 0
            tempStr = strList[0][i]
            for j in range(len(strList)):
                if tempStr != strList[j][i]:
                    flag = 1
                    break
            if flag == 0:
                str = str + tempStr
            else:
                break
        print('最长公共前缀：', str)

        # 方法2：从最后面开始，双重循环比较
        j = minLen
        while j >= 0:
            tempStr = strList[0][0:j]
            flag = 0
            for i in range(len(strList)):
                if strList[i][0:j] != tempStr:
                    j = j - 1
                    flag = 1
                    break
            if flag == 0:
                break
        if flag == 1:
            tempStr = ''
        print('最长公共前缀：', tempStr)

        return 0

    """
        20. 有效的括号：给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串s，判断字符串是否有效。
            有效字符串需满足：左括号必须用相同类型的右括号闭合。左括号必须以正确的顺序闭合。每个右括号都有一个对应的相同类型的左括号。
            https://leetcode.cn/problems/valid-parentheses/
            标签：字符串，栈
    """

    def validParentheses_20(self, s=''):
        # 这是一个典型的栈思路，先进后出。
        # 将字符串中的每个字符按顺序压入栈，并跟前一个字符比较，如果成对，那么将这对括号从栈中压出。如果不成对，继续判断下一个字符。
        stackStr = ''
        for i in range(len(s)):
            stackStr = stackStr + s[i]
            if len(stackStr) >= 2:
                str = stackStr[len(stackStr) - 2:len(stackStr)]
                if str == '()' or str == '[]' or str == '{}':
                    stackStr = stackStr.replace(str, '')
        if stackStr == '':
            print('该字符串是有效的括号。')
        else:
            print('该字符串不是有效的括号。')

    """
        21. 合并两个有序链表：将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 
            标签：链表
            https://leetcode.cn/problems/merge-two-sorted-lists/
    """

    def mergeTwoSortedLists_21(self, l1=[], l2=[]):
        # 思路：两个指针分别指向两个列表，循环比较当前2个数字大小，小的压进新链表，指针向前
        finalList = []
        p1 = 0
        p2 = 0
        while p1 < len(l1) and p2 < len(l2):
            if l1[p1] <= l2[p2]:
                finalList.append(l1[p1])
                p1 = p1 + 1
            else:
                finalList.append(l2[p2])
                p2 = p2 + 1
        # 没比完的尾巴加入最终列表
        if p1 < len(l1):
            finalList = finalList + l1[p1:len(l1)]
        if p2 < len(l2):
            finalList = finalList + l2[p2:len(l2)]
        print(finalList)

    """
        26. 删除有序数组中的重复项：
            给你一个 升序排列 的数组 nums ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。
            元素的 相对顺序 应该保持 一致 。然后返回 nums 中唯一元素的个数。
            考虑 nums 的唯一元素的数量为 k ，你需要做以下事情确保你的题解可以被通过：
            更改数组 nums ，使 nums 的前 k 个元素包含唯一元素，并按照它们最初在 nums 中出现的顺序排列。nums 的其余元素与 nums 的大小不重要。
            标签：数组
            https://leetcode.cn/problems/remove-duplicates-from-sorted-array/
    """

    def rmDuplctsFromSortedArray_26(self, nums=[]):
        # 思路：循环逐步判断一个元素是否跟下一个元素相等，如果相等则删除该元素
        i = 0
        while i < len(nums) - 1:
            if nums[i] == nums[i + 1]:
                del nums[i]
            i = i + 1
        print('除重后数组个数为：', len(nums), '，内容为：', nums)

    """
        27. 移除元素：给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。
            不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。
            元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。
            标签：数组
            https://leetcode.cn/problems/remove-element/
    """

    def rmElement_27(self, nums=[], val=0):
        # 思路：循环逐步判断一个元素是否等于val，如果相等则删除该元素
        i = 0
        while i < len(nums):
            if nums[i] == val:
                del nums[i]
            else:
                i = i + 1
        print('删除后数组个数为：', len(nums), '，内容为：', nums)

    """
        28. 找出字符串中第一个匹配项的下标：
            给你两个字符串 haystack 和 needle ，请你在 haystack 字符串中找出 needle 字符串的第一个匹配项的下标（下标从 0 开始）。
            如果 needle 不是 haystack 的一部分，则返回  -1 。
            标签：字符串
            https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/
    """

    def findFirstString_28(self, haystack='', needle=''):
        # 思路：从头到尾匹配
        i = 0
        while i < len(haystack) - len(needle):
            if haystack[i:len(needle) - 1] == needle:
                break
            else:
                i = i + 1
        if i == len(haystack) - len(needle):
            print('没有匹配的：', -1)
        else:
            print('匹配的下标号为：', i)

    """
        35. 搜索插入位置：
            给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
            请必须使用时间复杂度为 O(log n) 的算法。
            标签：数组，二分查找
            https://leetcode.cn/problems/search-insert-position/
    """

    def searchPosition_35(self, nums=[], val=0):
        # 思路：简单粗暴点就从头循环到尾，但这个时间复杂度是O(n)，如果求时间复杂度O(log n)，那么只能使用二分查找法

        # 1、先判断是否头尾
        if val <= nums[0]:
            print(0)
        elif val >= nums[len(nums) - 1]:
            print(len(nums))
        else:
            # 2、二分查找法
            leftside = 0
            rightside = len(nums)
            i = int(len(nums) / 2)
            flag = 0
            while flag == 0:
                if nums[i - 1] <= val <= nums[i]:
                    flag = 1
                elif val < nums[i - 1]:
                    rightside = i - 1
                    i = int((rightside + leftside) / 2)
                elif nums[i] < val:
                    leftside = i
                    i = math.ceil((rightside + leftside) / 2)
            print(i)

    """
        58. 最后一个单词的长度：给你一个字符串 s，由若干单词组成，单词前后用一些空格字符隔开。返回字符串中 最后一个 单词的长度。
            单词 是指仅由字母组成、不包含任何空格字符的最大子字符串。
            标签：字符串
            https://leetcode.cn/problems/length-of-last-word/
    """

    def lengthOfLastWord_58(self, word=''):
        i = len(word) - 1
        while i > 0 and word[i] != ' ':
            i = i - 1
        if word[i] == ' ':
            i = i + 1
        print('字符串最后一个单词是', word[i:len(word)], '长度是', len(word) - i)


if __name__ == "__main__":
    ea = EasyAlgorithm()
    # ea.findTwoNumbers_1([1, 3, 5, 7, 9], 10)
    # ea.findTwoNumbers_1([1, 3, 5, 8], 10)
    # ea.palindromicNumber_9(1234543)
    # ea.palindromicNumber_9(123456789)
    # ea.longestCommonPrefix_14(["flower", "flow", "flight"])
    # ea.longestCommonPrefix_14(["dog", "racecar", "car"])
    # ea.validParentheses_20('{[]}[]()')
    # ea.mergeTwoSortedLists_21([1,3,4,9,10,11,21],[1,2,3,4,5,7])
    # ea.rmDuplctsFromSortedArray_26([-1, 0, 0, 3, 5, 7, 7, 8, 8, 9])
    # ea.rmElement_27([-1, 0, 0, 3, 5, 7, 7, 8, 8, 9], 7)
    # ea.searchPosition_35([-1, 0, 0, 3, 5, 7, 7, 8, 8, 9], 2)
    # ea.searchPosition_35([-1, 0, 0, 3, 5, 7, 7, 8, 8, 9], -1)
    # ea.searchPosition_35([-1, 0, 0, 3, 5, 7, 7, 8, 8, 9], 10)
    ea.lengthOfLastWord_58('Hello World')
