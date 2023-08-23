"""
    力扣算法题Python实践：https://leetcode.cn/problemset/algorithms/，可用于中学编程教学
    DATE        AUTHOR        CONTENTS
    2023-08-23  Jerry Chang   Create
"""


class MediumAlgorithm0_99:
    """    构造函数，什么都不做    """

    def __init__(self):
        print('Hello World!')

    """
        3. 无重复字符的最长子串：给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
            示例 1:输入: s = "abcabcbb"，输出: 3 。解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
            标签：哈希表，字符串，滑动窗口
            https://leetcode.cn/problems/longest-substring-without-repeating-characters/
    """

    def longestSubstrWithoutRepeatChars_3(self, s):
        # 思路1：暴力循环，从2开始到字符串最长度，看看每个长度的每个子串是否有重复字符，该算法复杂度O(n!)
        finalStr = ''
        tmpStr = ''
        lst = []
        for i in range(1, len(s) + 1):  # i表示长度窗口
            for j in range(0, len(s) - i):  # j表示起始位置为j、长度为i取字符串
                tmpStr = s[j: j + i]
                lst = []
                for k in range(j, j + i):
                    if s[k] not in lst:
                        lst.append(s[k])
                    else:
                        tmpStr = ''
                        break
                if len(tmpStr) > len(finalStr):
                    finalStr = tmpStr
                # print(i, j, tmpStr, lst, finalStr)
        print(finalStr)

        # 思路2、滑动窗口：初始化左右2个指针分别为0和1，然后开始循环：判断该子串是否包含重复字符：
        # 如果包含：左指针+1，进行下一轮循环
        # 如果不包含：右指针+1，同时判断当前子串是否最长，如果是最长，相关值赋给final变量
        # 该算法力扣官方答案讲的不是很清楚，复杂度O(n)
        finalStr = s[0:1]
        finalLeft = 0
        finalRight = 1
        tmpLeft = 0
        tmpRight = 1
        while tmpRight < len(s):
            lst = []
            flag = True
            for i in range(tmpLeft, tmpRight + 1):
                if s[i] not in lst:
                    lst.append(s[i])
                else:
                    flag = False
            if flag:
                tmpRight = tmpRight + 1
                if (finalRight - finalLeft) < (tmpRight - tmpLeft):
                    finalStr = s[tmpLeft:tmpRight]
                    finalLeft = tmpLeft
                    finalRight = tmpRight
            else:
                tmpLeft = tmpLeft + 1
        print(finalStr)

    """
        5. 最长回文子串：给你一个字符串 s，找到 s 中最长的回文子串。如果字符串的反序与原始字符串相同，则该字符串称为回文字符串。
            标签：字符串，动态规划
            https://leetcode.cn/problems/longest-palindromic-substring/
    """

    def longestPalindromicSubstr_5(self, s):
        # 思路1、暴力循环：把每种子串拿出来判断是否回文，取最大的那个。时间复杂度大约O(n**2)
        finalStr = ''
        for i in range(0, len(s) - 1):  # 子串起始位置
            for j in range(i + 2, len(s)):  # 子串终止位置
                tmpStr = s[i:j]
                flag = True
                for k in range(0, (j - i) // 2):
                    if tmpStr[k] != tmpStr[j - i - k - 1]:
                        flag = False
                if flag:
                    if len(finalStr) < len(tmpStr):
                        finalStr = tmpStr
        print('最大回文串是：', finalStr)

        # 思路2、动态规划：假设一个字母a，它前面和后面的字母相同，才构成一个回文串。
        # 也就是说，一个回文串，它前面和后面的字母相同，才构成一个更长的新的回文串。
        # 我们初始化一个list，这个list包含s[1]~s[len-2]的所有单个字符，和，2个回文的子串，再逐个对这个list的每个元素判断：
        # 如果它前后字母都相同，那就构成一个新的更长的回文，把这个回文追加入list中；
        # 如果它前后字母不同，那就说明这个字串不能再扩展构成新的更长的回文，不做任何操作。
        # 这样下来list遍历完毕，其最后一个元素就是最长的回文串，时间复杂度大约O(n)
        # 个人觉得这个算法比力扣官方解答简洁一些，就是lst耗费点空间，不过受力扣官方解答启发，这个lst里的子串内容也可以不记
        lst = []
        for i in range(1, len(s) - 1):  # 初始化list
            l = [i, i, s[i]]  # 每个元素也是一个列表，分别记录字串起止位置、子串内容
            lst.append(l)
            if i < (len(s) - 1) and s[i] == s[i + 1]:  # 如果连续的2个字符相同，也构成回文，塞入初始化列表中
                ll = [i, i + 1, s[i:i + 2]]
                lst.append(ll)
            if i == 1 and s[i - 1] == s[i]:  # 如果连续的2个字符相同，也构成回文，塞入初始化列表中
                ll = [i - 1, i, s[i - 1:i + 1]]
                lst.append(ll)
        # print(lst)
        for j in lst:  # 对list中的每个元素判断
            # 如果前后2个字符相同，那么能构成新的更长的回文串，记录起止位置和内容，塞入列表中
            if j[0] != 0 and j[1] != len(s) - 1 and s[j[0] - 1] == s[j[1] + 1]:
                lll = [j[0] - 1, j[1] + 1, s[j[0] - 1:j[1] + 2]]
                lst.append(lll)
        # print(lst)
        print('最大回文串是：', lst[-1][2])


if __name__ == "__main__":
    ma = MediumAlgorithm0_99()
    # ma.longestSubstrWithoutRepeatChars_3('abcabcdbb')
    ma.longestPalindromicSubstr_5('aabcbaeeuiywpwiud')
