"""
    力扣算法题Python实践：https://leetcode.cn/problemset/algorithms/，可用于中学编程教学
    DATE        AUTHOR        CONTENTS
    2023-08-23  Jerry Chang   Create
"""
import math
import re
from typing import Tuple


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

    def longestSubstrWithoutRepeatChars_3(self, s: str = '') -> Tuple[int, str]:
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

        return len(finalStr), finalStr

    """
        5. 最长回文子串：给你一个字符串 s，找到 s 中最长的回文子串。如果字符串的反序与原始字符串相同，则该字符串称为回文字符串。
            标签：字符串，动态规划
            https://leetcode.cn/problems/longest-palindromic-substring/
    """

    def longestPalindromicSubstr_5(self, s: str = '') -> str:
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
        # 个人觉得这个算法比力扣官方解答简洁一些，就是lst耗费点空间，不过受力扣官方解答启发，这个lst里的子串内容也可以不记，只记下标即可
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
        for j in lst:  # 对list中的每个元素判断
            # 如果前后2个字符相同，那么能构成新的更长的回文串，记录起止位置和内容，塞入列表中
            if j[0] != 0 and j[1] != len(s) - 1 and s[j[0] - 1] == s[j[1] + 1]:
                lll = [j[0] - 1, j[1] + 1, s[j[0] - 1:j[1] + 2]]
                lst.append(lll)
        print('最大回文串是：', lst[-1][2])

        return lst[-1][2]

    """
        6. N 字形变换：将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z 字形排列。
            比如输入字符串为 "PAYPALISHIRING"，
            行数为 3 时，排列如下：        行数为 4 时，排列如下：         行数为 5 时，排列如下：
            P   A   H   N               P     I     N                P       H
            A P L S I I G               A   L S   I G   +4           A     S I        +6
            Y   I   R                   Y A   H R       +2           Y   I   R        +4
                                        P     I                      P L     I G      +2
                                                                     A       N
            之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如行数为3时："PAHNAPLSIIGYIR"，行数为4时："PINALSIGYAHRPI"。
            标签：字符串
            https://leetcode.cn/problems/zigzag-conversion/
    """

    def zigzagConversion_6(self, s: str = '', numRows: int = 1) -> str:
        # 思路：先找规律，看看每一行字符所在原字符串中的下标序号有什么规律。
        # 当numRows=2时，第一行0，2，4，6……看作(2+0)*n，第二行1，3，5，7……看作(2+0)*n + (2-1)；
        # 当numRows=3时，第一行0，4，8，12……看作(3+1)*n,第三行2，6，10，14……看作(3+1)*n + (3-1)；
        # 当numRows=4时，第一行0，6，12……看作(4+2)*n；第四行3，9，15……看作(4+2)*n + (4-1)；
        # 当numRows=5时，第一行0，8，16……看作(5+3)*n，第五行4，12，20……看作(5+3)*n + (5-1)。

        # 综合规律看，当需要切割成numRows行时，排列的规律是，头尾两行，每行一个来回都只有一个字符：
        # 第1行下标：(numRows + numRows -2)*n = (numRows-1)*2n  #n>=0的自然数
        # 第numRows行下标：第1行下标+numRows-1，即：(numRows-1)*2n + numRows - 1  #n>=0的自然数

        # 头尾两行的序号确定后，再看中间部分第row行下标的规律，每一个来回，都包含2个字符：
        # 第1个字符：第1行的下标+(row-1) = (numRows-1)*2n + (row-1)；
        # 第2个字符：第一个字符的下标+(numRows-row)*2 = (numRows-1)*2n + (row-1) + (numRows-row)*2

        # 以下是算法实施步骤。
        # 1、初始化一个数组，该数组包含numRows个元素，每个元素包含每行的字符串
        finalLst = [''] * numRows
        # 2、初始化第一行数据和最后一行数据
        # 先看看能有几个来回
        bout = len(s) // ((numRows - 1) * 2) + 1
        # 再根据回合数写头尾两行字符
        for i in range(bout):
            finalLst[0] = finalLst[0] + s[(numRows - 1) * 2 * i]
            if ((numRows - 1) * 2 * i + numRows - 1) < len(s):
                finalLst[numRows - 1] = finalLst[numRows - 1] + s[(numRows - 1) * 2 * i + numRows - 1]
        # 3、初始化中间行的数据
        for i in range(1, numRows - 1):
            for j in range(bout):
                ind = (numRows - 1) * 2 * j + i
                if ind < len(s):
                    finalLst[i] = finalLst[i] + s[ind]
                ind = (numRows - 1) * 2 * j + i + (numRows - i - 1) * 2
                if ind < len(s):
                    finalLst[i] = finalLst[i] + s[ind]
        # 4、把这些list按顺序合并成一个字符串并打印
        print(''.join(finalLst))

        # 思路2来自力扣官方，看完瞬间觉得我傻了……
        # 它是初始化一个矩阵，再遍历一遍字符串，判断每个字符应当处于矩阵的什么位置（i和j坐标值），填入矩阵，最后把矩阵中的非空字符拼接起来
        # 再简化一点，初始化一个列表，再遍历一遍字符串，判断每个字符应当处于哪一行，追加到对应列表元素字符串的末尾
        # 仔细判断：一个来回需要的元素个数是中间行数*2 + 头尾两行各1个，(numRows-2)*2+2 = (numRows-1)*2
        # 实际上就是要把每个元素下标模(numRows-1)*2后，按到对应那一行，假若模之后的值为r，那么：
        # 如果r<numRows，该元素应当按到第r行，否则应当按到第numRows-(r-numRows)-2 = 2*numRows-r-2行
        finalLst = [''] * numRows
        for i in range(len(s)):
            r = i % ((numRows - 1) * 2)
            if r < numRows:
                finalLst[r] = finalLst[r] + s[i]
            else:
                finalLst[2 * numRows - r - 2] = finalLst[2 * numRows - r - 2] + s[i]
        print(''.join(finalLst))

        return ''.join(finalLst)

    """
        7. 整数反转：给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。
            如果反转后整数超过 32 位的有符号整数的范围 [−231,  231 − 1] ，就返回 0。假设环境不允许存储 64 位整数（有符号或无符号）。
            标签：数学
            https://leetcode.cn/problems/reverse-integer/
    """

    def reverseInteger_7(self, x: int = 0) -> int:
        # 思路：不断取模，根据位次乘以10的相应次方累加。
        # 这里需要注意的是，负数由于后台是以补码存储的，它取模的结果不是个位数，所以要分正负数不同情况处理，很不优雅
        n = abs(x)
        i = int(math.log10(abs(x)))
        finalx = 0
        while n != 0:
            print(finalx, n, i)
            finalx = finalx + (n % 10) * (10 ** i)
            n = n // 10
            i = i - 1
        if x < 0:
            finalx = -finalx
        print(finalx)
        return finalx

    """
        8. 字符串转换整数 (atoi)：
            请你来实现一个 myAtoi(string s) 函数，使其能将字符串转换成一个 32 位有符号整数（类似 C/C++ 中的 atoi 函数）。
            函数 myAtoi(string s) 的算法如下：
            1）读入字符串并丢弃无用的前导空格
            2）检查下一个字符（假设还未到字符末尾）为正还是负号，读取该字符（如果有）。 确定最终结果是负数还是正数。 如果两者都不存在，则假定结果为正。
            3）读入下一个字符，直到到达下一个非数字字符或到达输入的结尾。字符串的其余部分将被忽略。
            4）将前面步骤读入的这些数字转换为整数（即，"123" -> 123， "0032" -> 32）。如果没有读入数字，则整数为 0 。必要时更改符号（从步骤 2 开始）。
            5）如果整数数超过 32 位有符号整数范围 [−231,  231 − 1] ，需要截断这个整数，使其保持在这个范围内。
                具体来说，小于 −231 的整数应该被固定为 −231 ，大于 231 − 1 的整数应该被固定为 231 − 1 。
            6）返回整数作为最终结果。
            注意：本题中的空白字符只包括空格字符 ' ' ；除前导空格或数字后的其余字符串外，请勿忽略 任何其他字符。
            标签：字符串
            https://leetcode.cn/problems/string-to-integer-atoi/
    """

    def strToIntegerAtoi_8(self, s: str = '') -> int:
        # 1、先用正则表达式摒弃所有非+、-、数字的字符
        ss = re.sub('[^\d+-]', '', s)
        print(ss)

        # 2、摈弃所有最左边的+、-、0，保留记住+、-号
        i = 0
        sign = '+'
        sss = ''
        while ss[i] in ['+', '-', '0']:
            sss = ss[i + 1:]
            if ss[i] in ['+', '-']:
                sign = ss[i]
            i = i + 1

        # 3、再次摈弃所有非数字的字符
        sss = re.sub('[^\d]', '', sss)
        print(sss)

        # 4、判断长度是否超出32，若超出，按题意返回−231或230
        if len(sss) > 32:
            if sign == '+':
                print(320)
                return
            elif sign == '-':
                print(-321)
                return

        # 5、开始计算转换成数字
        i = 0
        finalInt = 0
        while i < len(sss):
            finalInt = finalInt * 10 + int(sss[i])
            i = i + 1

        # 6、加上正负号
        if sign == '-':
            finalInt = - finalInt

        # 返回结果
        print(finalInt)
        return finalInt

    """
        11. 盛最多水的容器：给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。
            找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。返回容器可以储存的最大水量。说明：你不能倾斜容器。
            标签：贪心，数组，双指针
            https://leetcode.cn/problems/container-with-most-water/
    """

    def containerWithMostWater_11(self, height: list = []) -> Tuple[int, int, int, int, int]:
        # 思路1、简单粗暴双重循环，判断并选出最大的容积，时间复杂度O(n**2)
        idx1, idx2, volume = 0, 0, 0
        for i in range(len(height) - 1):
            for j in range(i + 1, len(height)):
                if volume < (j - i) * min(height[i], height[j]):
                    idx1, idx2, volume = i, j, (j - i) * min(height[i], height[j])
        print(idx1, height[idx1], idx2, height[idx2], volume)
        # return idx1, height[idx1], idx2, height[idx2], volume

        # 思路2、参考力扣官方解答，头尾双指针，移动相对短的指针，判断最大容积，可以证明该方法能够获取最大容积，时间复杂度O(n)
        idx1, idx2, volume = 0, len(height) - 1, (len(height) - 1) * min(height[0], height[len(height) - 1])
        i, j = 0, len(height) - 1
        while i < j:
            if volume < (j - i) * min(height[i], height[j]):
                idx1, idx2, volume = i, j, (j - i) * min(height[i], height[j])
            if height[i] < height[j]:
                i = i + 1
            else:
                j = j - 1
        print(idx1, height[idx1], idx2, height[idx2], volume)
        return idx1, height[idx1], idx2, height[idx2], volume





if __name__ == "__main__":
    ma = MediumAlgorithm0_99()
    # ma.longestSubstrWithoutRepeatChars_3('abcabcdbb')
    # ma.longestPalindromicSubstr_5('aabcbaeeuiywpwiud')
    # ma.zigzagConversion_6('PAYPALISHIRING', 3)
    # ma.zigzagConversion_6('PAYPALISHIRING', 4)
    # ma.zigzagConversion_6('PAYPALISHIRING', 5)
    # ma.reverseInteger_7(-123)
    # ma.strToIntegerAtoi_8('-00ieur23857021+hfd-hg3456')
    ma.containerWithMostWater_11([1, 8, 6, 2, 5, 4, 8, 3, 7])
