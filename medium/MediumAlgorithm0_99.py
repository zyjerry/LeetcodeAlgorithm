"""
    力扣算法题Python实践：https://leetcode.cn/problemset/algorithms/，可用于中学编程教学
    DATE        AUTHOR        CONTENTS
    2023-08-23  Jerry Chang   Create
"""
import math
import re
from idlelib.pyshell import restart_line
from typing import Tuple
import operator


class MediumAlgorithm0_99:
    """    构造函数，什么都不做    """

    def __init__(self):
        print('Hello World!')

    """
        2. 两数相加：给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。
           请你将两个数相加，并以相同形式返回一个表示和的链表。你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

            示例 1：输入：l1 = [2,4,3], l2 = [5,6,4]，输出：[7,0,8]，解释：342 + 465 = 807.
            示例 2：输入：l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]，输出：[8,9,9,9,0,0,0,1]
            标签：递归，链表，数学
            https://leetcode.cn/problems/add-two-numbers/description/
    """

    def AddTwoNumbers_2(self, num1: list = [], num2: list = []) -> list :
        # 思路：从左向右，把对应位置上的数字相加，如果有进位，留到下一轮加入
        result = []
        carrynumber = 0
        a = 0
        b = 0
        for i in range(0,max(len(num1),len(num2))):
            if i<len(num1):
                a = num1[i]
            else:
                a = 0
            if i<len(num2):
                b = num2[i]
            else:
                b = 0
            number = (a + b + carrynumber)%10
            result.append(number)
            carrynumber =  (a + b + carrynumber)//10
        if carrynumber > 0:
            result.append(carrynumber)
        print(result)
        return result


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
        ss = re.sub('[^\\d+-]', '', s)
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
        sss = re.sub('[^\\d]', '', sss)
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

    """
        15. 三数之和：给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足：
            i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。
            请你返回所有和为 0 且不重复的三元组。注意：答案中不可以包含重复的三元组。
            标签：数组，双指针，排序
            https://leetcode.cn/problems/3sum/
    """

    def sum3_15(self, nums: list = []) -> set:
        # 思路1、简单粗暴三重循环判断，由于要求答案中不可以包含重复的三元组，所以这里存储答案使用集合（它不会有重复数据）
        s = set()
        for i in range(len(nums) - 2):
            for j in range(i + 1, len(nums)):
                for k in range(j + 1, len(nums)):
                    if nums[i] + nums[j] + nums[k] == 0:
                        print('nums[', i, ']+nums[', j, ']+nums[', k, '] = ', nums[i], '+', nums[j], '+', nums[k], '=0')
                        l = [nums[i], nums[j], nums[k]]
                        l.sort()
                        s.add(str(l))
        print(s)
        # return s

        # 思路2、力扣官方解答：首先将数组排序，然后依然做三重循环。
        # 但是有一些技巧，为确保不包含重复的三元组，在二重和三重循环中，可以跳过和当前同样的数字；
        # 同时在末尾增加指针，随着前两重循环将数字向右推，末尾指针也可以向左推，以减少循环次数。
        nums.sort()
        print(nums)
        l = []
        s = set()
        for i in range(len(nums) - 2):
            for j in range(i + 1, len(nums) - 1):
                if j == i + 1 or (j > i + 1 and nums[j] != nums[j - 1]):
                    k = j + 1
                    rightidx = len(nums)
                    while k < rightidx:
                        if (k == j + 1 or (k > j + 1 and nums[k] != nums[k - 1])) and nums[i] + nums[j] + nums[k] == 0:
                            s.add(str([nums[i], nums[j], nums[k]]))
                            rightidx = k
                        print(i, j, k, nums[i], nums[j], nums[k], rightidx)
                        k = k + 1
        print(s)
        return s

    """
        16. 最接近的三数之和：给你一个长度为 n 的整数数组 nums 和 一个目标值 target。
            请你从 nums 中选出三个整数，使它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在恰好一个解。
            标签：数组，双指针，排序
            https://leetcode.cn/problems/3sum-closest/
    """

    def sum3Closest_16(self, nums: list = [], target: int = 0) -> Tuple[int, int, int]:
        # 思路：该题跟上一题思路一致，只是把条件从三数和为0变成了三数和与某个数最接近
        nums.sort()
        print(nums)
        mindiff = 1000
        for i in range(len(nums) - 2):
            for j in range(i + 1, len(nums) - 1):
                k = j + 1
                rightidx = len(nums)
                while k < rightidx:
                    if nums[i] + nums[j] + nums[k] - target > 0:
                        rightidx = k
                    if abs(nums[i] + nums[j] + nums[k] - target) < mindiff:
                        mindiff = abs(nums[i] + nums[j] + nums[k] - target)
                        mini = i
                        minj = j
                        mink = k
                    k = k + 1
        print(mini, minj, mink, nums[mini] + nums[minj] + nums[mink])
        # return mini, minj, mink, nums[mini] + nums[minj] + nums[mink]

        # 优化思路：两重循环，对于每一个i，找到距离最接近target的j和k，j、k同时向中间靠拢，就又比上一个思路减少一重循环
        mindiff = 1000
        for i in range(len(nums) - 2):
            j = i + 1
            k = len(nums) - 1
            while j < k:
                if abs(nums[i] + nums[j] + nums[k] - target) < mindiff:
                    mindiff = abs(nums[i] + nums[j] + nums[k] - target)
                    mini = i
                    minj = j
                    mink = k
                if nums[i] + nums[j] + nums[k] - target > 0:
                    k = k - 1
                elif nums[i] + nums[j] + nums[k] - target < 0:
                    j = j + 1
                else:
                    break
            if mindiff == 0:
                break
        print(mini, minj, mink, nums[mini] + nums[minj] + nums[mink])
        return mini, minj, mink, nums[mini] + nums[minj] + nums[mink]

    """
    17. 电话号码的字母组合：给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。答案可以按 任意顺序 返回。
        给出数字到字母的映射如下（与电话9键按键相同）。注意 1 不对应任何字母。
        示例 1：输入：digits = "23"，输出：["ad","ae","af","bd","be","bf","cd","ce","cf"]
        标签：哈希表，字符串，回溯
        https://leetcode.cn/problems/letter-combinations-of-a-phone-number/
    """

    def tel2monogram_17(self, digits = '') -> list:
        # 思路：可以用递归的方式，
        # 这里简单点，就用3个循环，关键是把中间已经组合字符串列表带入下个数字的判断组合

        # 初始化数字和字母的映射表,使用dict类型
        telDict = {'2':['a','b','c'], '3':['d','e','f'], '4':['g','h','i'], '5':['j','k','l'],
                   '6':['m','n','o'], '7':['p','q','r','s'], '8':['t','u','v'], '9':['w','x','y','z']}
        # 存储每个循环后的中间列表，用于下一个循环的输入，初始化为第一个数字对应的字母列表
        tempList = telDict.get(digits[0])
        # 存储每个循环后的结果表，作为tempList的输入，初始化为空
        resultList = []

        # 从输入的第二个数字开始，把tempList和数字对应list的每个元素做双循环组合
        for i in range(1,len(digits)):
            print('i:',i)
            for j in tempList:
                for k in telDict.get(digits[i]):
                    resultList.append(j+k)
            tempList = resultList
            resultList = []
        resultList = tempList
        print(resultList)
        return resultList

    """
    18.四数之和：给你一个由 n 个整数组成的数组 nums ，和一个目标值 target 。请你找出并返回满足下述全部条件且不重复的四元组 [nums[a], nums[b], nums[c], nums[d]] （若两个四元组元素一一对应，则认为两个四元组重复）：
       0 <= a, b, c, d < n
       a、b、c 和 d 互不相同
       nums[a] + nums[b] + nums[c] + nums[d] == target
       你可以按 任意顺序 返回答案 。
       标签：数组，双指针，排序
       https://leetcode.cn/problems/4sum/description/
    """

    def sum4_18(self, nums: list = [], target: int = 0) -> set:
        # 思路：四循环硬算吧
        resultSet = set()
        for p1 in range(len(nums)-3):
            for p2 in range(p1+1, len(nums)-2):
                for p3 in range(p2+1, len(nums)-1):
                    for p4 in range(p3 + 1, len(nums) ):
                        if (nums[p1]+nums[p2]+nums[p3]+nums[p4])==target:
                            l = [nums[p1],nums[p2],nums[p3],nums[p4]]
                            resultSet.add(str(l))
        print(resultSet)
        return resultSet

    """
    19.删除链表的倒数第N个节点：给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
       示例：输入：head = [1,2,3,4,5], n = 2，输出：[1,2,3,5]
       标签：链表，双指针
       https://leetcode.cn/problems/remove-nth-node-from-end-of-list/description/
    """
    def removeNthNodeFromEndOfList_19(self, head:list = [], n:int=0) -> list:
        resultList = head[0:len(head)-n] + head[len(head)-n+1:len(head)]
        print(resultList)
        return resultList


    """
    22. 括号生成：数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
        标签：字符串、动态规划、回溯
        https://leetcode.cn/problems/generate-parentheses/
    """

    def generateParentheses_22(self, n: int = 1) -> list:
        # 思路1：采用动态规划生长的策略（类似于力扣官方解答的方法二回溯法，只是这里没有用递归）：
        # 1、起始从一个左括号开始；
        # 2、下一步判断是否可以插入左括号、右括号，如果都可以，生成新的字符分别插入左右括号，推入list中；
        # 3、list轮询，判断，直至最后一个字符串长度为n * 2。
        totalLst = ['(']
        while len(totalLst[len(totalLst) - 1]) < n * 2:
            # 这里每一轮list都要重新赋值，感觉python体系里list的操做是传地址不是传值
            # 另外也考虑list如果直接删元素也会给下一轮循环判断条件产生困扰，就不删了，直接赋新值
            tempLstPre = []  # 存储本轮增加左右括号后的新字符串
            tempLstAft = totalLst.copy()  # 专门用于循环判断，省得list变化后循环就不对了
            for j in range(len(tempLstAft)):
                if tempLstAft[j].count(')') <= n and tempLstAft[j].count(')') < tempLstAft[j].count('('):
                    str1 = tempLstAft[j] + ')'
                    tempLstPre.append(str1)
                if tempLstAft[j].count('(') < n:
                    str2 = tempLstAft[j] + '('
                    tempLstPre.append(str2)
                totalLst = tempLstPre.copy()
        print(totalLst)
        # return totalLst

        # 思路2、力扣官网启发，思路和上述我自己的思路差不多，但这里用了递归方式，代码简洁一些
        ans = []

        def backtrack(S, left, right):
            if len(S) == 2 * n:
                ans.append(''.join(S))
                return
            if left < n:
                S.append('(')
                backtrack(S, left + 1, right)
                S.pop()
            if right < left:
                S.append(')')
                backtrack(S, left, right + 1)
                S.pop()

        backtrack([], 0, 0)
        print(ans)
        return ans

        # 上面2个思路打印出来的结果可以对比区别，取值的顺序是反的

    """
    24.两两交换链表中的节点：给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。
       你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。
       示例：输入：head = [1,2,3,4]，输出：[2,1,4,3]
       标签：递归，链表
       https://leetcode.cn/problems/swap-nodes-in-pairs/description/
    """

    def swapNodesInPairs_24(self, head:list) -> list:
        for i in range(0,len(head)-1, 2):
            tmp = head[i]
            head[i] = head[i+1]
            head[i+1] = tmp
        print(head)
        return head

    """
        29. 两数相除：给你两个整数，被除数 dividend 和除数 divisor。将两数相除，要求 不使用 乘法、除法和取余运算。
            整数除法应该向零截断，也就是截去（truncate）其小数部分。例如，8.345 将被截断为 8 ，-2.7335 将被截断至 -2 。
            返回被除数 dividend 除以除数 divisor 得到的 商 。
            注意：假设我们的环境只能存储 32 位 有符号整数，其数值范围是 [−2**31,  2**31 − 1] 。
            本题中，如果商 严格大于 231 − 1 ，则返回 2**31 − 1 ；如果商 严格小于 -2**31 ，则返回 -2**31 。
            标签：位运算，数学
            https://leetcode.cn/problems/divide-two-integers/
    """

    def divideTwoIntegers_29(self, dividend: int = 0, divisor: int = 1) -> int:
        # 按题目要求，先对临界情况判断。
        # 如果被除数=−2**31：若除数=1，则返回−2**31；若除数=-1，则返回2**31 − 1；
        # 如果除数=−2**31：若被除数=−2**31，则返回1；其余情况返回0；
        # 如果除数为0 ，返回0
        if dividend == -(2 ** 31):
            if divisor == 1:
                return -2 ** 31
            elif divisor == -1:
                return 2 ** 31 - 1
        if divisor == -(2 ** 31):
            if dividend == -2 ** 31:
                return 1
            else:
                return 0
        if divisor == 0:
            return 0

        # 思路1、笨办法，让被除数不停地减除数，一直减到被除数小于除数
        i = 0
        dividend1 = abs(dividend)
        divisor1 = abs(divisor)
        while dividend1 > divisor1:
            i = i + 1
            dividend1 = dividend1 - divisor1

        if (dividend > 0 > divisor) or (dividend < 0 < divisor):
            i = - i
        print(i)

        # return i

        # 思路2、参考力扣官方解答：我们要寻找的是x满足x*divisor<dividend<(x+1)*divisor
        # 也就是可以在1 ~ 2**31−1 的范围内，用二分查找法定位到x
        # 其次，在具体的x*divisor<dividend<(x+1)*dividend判断中，如何避免使用乘法，把乘法化为快速加，这里增加一个单独的函数定义
        def fastAdd(xx: int = 0, yy: int = 0) -> int:
            x1, y1 = xx, yy
            while y1 > 1:
                if (y1 & 1) == 1:  # 看y1的位数是否奇数，如果是奇数，x = x*2 +y
                    x1 = (x1 << 1) + x1
                else:  # 如果y1是偶数，x = x*2
                    x1 = x1 << 1
                y1 = y1 >> 1  # 无论y1是奇数还是偶数，y1整除2，继续下一轮循环
            return x1

        # 二分法判断x
        x, left, right = 2 ** 31 >> 1, 0, 2 ** 31
        dividend1 = abs(dividend)
        divisor1 = abs(divisor)
        while True:
            a = fastAdd(x, divisor1)
            if a <= dividend1 <= a + divisor1:
                break
            elif a + divisor1 < dividend1:
                left = x
                x = (right + left) >> 1
            elif dividend1 < a:
                right = x
                x = (right + left) >> 1

        if (dividend > 0 > divisor) or (dividend < 0 < divisor):
            x = - x
        print(x)
        return x

    """
        31. 下一个排列：整数数组的 下一个排列 是指其整数的下一个字典序更大的排列。
            更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，那么数组的 下一个排列 就是在这个有序容器中排在它后面的那个排列。
            如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。
            例如，arr = [1,2,3] 的下一个排列是 [1,3,2] 。类似地，arr = [2,3,1] 的下一个排列是 [3,1,2] 。
            而 arr = [3,2,1] 的下一个排列是 [1,2,3] ，因为 [3,2,1] 不存在一个字典序更大的排列。
            给你一个整数数组 nums ，找出 nums 的下一个排列。必须 原地 修改，只允许使用额外常数空间。
            标签：数组，双指针
            https://leetcode.cn/problems/next-permutation/
    """

    def nextPermutation_31(self, num: list = []) -> list:
        # 思路：
        # 1、从右往左，找到第一个num[i]<num[i+1]的位置i，这个i是最靠右的相对较小可以换的数字；
        # 2、此时i右边的数列一定是降序排列的，从右往左，找到第一个num[j]>num[i]的位置j；
        # 3、交换i和j的值，此时i右边的数列依然是降序排列的，将其反转即可
        i = len(num) - 1
        flag = True
        # 定位i
        while flag:
            if num[i] > num[i - 1] or i == 0:
                flag = False
            i = i - 1
        # 如果num完全是倒序排列的，那就返回正序排列值
        if i == -1:
            num.reverse()
            print('完全是倒序排列的', num)
        # 否则，
        else:
            # 再从右往左，找到第一个num[j]>num[i]的位置j
            j = len(num) - 1
            while num[j] < num[i] and j != i:
                j = j - 1
            # 交换num[i]和num[j]的值
            a = num[i]
            num[i] = num[j]
            num[j] = a
            print('i=', i, 'j=', j, '交换num[i]和num[j]的值', num)
            # i后面的数列倒叙排序
            for k in range((len(num) - i) // 2):
                print('k=', k, '(len(num) - i) // 2=', (len(num) - i) // 2)
                a = num[len(num) - k - 1]
                num[len(num) - k - 1] = num[i + k + 1]
                num[i + k + 1] = a
            print('排序后的值：', num)
        return num

    """
        33. 搜索旋转排序数组：整数数组 nums 按升序排列，数组中的值 互不相同 。
            在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为
             [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。
            例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。
            给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。
            你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。
            标签：数组，二分查找
            https://leetcode.cn/problems/search-in-rotated-sorted-array/
    """

    def searchInRotatedArray_33(self, nums: list = [], target: int = 0) -> int:
        # 根据旋转数组的定义，第一个数肯定大于最后一个数：nums[0]>nums[n-1]，但我们不知道最大值或最小值所在的位置

        # 思路1：首先用二分查找法定位最大值或最小值所在的位置，其次再用二分查找法找到target的位置，这个时间复杂度相当于O(2 * log n)
        beginidx = 0
        endidx = len(nums) - 1
        while endidx > beginidx:
            i = (endidx + beginidx) // 2
            if nums[i] < nums[0]:
                beginidx = (i + beginidx) // 2
                endidx = i
            else:
                endidx = (endidx + i) // 2
                beginidx = i

        # 此时i是最大数所在的位置
        print('最大值的位置在', i)

        # 用二分查找法定位target，先确定在哪半边
        if nums[0] <= target <= nums[i]:
            beginidx = 0
            endidx = i
        elif nums[i + 1] <= target <= nums[len(nums) - 1]:
            beginidx = i + 1
            endidx = len(nums) - 1
        else:
            print('target', target, '超出数列范围了。')
            return -1
        # 再用二分法查找定位
        while endidx > beginidx:
            j = (endidx + beginidx) // 2
            if target == nums[j]:
                print('target', target, '位置在', j)
                return j
            elif target < nums[j]:
                endidx = j
            else:
                beginidx = j + 1
        j = (endidx + beginidx) // 2

        if target == nums[j]:
            print('target', target, '位置在', j)
            return j
        else:
            print('target', target, '不在数列中。')
            return -1

        # 思路2：这是一个变相的二分查找法。
        # 在做二分查找时，需要判断，每二分出来的三个点数据形状，可能是山峰或者河谷或者上坡，三种情况判断的逻辑略有些不同
        # 其实if条件可以不用写的那么复杂，这里不优化了
        # 这个时间复杂度是O(log n)
        beginidx = 0
        endidx = len(nums) - 1
        while endidx > beginidx:
            i = (endidx + beginidx) // 2
            if nums[i] < nums[endidx] < nums[beginidx]:  # 河谷型
                if nums[i] <= target <= nums[endidx]:
                    beginidx = i
                elif target >= nums[beginidx] or target < nums[i]:
                    endidx = i - 1
            elif nums[endidx] < nums[beginidx] < nums[i]:  # 山峰型
                if nums[beginidx] <= target <= nums[i]:
                    endidx = i
                elif target > nums[i] or target <= nums[endidx]:
                    beginidx = i + 1
            elif nums[beginidx] < nums[i] < nums[endidx]:  # 上坡型
                if nums[beginidx] <= target <= nums[i]:
                    endidx = i - 1
                elif nums[i] < target <= nums[endidx]:
                    beginidx = i + 1
        i = (endidx + beginidx) // 2
        print(beginidx, i, endidx)
        if target == nums[i]:
            print('target', target, '位置在', i)
            return j
        else:
            print('target', target, '不在数列中。')
            return -1

    """
        34. 在排序数组中查找元素的第一个和最后一个位置。
            给你一个按照非递减顺序排列的整数数组 nums，和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。
            如果数组中不存在目标值 target，返回 [-1, -1]。你必须设计并实现时间复杂度为 O(log n) 的算法解决此问题。
            示例 1：输入：nums = [5,7,7,8,8,10], target = 8，输出：[3,4]
            示例 2：输入：nums = [5,7,7,8,8,10], target = 6，输出：[-1,-1]
            标签：数组，二分查找
            https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/
    """

    def findPositionsInSortedArray_34(self, nums: list = [], target: int = 0) -> list:
        # 先判断元素是否在列表中
        beginidx = 0
        endidx = len(nums) - 1

        if not target in nums:
            print('target', target, '不在列表中')
            return [-1, -1]

        # 思路1：本题难点在于不是要定位一个位置，而是定位一个位置范围。
        # 在二分查找过程中，最难处理的是当中间二分的值正好等于target的时候，不好移动左右坐标进一步二分，这里只能是把左右坐标往里面缩一格，
        # 此时最坏的情况就是target在正中间，时间复杂度O(n/2)

        # 直到beginidx和endidx都等于target时才跳出循环
        while nums[beginidx] < target or nums[endidx] > target:
            i = (beginidx + endidx) // 2
            # print('start:', beginidx, i, endidx)
            if nums[i] == target:  # 中间值等于target时，两边各缩进1位，但也要判断是否等于target，等于的话就不能缩了
                if nums[beginidx] < target:
                    beginidx = beginidx + 1
                if nums[endidx] > target:
                    endidx = endidx - 1
            elif nums[i] < target:  # 中间值大于或小于target的时候，就可以二分缩进了
                beginidx = i + 1
                if nums[endidx] > target:
                    endidx = endidx - 1
            elif nums[i] > target:
                if nums[beginidx] < target:
                    beginidx = beginidx + 1
                endidx = i - 1
            # print('end:', beginidx, i, endidx)

        # print(beginidx, endidx)
        # if beginidx == endidx:
        #     return [beginidx]
        # else:
        #     return [beginidx, endidx]

        # 思路2、参考力扣官网，我们要找的两边坐标本质是，寻找第一个num[beginidx]=target 和第一个 num[endidx]>target
        # 所以用2个二分查找法，分别找beginidx, endidx，力扣官网把它们抽象出一个函数来了，这里没有，就代码重复罗嗦点
        beginidx = 0
        endidx = len(nums) - 1
        while beginidx < endidx:  # 寻找第一个num[i]=target
            i = (beginidx + endidx) // 2
            if nums[i] >= target:
                endidx = i
            else:
                beginidx = i + 1
        leftidx = beginidx

        beginidx = 0
        endidx = len(nums) - 1
        while beginidx < endidx:  # 寻找第一个 num[endidx]>target
            j = (beginidx + endidx) // 2
            if nums[j] > target:
                endidx = j
            else:
                beginidx = j + 1
        rightidx = beginidx - 1
        print('left', leftidx, 'right:', rightidx)
        return [leftidx, rightidx]

    """
        36. 有效的数独：请你判断一个 9 x 9 的数独是否有效。只需要 根据以下规则 ，验证已经填入的数字是否有效即可。
            数字 1-9 在每一行只能出现一次。数字 1-9 在每一列只能出现一次。数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。
            注意：一个有效的数独（部分已被填充）不一定是可解的。只需要根据以上规则，验证已经填入的数字是否有效即可。空白格用 '.' 表示。
            标签：数组，哈希表，矩阵
            https://leetcode.cn/problems/valid-sudoku/
    """

    def validSudoku_36(self, matrix99: list = []) -> bool:
        # 思路1：先设计一个子程序，入参是9个数，判断这9个数是否重复；再反复调用该子程序，判断每行、每列、每宫是否有重复数字
        # 这个思路比较慢，相当于整个矩阵遍历了3遍
        def hasRepNum(nums: list = []) -> bool:
            a = set()
            for i in range(9):
                if nums[i] != '.' and nums[i] in a:
                    return False
                elif nums[i] != '.' and nums[i] not in a:
                    a.add(nums[i])
            return True

        # 判断每一行
        for i in range(9):
            if not hasRepNum(matrix99[i]):
                print('False')
                return False
        # 判断每一列
        for i in range(9):
            l = []
            for j in range(9):
                l.append(matrix99[j][i])
            if not hasRepNum(l):
                print('False')
                return False
        # 判断每一宫
        for i in range(3):
            for j in range(3):
                l = []
                for k in range(3):
                    for m in range(3):
                        l.append(matrix99[i * 3 + k][j * 3 + m])
                if not hasRepNum(l):
                    print('False')
                    return False
        # 最终
        print('True')
        return True

        # 思路2：构造哈希表，存储每行、每列、每宫中，每个数字出现的次数，当次数超过1时即为False。只需遍历矩阵1次，相当于时间换空间
        # 每行、每列key的规则：两位数字，第一位0-8表示0-8行，9-17表示0-8列；第二位1-9表示数字，对应value值初始化为0
        # 每宫key的规则：三位数字，前两位分别是0-8表示每宫最左上角的坐标，第三位1-9表示数字，对应value值初始化为0

        # 初始化哈希表
        hm = {}
        for i in range(18):
            for j in range(9):
                hm[str(i) + str(j + 1)] = 0
        for i in range(3):
            for j in range(3):
                for k in range(9):
                    hm[str(i * 3) + str(j * 3) + str(k + 1)] = 0
        print(hm)
        # 遍历矩阵，计算数字出现次数，并计入相应哈希表
        for i in range(9):
            for j in range(9):
                if matrix99[i][j] != '.':
                    hm[str(i) + matrix99[i][j]] = hm[str(i) + matrix99[i][j]] + 1  # 行
                    hm[str(9 + j) + matrix99[i][j]] = hm[str(9 + j) + matrix99[i][j]] + 1  # 列
                    hm[str(i - i % 3) + str(j - j % 3) + matrix99[i][j]] \
                        = hm[str(i - i % 3) + str(j - j % 3) + matrix99[i][j]] + 1  # 宫
        print(hm)

        # 判断哈希表中是否有超过1次的情况，这个也可以放到上面那个循环里提升时间效率，为清晰展示单独拿出来
        for i in hm.keys():
            if hm[i] > 1:
                print('False')
                return False
        print('True')
        return True

    """
        38. 外观数列：给定一个正整数 n ，输出外观数列的第 n 项。
           「外观数列」是一个整数序列，从数字 1 开始，序列中的每一项都是对前一项的描述。
            前五项如下：
                1.     1               第一项是数字 1 
                2.     11              描述前一项，这个数是 1 即 “ 一 个 1 ”，记作 "11"
                3.     21              描述前一项，这个数是 11 即 “ 二 个 1 ” ，记作 "21"
                4.     1211            描述前一项，这个数是 21 即 “ 一 个 2 + 一 个 1 ” ，记作 "1211"
                5.     111221          描述前一项，这个数是 1211 即 “ 一 个 1 + 一 个 2 + 二 个 1 ” ，记作 "111221"
            要描述一个数字字符串，首先要将字符串分割为 最小 数量的组，每个组都由连续的最多相同字符组成。
            然后对于每个组，先描述字符的数量，然后描述字符，形成一个描述组。
            要将描述转换为数字字符串，先将每组中的字符数量用数字替换，再将所有描述组连接起来。
            标签：字符串
            https://leetcode.cn/problems/count-and-say/
    """

    def countAndSay_38(self, n: int = 1) -> str:
        # 思路：不要用递归了，循环吧，1~n
        finalStr = '1'
        if n == 1:
            print(finalStr)
            return finalStr
        else:
            # 从2开始循环到n
            for i in range(2, n + 1):
                temstr1 = finalStr  # temstr1记录每次需要计算的剩余字符串
                finalStr = ''  # finalStr记录每次计算出来的临时结果
                # 对剩余字符串反复判断：
                startpos = 0
                while startpos < len(temstr1):
                    # j是从0开始的坐标，一直+1，直到j+1的字符跟j不一样，表示当前一段重复字符判断结束
                    j = startpos
                    while j < len(temstr1) - 1 and temstr1[j] == temstr1[j + 1]:
                        j = j + 1
                    # 由于j表示坐标，实际上重复字符值的数量是j - startpos + 1
                    finalStr = finalStr + str(j - startpos + 1) + temstr1[j]
                    # 将startpos 标为 j + 1进入下一轮判断
                    startpos = j + 1
                print('i', i, 'finalStr', finalStr)

        print(finalStr)
        return finalStr

    """
        39. 组合总和：给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，
            找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。
            candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 
            对于给定的输入，保证和为 target 的不同组合数少于 150 个。
            标签：数组，回溯
            https://leetcode.cn/problems/combination-sum/
    """

    def combinationSum_39(self, candidates: list = [], target: int = 0) -> list:
        # 0、先把所有元素补上重复的部分，再从小到大排序。
        # 补重复的原则是补上target//i个，比如target是7，有个元素是2，那么就补2个，一共3个；有个元素是3，那么就补1个，共2个。
        # 例如：candidates = [2,3,6,7], target = 7，那么补充排序后的列表是：[2,2,2,3,3,6,7]
        #      candidates = [2,3,5], target = 8，那么补充排序后的列表是：[2,2,2,2,3,3,5]
        # 最终形成一个二维列表如[[2,2,2,2],[3,3],[5]]
        candidates.sort()
        readyL = []
        for i in candidates:
            a = []
            for j in range(target // i):
                a.append(i)
            readyL.append(a)
        print(readyL)

        # 死办法，每种组合都尝试一遍，看看组合是否等于tartget
        # 1、把每个组合都枚举出来，形成二维数组finalL，每个元素包含n个数字，分别代表n个不同值的系数
        allCoeL = []
        for j in range(len(readyL[0]) + 1):
            allCoeL.append([j])
        for i in range(1, len(readyL)):
            tempL = []
            for k in range(len(allCoeL)):
                for j in range(len(readyL[i]) + 1):
                    tempL.append(allCoeL[k] + [j])
            allCoeL = tempL
        print(len(allCoeL), allCoeL)
        # 2、计算每个组合的和是否等于target，把组合系数先提出来
        combCoeL = []
        for i in range(len(allCoeL)):
            s = 0
            for j in range(len(allCoeL[i])):
                s = s + allCoeL[i][j] * readyL[j][0]
            if s == target:
                combCoeL.append(allCoeL[i])
        print(combCoeL)
        # 3、形成最终的答案列表
        finalL = []
        for i in range(len(combCoeL)):
            l = []
            for j in range(len(combCoeL[i])):
                for k in range(combCoeL[i][j]):
                    l = l + [readyL[j][0]]
            finalL.append(l)
        print(finalL)
        return finalL

    """
        依然是39题，参考官方答案，使用回溯方法，这里要用到递归函数
        思路是：重复把target减去候选数组，直到最终target<=0，如果=0则符合条件，把结果加入results中，注意要把过程中每一步减去数字的过程记录下来
    """

    def combinationSum_39_LookBack(self, candidates: list = [], target: int = 0) -> list:
        # 核心递归函数参数：subtarget：目标和
        #                subtraction本轮被减数
        #                path：本轮搜索中已经经过的路径，记录已被减去的数字
        #                result：最终符合条件的结果集
        def recursionCombSum(subtarget: int = 0, subtraction: int = 0, path: list = [], result: list = []):

            print('begin subtarget:', subtarget, 'subtraction:', subtraction, 'path:', path, 'result:', result)
            # if subtarget < 0:
            #     return
            if subtarget == 0:
                result.append(path)
                print('==:', 'subtarget:', subtarget, 'subtraction:', subtraction, 'path:', path, 'result:', result)
                return
            if subtarget > 0:
                for j in candidates:
                    # 这里做一个剪枝操作
                    if subtarget - subtraction >= 0:
                        # 注意这里递归参数的带入，python函数的参数都是传址（引用），而不是传值，
                        # 但两个list相加会新生成一个对象，不会改变原来的path
                        recursionCombSum(subtarget - subtraction, j, path + [subtraction], result)
                        print()
                return

        results = []
        t = target
        for i in candidates:
            path = []
            recursionCombSum(t, i, path, results)
        print(results)

        # 上述递归函数调用结果可以通过打印看出，还是会有很对重复值的，此处再去重
        for i in results:
            i.sort()
        results.sort()
        i = 0
        while i < len(results) - 1:
            if results[i] == results[i + 1]:
                results.remove(results[i])
            else:
                i = i + 1
        print(results)
        return results

    """
        依然是39题，参考精选答案中的回溯方法和递归函数
        这个方法和上面的区别是不会产生重复的答案，且有剪枝操作，更为精巧
        https://leetcode.cn/problems/combination-sum/solutions/14697/hui-su-suan-fa-jian-zhi-python-dai-ma-java-dai-m-2/
    """

    def combinationSum_39_LookBack_Winnow(self, candidates: list, target: int):

        size = len(candidates)
        if size == 0:
            return []
        candidates.sort()

        # 递归函数，参数：
        #   begin：搜索候选数组的起始位置
        #   path：本轮搜索中已经经过的路径，记录已被减去的数字，
        #   res：最终符合条件的结果集
        #   target：目标和
        def dfs(begin, path, res, target):
            print('begin', begin, 'path', path, 'res', res, 'target', target)
            # if target < 0:
            #     return
            if target == 0:
                res.append(path)
                return

            for index in range(begin, size):
                residue = target - candidates[index]
                if residue < 0:
                    break
                dfs(index, path + [candidates[index]], res, residue)

        path = []
        res = []
        dfs(0, path, res, target)
        print(res)

    """
        40. 组合总和 II：给定一个整数列表 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。
            candidates 可能包含重复数字，每个数字在每个组合中只能使用 一次 。注意：解集不能包含重复的组合。 
            标签：数组，回溯
            https://leetcode.cn/problems/combination-sum-ii/
    """

    def combinationSumII_40(self, candidates: list = [], target: int = 0) -> list:

        # 这题跟上一题不同之处是，不能重复使用列表里的元素，那么在递归函数的调用中，要加上candidates列表参数，
        # 当使用一个元素后把它从candidates里去掉
        def dfs(subcandidates: list, path: list, subtarget: int, subtraction: int, result: list):
            if subtarget == 0:
                result.append(path)
                return
            if subtarget < 0:
                return

            l = subcandidates.copy()
            l.remove(subtraction)
            for i in l:
                if subtarget - subtraction >= 0:
                    dfs(l, path + [subtraction], subtarget - subtraction, i, result)

        results = []
        path = []
        for j in candidates:
            dfs(candidates, path, target, j, results)

        print(results)

        # 上述递归函数调用结果可以通过打印看出，还是会有很对重复值的，此处再去重
        for i in results:
            i.sort()
        results.sort()
        i = 0
        while i < len(results) - 1:
            if results[i] == results[i + 1]:
                results.remove(results[i])
            else:
                i = i + 1
        print(results)
        return results

        # 官方和精选解答，在递归过程中考虑了重复值和剪枝的问题，比这个更精妙，这里不写了。反正这个已经是我智力极限了，丧……

    """
    43. 字符串相乘：给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。
        注意：不能使用任何内置的 BigInteger 库或直接将输入转换为整数。
        标签：数学，字符串，模拟
        https://leetcode.cn/problems/multiply-strings/
    """

    def multiplyStrings_43(self, str1: str, str2: str) -> int:
        # 思路1：不完全的模拟整数竖式相乘
        # 先将str1转换成整数形式
        i = 0
        num1 = 0
        while i < len(str1):
            num1 = num1 * 10 + int(str1[i])
            i = i + 1
        print(num1)

        # 再将str2的每一位乘以num1
        i = len(str2) - 1
        num2 = 0
        while i >= 0:
            num2 = num2 + int(str2[i]) * num1 * (10 ** (len(str2) - i - 1))
            i = i - 1
        print(num2)

        # 思路2：完全的模拟整数竖式相乘，就是str1也一位一位地算
        i = len(str2) - 1
        num2 = 0
        while i >= 0:
            j = len(str1) - 1
            num1 = 0
            while j >= 0:
                num1 = num1 + int(str2[i]) * int(str1[j]) * (10 ** (len(str1) - j - 1))
                j = j - 1
            num2 = num2 + num1 * (10 ** (len(str2) - i - 1))
            i = i - 1
        print(num2)

        return num2

    """
    45. 跳跃游戏 II：给定一个长度为 n 的 0 索引整数数组 nums。初始位置为 nums[0]。
        每个元素 nums[i] 表示从索引 i 向前跳转的最大长度。
        换句话说，如果你在 nums[i] = j 处，你可以跳转到任意 nums[i + j] 处: 0 <= j <= nums[i] ，i + j < n
        返回 从nums[0] 到达 nums[n - 1] 的最小跳跃次数。生成的测试用例可以到达 nums[n - 1]。
        标签：贪心，数组，动态规划
        https://leetcode.cn/problems/jump-game-ii/
    """

    def jumpGameII_45(self, nums: list = []) -> int:
        # 思路：从nums[0]开始，看看能有几种跳法，把所有跳法加入队列中，再轮询队列中所有元素（子队列），直到到达或者超出nums[n - 1]
        # 这里也使用递归函数实现，参数：
        #     resultList：是个双重list，它的每个元素也是个list，记录每个步骤nums的坐标
        #     beginidx：记录每次路径从哪里开始动态规划
        #               因为resultList从开始记录了所有过程，但每一轮动态规划迭代后，只需要从最后一轮的若干候选list判断就好了，
        #               不需要从头判断，否则会死循环
        def dp(resultList: list, beginidx: int):
            i = beginidx
            while i < len(resultList):
                print('resultList begin', resultList, 'i', i, 'beginidx', beginidx)
                d = resultList[i]
                lastidx = d[len(d) - 1]
                lastval = nums[lastidx]
                if lastidx >= len(nums) - 1:  # 终止条件：跳转坐标大于等于num的最大坐标了
                    return
                else:  # 否则，继续寻找所有可跳转的方案，并把路径加入resultList
                    for j in range(1, lastval + 1):
                        dd = d.copy()
                        dd.append(lastidx + j)
                        resultList.append(dd)
                        # 这里增加一个终止条件，如果当前路径已经到达终点了，那么所有递归终止，这里就是最短路经
                        # 如果不加终止条件，会把所有路径方案加进去
                        if lastidx + j == len(nums) - 1:
                            return
                i = i + 1
                print('resultList end', resultList, 'i', i, 'resultList长度', len(resultList))

            dp(resultList, i - 1)

        # 调用递归函数
        resultLists = []
        di = [0]
        resultLists.append(di)
        dp(resultLists, 0)

        # 打印结果，此时结果包含所有过程路径
        print()
        print(resultLists)

        # 选出最后那个元素就是结果
        k = resultLists[len(resultLists) - 1]
        l = len(k)
        print('最短路径的坐标是', k, '步骤数', l)
        return l

    """
        46. 全排列：给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
            标签：数组，回溯
            https://leetcode.cn/problems/permutations/
    """

    def permutations_46(self, nums: list = []) -> list:
        # 思路：设计递归函数，深度回溯方法，列举所有可能的排列
        # 参数：path：目前已做的排列路径
        #      candidate：path中元素之外剩下的可选数字
        #      result：最终的所有排列集合
        def recursion(path: list, candidate: list, result: list):
            print('path', path, 'candidate', candidate, 'result', result)
            # 当所有数字已经排列完成时，加入result结果集
            if len(path) == len(nums):
                result.append(path)
                return
            # 在剩下可选的数字中，逐个加入path路径。
            # 注意，由于python函数的参数传递的都是引用（传址），
            # 所以这里递归调用的时候要新new参数，不能将原path、candidate直接作为参数传入
            # path+[i]本身就会新声称对象，所以不用显式new一个对象
            for i in candidate:
                l2 = candidate.copy()
                l2.remove(i)
                recursion(path + [i], l2, result)
            # 上轮循环完毕后进入下个循环前，要把path的最后一个元素吐出来，否则结果不对
            if len(path) > 0:
                path.pop()

        results = []
        paths = []
        candidates = nums.copy()
        recursion(paths, candidates, results)
        print('组合数量', len(results), 'results', results)
        return results

    """
    47. 全排列II：给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。
        示例 1：输入：nums = [1,1,2]，输出：[[1,1,2], [1,2,1], [2,1,1]]
        标签：数组，回溯，排序
        https://leetcode.cn/problems/permutations-ii/description/
    """

    def permutationsII_47(self,nums: list = []) -> list:
        # 思路1：简单粗暴，把递归函数中result参数设置成set类型，让set自行判断去重，这里就不写了
        # 思路2：加入判断，当待处理元素在之前已经出现过，就跳过不处理
        def recursion(path: list, candidate: list, result: list):
            print('path', path, 'candidate', candidate, 'result', result)
            # 当所有数字已经排列完成时，加入result结果集
            if len(path) == len(nums):
                result.append(path)
                return
            # 这里和46唯一不同的是，加入判断，当待处理元素在之前已经出现过，就跳过不处理
            for i in range(len(candidate)):
                flag = 0
                for j in range(i):
                    if candidate[j] == candidate[i]:
                        flag = 1
                if flag == 0:
                    l2 = candidate.copy()
                    l2.remove(candidate[i])
                    recursion(path + [candidate[i]], l2, result)
            # 上轮循环完毕后进入下个循环前，要把path的最后一个元素吐出来，否则结果不对
            if len(path) > 0:
                path.pop()

        results = []
        paths = []
        candidates = nums.copy()
        recursion(paths, candidates, results)
        print('组合数量', len(results), 'results', results)
        return results

    """
    48. 旋转图像：给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。
        你必须在 原地 旋转图像，这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。
        标签：数组，数学，矩阵
        https://leetcode.cn/problems/rotate-image/description/
    """

    def rotateImage_48(self,matrix:list ) -> list:
        # 思路：要求原地旋转图像，那么仅设置一个临时变量
        # 先旋转最外圈4个顶点，再旋转最外圈顶点之外的所有元素
        # 外圈旋转完了，把内圈看成新矩阵，思路同上
        # 下述被注释的代码写得不对，这题挺绕的
        dim = len(matrix)
        # 设计一个子程序，每一次执行完成一圈
        # 参数：n：完成第几圈的旋转,n从0开始
        def recursion(n:int):
            print('第',n,"圈")
            # # 先旋转最外圈4个顶点
            # tmp = matrix[n][n]
            # matrix[n][n] = matrix[dim-n-1][n]
            # matrix[dim-n-1][n] = matrix[dim-n-1][dim-n-1]
            # matrix[dim-n-1][dim-n-1] = matrix[n][dim-n-1]
            # matrix[n][dim-n-1] = tmp
            #
            # # 再旋转最外圈顶点之外的所有元素，这里j表示循环次数，同时也表示第一列待处理元素的下标dim-n-1
            # for j in range(n+1,dim-n-1):
            #     print("j:",j)
            #     tmp = matrix[j][n]
            #     print(dim-n-1,j,'->',j,n)
            #     matrix[j][n] = matrix[dim-n-1][j]
            #     print(dim-n-j-1, dim-n-1, '->', dim-n-1, j)
            #     matrix[dim-n-1][j] = matrix[dim-n-j-1][dim-n-1]
            #     print(n,dim-n-j-1,'->',dim-n-j-1,dim-n-1)
            #     matrix[dim-n-j-1][dim-n-1] = matrix[n][dim-n-j-1]
            #     print(j,n,'->',n,dim-n-j-1)
            #     matrix[n][dim-j-1] = tmp
        # 下述代码是抄的Leetcode标准答案，但是内循环j的取值方式我没想明白
        for i in range(dim//2):
            print('i:',i)
            for j in range((dim+1)//2):
                print('    j:',j)
                tmp = matrix[i][j]
                print('    ',dim-j-1,i,'->',i,j)
                matrix[i][j] = matrix[dim - j - 1][i]
                print('    ',dim-i-1,dim-j-1,'->',dim-j-1,i)
                matrix[dim - j - 1][i] = matrix[dim - i - 1][dim - j - 1]
                print('    ',j,dim-i-1, '->',dim-i-1,dim-j-1 )
                matrix[dim - i - 1][dim - j - 1] = matrix[j][dim - i - 1]
                print('    ',i,j, '->',j,dim-i-1 )
                matrix[j][dim - i - 1] = tmp
                print()
            print(matrix)

        return matrix

    """
    49. 字母以为词分组：给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。
        示例 1:输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]，输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
        标签：数组，哈希表，字符串，排序
        https://leetcode.cn/problems/group-anagrams/description/
    """

    def groupAnagrams_49(self,strings:list) -> list:
        # 思路1：双重循环，硬算，每个单词和其他所有单词比较是否是异位词，是的话放在一起

        # 先定义一个子函数，判断2个单词是否异位词，这里先把字符串劈成单个字母的列表，再将列表排序，再对比，有点子蠢
        def anagrams(a:str,b:str) -> bool:
            a1 = list(a)
            a1.sort()
            b1 = list(b)
            b1.sort()
            if a1==b1:
                return True
            else:
                return False

        strs1 = strings.copy()
        strs2 = strings.copy()
        resultList = []
        # 这个方法有不足，就是无法处理列表中有重复的单词，最终结果是没有重复的
        for i in strs1:
            tmpList = [i]
            strs2.remove(i)
            for j in strs2:
                if anagrams(i,j):
                    tmpList.append(j)
                    strs1.remove(j)
                    strs2.remove(j)
            resultList.append(tmpList)

        print('resultList:', resultList)
        # return resultList

        # 思路2：使用哈希表，单词排序后的值作为key，符合该条件的所有单词列表作为value
        # 这个方法可以处理保留列表中重复的单词，另外队友所有单词只排一次序，性能上优于思路1
        resultDict = {}
        for i in strings:
            l1 = list(i)
            l1.sort()
            s1= "".join(l1)
            if s1 in resultDict:
                resultDict[s1].append(i)
            else:
                resultDict[s1] = [i]

        resultList = list(resultDict.values())
        print('resultList:', resultList)
        return resultList

    """
    50. Pow(x,n)：实现 pow(x, n) ，即计算 x 的整数 n 次幂函数。
        标签：递归，数学
        https://leetcode.cn/problems/powx-n/description/
    """

    def pow_50(self, x:int,n:int) -> int:
        # 思路1：最原始比较蠢的办法，一步一步循环n次乘
        result = 1
        for i in range(n):
            result *= x
        print(result)
        # return result

        # 思路2：官方题解方法一：快速幂+递归
        # 不断平方上来，遇到奇数多乘一个x
        def quickMul(nn):
            if nn==0:
                return 1
            y = quickMul(nn//2)
            if (nn%2)==0:
                y = y * y
            else:
                y = y * y * x
            return y

        result = quickMul(n)
        print(result)
        # return result

        # 思路3：官方题解方法二：快速幂+迭代
        # 思路跟2差不多，只是不用递归的方式表达，用循环
        result = 1
        tmpx = x
        while n>0:
            if n%2==1:
                result = result * tmpx
            tmpx = tmpx * tmpx
            n = n//2
        print(result)
        return result

    """
    53. 最大子数组和：给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。子数组是数组中的一个连续部分。
        示例 1：输入：nums = [-2,1,-3,4,-1,2,1,-5,4]，输出：6，解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。
        标签：数组，分治，动态规划
        https://leetcode.cn/problems/maximum-subarray/description/
    """

    def maximumSubarray_53(self, nums: list[int]) -> list[int]:
        # 思路1：简单粗暴双循环，每种组合都尝试一下，保留和最大的情况
        maxval = nums[0]
        maxbegin = 0
        maxend = 0
        # i表示每次判断的子数组长度，范围从1到len(nums)
        for i in range(1, len(nums)+1):
            # j表示每次判断的起始坐标，范围从0到len(nums)-i
            for j in range(len(nums)-i+1):
                print(i,j,j+i-1)
                # 累计坐标j:j+i-1区间的元素
                tmplist = nums[j:j+i-1]
                tmpsum = 0
                for k in tmplist:
                    tmpsum += k
                # 判断是否最大，如是记下来
                if maxval < tmpsum:
                    maxval = tmpsum
                    maxbegin = j
                    maxend = j+i-1
        print('maxval:', maxval)
        print(nums[maxbegin:maxend])
        # return nums[maxbegin:maxend]

        # 思路2：官方解答，动态规划，这个只能求出最终的和，不能给出具体的数列，性能优于思路1
        # 假设时刻记录截至当前节点时，前面序列最大的可能和，那么当前的最大值，就是前面序列最大的可能加上当前节点值再取最大
        # 那么从前到后循环遍历一遍列表，就能够记录完整列表的最大可能和
        # 初始化，每个节点前面数列和的最大值pre为0，最终所有的数据的和最大值maxSum为第一个元素
        pre = 0
        maxSum = nums[0]
        for i in nums:
            # 取前序最大值加上当前节点后，与当前节点的最大值
            pre = max(pre+i, i)
            # 再取上述和最终结果比较取最大值
            maxSum = max(maxSum, pre)
        print(maxSum)
        return []

if __name__ == "__main__":
    ma = MediumAlgorithm0_99()
    # ma.longestSubstrWithoutRepeatChars_3('abcabcdbb')
    # ma.longestPalindromicSubstr_5('aabcbaeeuiywpwiud')
    # ma.zigzagConversion_6('PAYPALISHIRING', 3)
    # ma.zigzagConversion_6('PAYPALISHIRING', 4)
    # ma.zigzagConversion_6('PAYPALISHIRING', 5)
    # ma.reverseInteger_7(-123)
    # ma.strToIntegerAtoi_8('-00ieur23857021+hfd-hg3456')
    # ma.containerWithMostWater_11([1, 8, 6, 2, 5, 4, 8, 3, 7])
    # ma.sum3_15([-1, 0, 1, 2, -1, -4])
    # ma.sum3Closest_16([-1, 2, 1, -4], 1)
    # ma.generateParentheses_22(4)
    # ma.divideTwoIntegers_29(8, 3)
    # ma.divideTwoIntegers_29(-10, 3)
    # ma.nextPermutation_31([1, 2, 3, 4, 5, 6])
    # ma.nextPermutation_31([1, 2, 5, 6, 4, 3])
    # ma.nextPermutation_31([6, 5, 4, 3, 2, 1])
    # ma.searchInRotatedArray_33([7, 8, 9, 10, 11, 12, 3, 4, 5], 5)
    # ma.findPositionsInSortedArray_34([5, 7, 7, 8, 8, 8, 9, 10], 8)
    # ma.validSudoku_36([["8", "3", ".", ".", "7", ".", ".", ".", "."]
    #                       , ["6", "8", ".", "1", "9", "5", ".", ".", "."]
    #                       , [".", "9", ".", ".", ".", ".", ".", "6", "."]
    #
    #                       , [".", ".", ".", ".", "6", ".", ".", ".", "3"]
    #                       , ["4", ".", ".", "8", ".", "3", ".", ".", "1"]
    #                       , ["7", ".", ".", ".", "2", ".", ".", ".", "6"]
    #
    #                       , [".", "6", ".", ".", ".", ".", "2", "8", "."]
    #                       , [".", ".", ".", "4", "1", "9", ".", ".", "5"]
    #                       , [".", ".", ".", ".", "8", ".", ".", "7", "9"]]
    #                   )
    # ma.countAndSay_38(10)
    # ma.combinationSum_39([3, 5, 2], 8)
    # ma.combinationSum_39_LookBack([3, 5, 2], 8)
    # ma.combinationSum_39_LookBack_Winnow([3, 5, 2], 8)
    # ma.combinationSumII_40([10, 1, 2, 7, 6, 1, 5], 8)
    # ma.multiplyStrings_43('123', '456')
    # ma.jumpGameII_45([2, 3, 1, 1, 4])
    # ma.permutations_46([2, 3, 1, 4])
    # ma.AddTwoNumbers_2([9,9,9,9,9,9,9], [9,9,9,9])
    # ma.sum4_18([2,2,2,2,2],8)
    # ma.removeNthNodeFromEndOfList([1,2,3,4,5], 2)
    # ma.swapNodesInPairs_24([1,2,3,4,5,6])
    # ma.permutationsII_47( [1,1,2,2])
    # ma.rotateImage_48([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
    # ma.rotateImage_48([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]])
    # ma.groupAnagrams_49(["eat", "tea", "tan", "ate","eat", "nat", "bat"])
    # ma.pow_50(2,10)
    ma.maximumSubarray_53([-2,1,-3,4,-1,2,1,-5,4])