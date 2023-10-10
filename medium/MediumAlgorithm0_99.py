"""
    力扣算法题Python实践：https://leetcode.cn/problemset/algorithms/，可用于中学编程教学
    DATE        AUTHOR        CONTENTS
    2023-08-23  Jerry Chang   Create
"""
import math
import re
from typing import Tuple
import operator


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
        for i in results:
            if results.count(i) > 1:
                results.remove(i)
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
    ma.combinationSum_39_LookBack_Winnow([3, 5, 2], 8)
