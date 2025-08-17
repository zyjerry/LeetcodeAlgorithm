"""
    力扣算法题Python实践：https://leetcode.cn/problemset/algorithms/，可用于中学编程教学
    DATE        AUTHOR        CONTENTS
    2023-08-23  Jerry Chang   Create
"""
import math
import re
from sys import flags
from typing import Tuple

# 定义一个二叉树的结构，用户后续关于树的算法
class BinaryTreeNode:

    # 初始化节点结构
    def __init__(self,v:int=0,l=None,r=None):
        self.val = v         # 节点本身的值
        self.left = l        # 节点左子树节点的指针
        self.right = r       # 节点右子树节点的指针
        self.leftmin = v
        self.leftmax = v
        self.rightmin = v
        self.rightmax = v

    # 前序深度遍历该节点下的所有树结构，并返回一个所有元素的val列表。根-左-右
    def DLRTraversal(self,l:list)->list[int]:
        l.append(self.val)
        # 叶子节点，minvalue和maxvalue都是自己
        if self.left == None :
            if self.right != None:
                l.append(None)
        else:
            self.left.DLRTraversal(l)

        if self.right != None:
            self.right.DLRTraversal(l)

        return l

    # 中序深度遍历该节点下的所有树结构，并返回一个所有元素的val列表。左-根-右
    def LDRTraversal(self,l:list)->list[int]:
        if self.left == None:
            l.append(self.val)
        else:
            self.left.LDRTraversal(l)
            l.append(self.val)
        if self.right != None:
            self.right.LDRTraversal(l)

        return l

    # 后序深度遍历该节点下的所有树结构，并返回一个所有元素的val列表。左-右-根
    def LRDTraversal(self,l:list)->list[int]:
        if self.left != None:
            self.left.LRDTraversal(l)
        if self.right != None:
            self.right.LRDTraversal(l)
        l.append(self.val)

        return l

    # 后序深度遍历该节点下的所有树结构，并计算每个节点下左子树的最大最小值、右子树的最大最小值。
    # 返回False就是False，返回None表示True，另返回当前存在偏差的节点和
    def validateBinarySearchTree(self):
        if self.left == None and self.right == None:
            self.leftmin = self.val
            self.leftmax = self.val
            self.rightmin = self.val
            self.rightmax = self.val

        if self.left != None:
            self.left.validateBinarySearchTree()
        if self.right != None:
            self.right.validateBinarySearchTree()

        if self.left != None:
            self.leftmin = min(self.left.leftmin, self.left.leftmax, self.left.rightmin, self.left.rightmax, self.left.val)
            self.leftmax = max(self.left.leftmin, self.left.leftmax, self.left.rightmin, self.left.rightmax, self.left.val)
        if self.right != None:
            self.rightmin = min(self.right.leftmin, self.right.leftmax, self.right.rightmin, self.right.rightmax, self.right.val)
            self.rightmax = max(self.right.leftmin, self.right.leftmax, self.right.rightmin, self.right.rightmax, self.right.val)

        if not self.leftmax < self.val < self.rightmin:
            return False,self


    # 广度遍历该节点下的所有树结构，并返回一个元素列表
    def breadthFirstTraversal(self)->list[int]:
        queue = [self]
        l=[]
        while queue:
            n = len(queue)
            # print('n',n,'queue',queue)
            for i in range(n):
                q = queue.pop(0)
                l.append(q.val if q else None)
                if q:
                    queue.append(q.left if q.left else None)
                    queue.append(q.right if q.right else None)
        # 把最终列表中末尾的None去掉
        while l[-1] == None:
            l.pop(-1)
        return l

# 主类，算法实现都在这里面
class MediumAlgorithm0_99:
    """    构造函数，什么都不做    """

    def __init__(self):
        print('Hello World! I''m MediumAlgorithm0_99''')

    """    小工具，把广度遍历排列的list格式改成二叉树BinaryTreeNode的格式，初始化该数据结构，用于后续跟树相关的算法    """

    def initiateBinaryTreeFromList(self,root:list) -> BinaryTreeNode:
        # 先把list格式改成二叉树BinaryTreeNode的格式，初始化该数据结构
        rootbtn = BinaryTreeNode(root[0], None, None)
        poplist = [rootbtn]
        i = 1
        while poplist != [] or i < len(root):
            currentrootbtn = poplist[0]
            if i < len(root) and root[i] != None:
                leftbtn = BinaryTreeNode(root[i],None,None)
            else:
                leftbtn = None
            currentrootbtn.left=leftbtn
            i = i + 1

            if i < len(root) and root[i] != None:
                rightbtn = BinaryTreeNode(root[i],None,None)
            else:
                rightbtn = None
            currentrootbtn.right = rightbtn
            i = i + 1

            if leftbtn != None:
                poplist.append(leftbtn)
            if rightbtn != None:
                poplist.append(rightbtn)

            poplist.pop(0)

        return rootbtn

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

    """
    54. 螺旋矩阵：给你一个 m 行 n 列的矩阵 matrix ，请按照 顺时针螺旋顺序 ，返回矩阵中的所有元素。
        标签：数组，矩阵，模拟
        https://leetcode.cn/problems/spiral-matrix/description/
    """

    def  spiralMatrix_54(self, matrix:list[list[int]]) -> list[list[int]]:
        # 思路1：设计一个递归函数，参数是矩阵，执行结果是把该矩阵最外圈元素按顺序推进结果列表中
        #      再递归执行内圈的矩阵，直至矩阵为单维
        resultList = []

        def recursion(matrixr:list[list[int]]):
            rows, cols = len(matrixr), len(matrixr[0])

            # 执行到内圈，有五种情况，分别判断，这样写法有点蠢：
            # 情况一：空矩阵，针对n*n维正方形矩阵的最后一轮，此时无可争议，直接结束
            if rows ==0 and cols == 0:
                return
            # 情况二：1*1维矩阵，此时无可争议，仅把这一个元素加入结果列表，结束
            if rows ==1 and cols == 1:
                resultList.append(matrixr[0][0])
            # 情况三：n*1维矩阵，单列，此时顺序方向一定是自上而下，依次遍历把元素加入结果列表，结束
            elif rows > 1 and cols == 1:
                column = [row[0] for row in matrixr]
                resultList.extend(column)
            # 情况四：1*n维矩阵，单行，此时顺序方向一定是自左而右，依次遍历把元素加入结果列表，结束
            elif rows == 1 and cols > 1:
                resultList.extend(matrixr[0])
            # 情况五：m*n维矩阵，此时先把该矩阵外圈按顺序加入结果列表，再基于内圈的新矩阵，继续递归计算
            else:
                # 矩阵第一行
                resultList.extend(matrixr[0])
                # 矩阵最右列
                column = [row[cols-1] for row in matrixr]
                column.pop(0)
                resultList.extend(column)
                # 矩阵最下行，从右向左推入结果列表
                for k in range(cols-2, -1, -1):
                    resultList.append(matrixr[rows-1][k])
                # 矩阵最左列，从下而上推入结果列表
                for l in range(rows-2, 0, -1):
                    resultList.append(matrixr[l][0])
                # 递归执行内圈矩阵,由于list不支持直接取子矩阵，只能手工拼一下
                tmpmatrix = []
                for i in range(1,rows-1):
                    tmpmatrix.append(matrixr[i][1:cols-1])
                recursion(tmpmatrix)

        recursion(matrix)
        print(resultList)
        return resultList

    def spiralMatrix_54_standard(self, matrix:list[list[int]]) -> list[list[int]]:
        # 思路2：官网解答方法一，这个比较优雅高效，但是思路有点绕，想了好一会儿才想通
        # 首先初始化一个同样大小的矩阵，存储每个元素是否已被处理，默认为False
        rows, columns = len(matrix), len(matrix[0])
        flagmatrix = [[False] * columns for _ in range(rows)]
        # 其次解决顺时针绕圈的逻辑，观察可以发现规律：总共就是依次向左、向下、向右、向上不停循环
        # 我们用一个4*2的数组记录“下一个元素的坐标应该向哪里走”，里面的每一对值分别表示元素坐标的增量
        # 向左[0, 1]、向下[1, 0]、向右[0, -1]、向上[-1, 0]
        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        # directionIndex表示当前应该调转为哪个方向，初始化为0向左，对应directions[0]，它总是在0/1/2/3中循环
        directionIndex = 0
        resultList = []
        # x,y记录每个循环内当前要处理的元素的坐标，初始化为0，0
        x, y = 0, 0
        for _ in range(rows * columns):
            print('当前坐标：',x,y,'当前方向：',directionIndex,directions[directionIndex])
            # 把元素加入结果列表
            resultList.append(matrix[x][y])
            # 把对应坐标标记为“已处理”
            flagmatrix[x][y] = True
            # 判断是否要调转方向，条件满足下述任一就要调转：
            # 1、当前方向的下一个元素已被处理
            # 2、当前方向的下一个元素所在x、y坐标超出矩阵范围
            nextrow, nextcol = x + directions[directionIndex][0], y + directions[directionIndex][1]
            if  not ( (0<= nextrow < rows) and (0 <= nextcol < columns) and not flagmatrix[nextrow][nextcol]):
                directionIndex = (directionIndex + 1) % 4
                print('换方向：',directionIndex,directions[directionIndex])
            # 确定下一个待处理元素坐标
            x, y = x + directions[directionIndex][0], y + directions[directionIndex][1]

        print(resultList)
        return resultList

    """
    55. 跳跃游戏：给你一个非负整数数组 nums ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。
        判断你是否能够到达最后一个下标，如果可以，返回 true ；否则，返回 false 。
        示例 1：输入：nums = [2,3,1,1,4],输出：true.
               解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
        标签：贪心，数组，动态规划
        https://leetcode.cn/problems/jump-game/description/
    """

    def jumpGame_55(self, nums: list[int]) -> bool:
        # 思路1：从0开始逐步向右，每一步判断所有可能跳跃到的位置，把位置存储在一个列表里
        # 每一步同时也判断列表是否包含最末的位置，如果有，说明可以跳到，立刻停止返回True
        # 如果列表中所有元素都判断完了仍没有包含最末的位置，说明没有，返回False
        tmpList = [0]
        flag = False
        while tmpList != [] and tmpList[0]< len(nums):
            print(tmpList[0],tmpList)
            i = tmpList[0]
            tmpList.pop(0)
            for j in range(1, nums[i]+1):
                tmpList.append(i+j)
            print(tmpList)
            if (len(nums)-1) in tmpList:
                flag = True
                break
        print('flag:', flag)
        # return flag

        # 思路2：官方解答，比思路1简单一些。
        # 对于每一个可以到达的位置x，它使得 x+1,x+2,⋯,x+nums[x] 这些连续的位置都可以到达。
        # 我们依次遍历数组中的每一个位置，并实时维护 最远可以到达的位置，
        # 如果 最远可以到达的位置 大于等于数组中的最后一个位置，那就说明最后一个位置可达，我们就可以直接返回 True 作为答案
        maxindex = 0
        for i in range(len(nums)):
            # 为什么要加这个判断：当i超过maxindex时，说明i坐标的元素根本不可到达，那就没必要继续判断了
            if i <= maxindex:
                maxindex = max(maxindex, i+nums[i])
                print('i',i,'maxindex',maxindex)
                if maxindex>=len(nums)-1:
                    print('True')
                    return True
            else:
                print('False')
                return False
        return False

    """
    56. 合并区间：以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。
                请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。
        示例 1：输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
               输出：[[1,6],[8,10],[15,18]]
               解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
        标签：数组，排序
        https://leetcode.cn/problems/merge-intervals/description/
    """

    def mergeIntervals_56(self, intervals: list[list[int]]) -> list[list[int]]:
        # 思路：考虑到数据区间不总都是向上的，存在波动的情况，首先要对数组的0坐标值排序，确保排序后区间保持向上
        # 然后再从左向右逐步判断合并
        result = intervals
        result.sort(key=lambda x: x[0])

        i = 0
        while i < len(result)-1:
            print(i,len(result),result)
            # 这里只针对第一个区间再第二个区间左边的情况，假设第一个区间在第二个区间右边就没法判断了
            if result[i+1][0] <= result[i][1]  and result[i+1][1] >= result[i][0]:
                l = [min(result[i][0],result[i+1][0]),max(result[i][1],result[i+1][1])]
                result.pop(i)
                result.pop(i)
                result.insert(i,l)
            else:
                i = i+1
        print(result)
        return result

    """
    57. 插入区间：给你一个 无重叠的 ，按照区间起始端点排序的区间列表 intervals，
               其中 intervals[i] = [starti, endi] 表示第 i 个区间的开始和结束，并且 intervals 按照 starti 升序排列。
               同样给定一个区间 newInterval = [start, end] 表示另一个区间的开始和结束。
               在 intervals 中插入区间 newInterval，使得 intervals 依然按照 starti 升序排列，且区间之间不重叠（如果有必要的话，可以合并区间）。
               返回插入之后的 intervals。注意 你不需要原地修改 intervals。你可以创建一个新数组然后返回它。
        示例 1：输入：intervals = [[1,3],[6,9]], newInterval = [2,5]，输出：[[1,5],[6,9]]
        标签：数组
        https://leetcode.cn/problems/insert-interval/description/
    """

    def insertIntervals_57(self, intervals: list[list[int]], newInterval:list[int,int]) -> list[list[int]]:
        # 思路：从前到后对已有列表的每个元素和newInterval对比判断，有几种情况：
        # 情况1、newInterval完全在当前元素的左边不搭界，那么在已有列表左边插入newInterval
        # 情况2、newInterval和当前元素有重叠，那么修改当前元素，变成和newInterval的并集，此时后续工作是判断当前元素和下一元素是否右重叠，如有则合并，一直到最后
        # 情况3、newInterval完全在当前元素的右边不搭界，那么循环到下一元素继续判断
        resultlist = intervals.copy()
        i = 0

        # 情况1、先判断最左边，直接插入
        if newInterval[0] <= newInterval[1] < resultlist[0][0] :
            resultlist.insert(0, newInterval)
            print(resultlist)
            return resultlist
        # 再循环遍历
        while i < len(intervals):
            # 情况2、newInterval和当前元素有重叠
            if not resultlist[i][1] < newInterval[0]:
                # 合并newInterval和当前元素
                resultlist[i][0] = min(resultlist[i][0],newInterval[0])
                resultlist[i][1] = max(resultlist[i][1], newInterval[1])
                # 循环合并后面有可能重叠的元素
                j = i+1
                while j < len(intervals):
                    if resultlist[i][1] >= resultlist[j][0]:
                        resultlist[i][1] = max(resultlist[i][1],resultlist[j][1])
                        del resultlist[j]
                    else:
                        j = j + 1
                break
            # 情况3、newInterval完全在当前元素的右边不搭界，在下一元素左边也不搭界，则插入这个newInterval
            elif (i < len(intervals)-1 and newInterval[0] <= newInterval[1] < resultlist[i+1][0]) \
                    or (i == len(intervals)-1 and resultlist[i][0] < newInterval[0]) :
                resultlist.insert(i+1, newInterval)
                break

            # 情况2、情况3都不是的话，进入下一个元素判断
            i = i + 1

        print(resultlist)
        return resultlist

    def insertIntervals_57_standard(self, intervals: list[list[int]], newInterval: list[int, int]) -> list[list[int]]:
        # 思路2：官网答案，比上面这个优雅一些。先遍历所有元素，找出和newInterval有重叠的，最后一起处理，删除合并元素，插入新区间
        resultlist = []
        left, right = newInterval
        flag = False
        for li, ri in intervals:
            # newInterval在当前元素左边，插入newInterval
            if right < li:
                # 这里要做个标记已经插入newInterval了，后续循环就不要再重复插入了
                if not flag:
                    resultlist.append([left, right])
                    flag = True
                resultlist.append([li,ri])
            # newInterval在当前元素右边，插入当前元素
            elif ri < left:
                resultlist.append([li,ri])
            # 有重叠，计算并集
            else:
                left = min(left,li)
                right = max(right,ri)
            print(li, ri, resultlist,left, right )

        # 最后再考虑newInterval在整个列表最右的情况
        if resultlist[len(resultlist) - 1][1] < left:
            resultlist.append([left, right])

        print(resultlist)
        return resultlist

    """
    59. 螺旋矩阵 II：给你一个正整数 n ，生成一个包含 1 到 n**2 所有元素，且元素按顺时针顺序螺旋排列的 n x n 正方形矩阵 matrix 。
        标签：数组，矩阵，模拟
        https://leetcode.cn/problems/spiral-matrix-ii/description/
    """

    def spiralMatrix_59(self, n: int) -> list[list[int]]:
        # 思路：参考spiralMatrix_54_standard，本题由于是正方形，并且只是填数而不是原地转置，相对容易一些
        # 首先初始化一个n*n的矩阵，既作为结果矩阵，也作为判断每个元素是否已被处理，填为0表示没处理
        resultmatrix = [[0] * n for _ in range(n)]
        # 其次解决顺时针绕圈的逻辑，观察可以发现规律：总共就是依次向左、向下、向右、向上不停循环
        # 我们用一个4*2的数组记录“下一个元素的坐标应该向哪里走”，里面的每一对值分别表示元素坐标的增量
        # 向左[0, 1]、向下[1, 0]、向右[0, -1]、向上[-1, 0]
        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        # directionIndex表示当前应该调转为哪个方向，初始化为0向左，对应directions[0]，它总是在0/1/2/3中循环
        directionIndex = 0
        # x,y记录每个循环内当前要处理的元素的坐标，初始化为0，0
        x, y = 0, 0
        for i in range(n*n):
            print('当前坐标：',x,y,'当前方向：',directionIndex,directions[directionIndex])
            # 把数字填入对应位置
            resultmatrix[x][y] = i+1
            # 判断是否要调转方向，条件满足下述任一就要调转：
            # 1、当前方向的下一个元素已被处理
            # 2、当前方向的下一个元素所在x、y坐标超出矩阵范围
            nextrow, nextcol = x + directions[directionIndex][0], y + directions[directionIndex][1]
            if  not ( (0<= nextrow < n) and (0 <= nextcol < n) and resultmatrix[nextrow][nextcol]==0):
                directionIndex = (directionIndex + 1) % 4
                print('换方向：',directionIndex,directions[directionIndex])
            # 确定下一个待处理元素坐标
            x, y = x + directions[directionIndex][0], y + directions[directionIndex][1]

        print(resultmatrix)
        return resultmatrix

    """
    61. 旋转链表：给你一个链表的头节点 head ，旋转链表，将链表每个节点向右移动 k 个位置。
        示例 1：输入：head = [1,2,3,4,5], k = 2，输出：[4,5,1,2,3]
        标签：链表，双指针
        https://leetcode.cn/problems/rotate-list/description/
    """

    def rotateList_61(self, head: list, k: int) -> list:
        # 思路1：简单粗暴不值一提，新生成一个链表，把原链表内容按照位移的新位置赋值
        k = k % len(head)
        resultlist = [0]*len(head)
        for i in range(len(head)):
            newindex = (i+k)%len(head)
            resultlist[newindex] = head[i]

        print(resultlist)
        # return resultlist

        # 思路2：原地旋转。

        # 本来想用蛙跳式轮询一遍，但是发现不行，当列表长度和k不能互质的时候，没法遍历到所有元素
        # i,tmp1,tmp2 = 0,head[0],0
        # for _ in range(len(head)):
        #     # 当前元素的目标坐标
        #     nexti = (i+k)%len(head)
        #     # 临时存储目标元素值
        #     tmp2 = head[nexti]
        #     # 目标元素值填为当前元素值
        #     head[nexti] = tmp1
        #     # tmp1赋值为目标元素值，用于下一轮循环作为被填入的内容
        #     tmp1 = tmp2
        #     i = nexti
        # print(head)
        # return head

        # 这样的话依然只能逐个轮询，一共循环k*n遍
        for _ in range(k):
            # 每遍把列表中每个元素向右移动一位
            tmp = head[len(head)-1]
            for i in range(len(head)-1,0,-1):
                head[i] = head[i-1]
            head[0] = tmp
        print(head)
        return head

        # 官网的标准解法是用纯链表数据结构体ListNode实现，思路是：先将给定的链表连接成环，然后将指定位置断开。这里就不用代码实现了。

    """
    62. 不同路径：一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
        机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。
        问总共有多少条不同的路径？
        示例 1：输入：m = 3, n = 7，输出：28
        标签：数学，动态规划，组合数学
        https://leetcode.cn/problems/unique-paths/description/
    """

    def uniquePaths_62(self, m: int, n: int) -> int:
        # 思路1：典型动态规划，用递归实现。当机器人处于坐标[x,y]时，他能到达目标的方法数=[x+1][y]方法数+[x][y+1]方法数
        def recursion(x:int,y:int) -> int:
            # 到达最末端返回1
            if (x==m and y==n-1) or (x==m-1 and y==n):
                return 1
            # 到达最下边缘返回x,y+1)
            elif (x==m and y<n-1):
                return recursion(x,y+1)
            # 到达最右边缘返回(x+1, y)
            elif (x<m-1 and y==n):
                return recursion(x+1, y)
            else:
                m1 = recursion(x+1,y)
                m2 = recursion(x,y+1)
                return m1+m2

        methods = recursion(1,1)
        print(methods)
        # return methods

        # 思路2：递归耗空间，不用递归，用小学奥数的标数法，从右下到左上逐步计算方法，一遍完成
        # 初始化一个(m+1)*(n+1)的矩阵，每个元素记录方法数，这里矩阵多加一个行列是为了便于计算
        resultlist =  [[0] * (n+1) for _ in range(m+1)]
        resultlist[m-1][n-2] = 1
        resultlist[m-2][n-1] = 1
        # 先把矩阵右下半角部分算完
        for y in range(n-3,-1,-1): #起始纵坐标y
            print('y:', y)
            j = 0
            while (m-1-j>=0) and (y+j)<n: #横纵坐标到达边界
                # 计算当前坐标值是下面和右面之和
                print('x,y',m-1-j, y+j)
                resultlist[m-1-j][y+j]= resultlist[m-j][y+j] + resultlist[m-1-j][y+j+1]
                j = j + 1
            print(resultlist)

        # 再算左上半部分
        for x in range(m-2,-1,-1): #起始纵坐标y
            print('x:', x)
            j = 0
            while (x-j>=0) : #纵标到达边界
                # 计算当前坐标值是下面和右面之和
                print('x,y',x-j, j)
                resultlist[x-j][j]= resultlist[x-j+1][j] + resultlist[x-j][j+1]
                j = j + 1
            print(resultlist)
        print(resultlist[0][0])
        # return resultlist[0][0]

        # 思路3：官网方法一，依然是动态规划。
        # 其实从左上角向右下角，和反方向是一样的，所以不用递归倒推。f(i,j)=f(i−1,j)+f(i,j−1)
        # 初始条件为 f(0,0)=1，最终的答案即为 f(m−1,n−1)。
        # 为了方便代码编写，我们可以将所有的 f(0,j) 以及 f(i,0) 都设置为边界条件，它们的值均为 1。
        resultlist = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]
        print(resultlist)
        for i in range(1,m):
            for j in range(1,n):
                resultlist[i][j] = resultlist[i-1][j] + resultlist[i][j-1]
        print(resultlist)
        # return resultlist[m-1][n-1]

        # 思路4：官网方法二，直接使用组合数学推论。
        # 从左上角到右下角的过程中，我们需要移动 m+n−2 次，其中有 m−1 次向下移动，n−1 次向右移动。
        # 因此路径的总数，就等于从 m+n−2 次移动中选择 m−1 次向下移动的方案数，即组合数：
        print(math.comb(m + n - 2, n - 1))
        return math.comb(m + n - 2, n - 1)

    """
    63. 不同路径 II：给定一个 m x n 的整数数组 grid。
        一个机器人初始位于 左上角（即 grid[0][0]）。机器人尝试移动到 右下角（即 grid[m - 1][n - 1]）。机器人每次只能向下或者向右移动一步。
        网格中的障碍物和空位置分别用 1 和 0 来表示。机器人的移动路径中不能包含 任何 有障碍物的方格。返回机器人能够到达右下角的不同路径数量。
        示例 1：输入：obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]，输出：2
        标签：数组，动态规划，矩阵
        https://leetcode.cn/problems/unique-paths-ii/description/
    """

    def uniquePathsII_63(self, obstacleGrid: list[list[int]]) -> int:
        # 思路1：参考上一题思路3，在循环中增加判断，当该元素为0时，递归函数返回0
        m,n = len(obstacleGrid),len(obstacleGrid[0])
        resultlist = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]
        for i in range(1,m):
            for j in range(1,n):
                if obstacleGrid[i][j] == 1:
                    resultlist[i][j] = 0
                else:
                    resultlist[i][j] = resultlist[i-1][j] + resultlist[i][j-1]
        print(resultlist)
        return resultlist[m-1][n-1]

    """
    64. 最小路径和：给定一个包含非负整数的 m x n 网格 grid ，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
        说明：每次只能向下或者向右移动一步。
        示例 1：输入：grid = [[1,3,1],[1,5,1],[4,2,1]]，输出：7，解释：因为路径 1→3→1→1→1 的总和最小。
        标签：数组，动态规划，矩阵
        https://leetcode.cn/problems/minimum-path-sum/description/
    """

    def minimumPathSum_64(self, grid: list[list[int]]) -> int:
        # 思路：动态规划，从左上到右下依次计算每个元素的数字和，计算完毕取最右下角元素数字即可
        # 由于每次只能向下或者向右，那么后续每个元素值只和左边和上面一行相关，所以可以精简数据结构，只保留上一行数列即可
        # 这个思路比官解简洁，参照62题官解方法一的最简形式
        # 初始化上一行数列
        uplist = []
        rows, columns = len(grid), len(grid[0])
        # 每个元素循环判断
        for i in range(0,rows):
            tmplist = []
            for j in range(0,columns):
                # 第一行、第一列特殊判断一下
                if i == 0 and j == 0:
                    tmplist.append(grid[i][j])
                elif i == 0 and j > 0:
                    tmplist.append(grid[i][j]+tmplist[j-1])
                elif j == 0 and i > 0:
                    tmplist.append(grid[i][j] + uplist[j])
                # 最后才是通用情况
                else:
                    #print(tmplist[j-1], uplist[i-1])
                    tmplist.append(min(grid[i][j]+tmplist[j-1], grid[i][j] + uplist[j]))
            uplist = tmplist.copy()
            print(uplist)

        print(uplist[columns-1])
        return uplist[columns-1]

    """
    71. 简化路径：给你一个字符串 path ，表示指向某一文件或目录的 Unix 风格 绝对路径 （以 '/' 开头），请你将其转化为 更加简洁的规范路径。
        标签：栈，字符串
        https://leetcode.cn/problems/simplify-path/description/
    """

    def simplifyPath_71(self, path: str) -> str:
        # 思路：将path以/为关键字切成多个子串列表，依次对于每个元素判断
        strlist = path.split('/')
        print(strlist)
        resultlist = []
        # 依次对于每个元素判断
        for i in strlist:
            # 如果当前元素是..，把resultlist上次已经追加元素吐出来
            if i == '..':
                resultlist.pop()
                resultlist.pop()
            # 把空白的、无效的.去掉，其余有效的追加进结果列表
            elif i !='' and i != '.':
                resultlist.append('/')
                resultlist.append(i)

        print(''.join(resultlist))
        return ''.join(resultlist)

    """
    72. 编辑距离：给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数。
        你可以对一个单词进行如下三种操作：插入一个字符、删除一个字符、替换一个字符
        示例 1：输入：word1 = "horse", word2 = "ros"，输出：3
        标签：字符串，动态规划
        https://leetcode.cn/problems/edit-distance/description/
    """

    def editDistance_72(self, word1: str, word2: str) -> int:
        # 思路：这题我不会……快速看官解，第一遍也没看明白。仔细看第二遍模糊明白。看评论很多人也懵了，先跳过吧。
        return 0

    """
    73. 矩阵置零：给定一个 m x n 的矩阵，如果一个元素为 0 ，则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。
        标签：数组，哈希表，矩阵
        https://leetcode.cn/problems/set-matrix-zeroes/description/
    """

    def setMatrixZeroes_73(self, matrix: list[list[int]]) -> None:
        # 思路：逻辑看上去简单，有个陷阱是怎样区分原始矩阵中的0和后来改的0，不能混淆，否则最后全矩阵都会变成0。
        # 另外主要是看怎么高效，但是我也没想好，先来一版简单粗暴的。
        # 先遍历矩阵元素，找到0所在的行和列，记在2个列表里，用set集合不重复，再根据set终记录的行列把矩阵对应元素改成0
        rowset,columnset = set(), set()
        for x in range(len(matrix)):
            for y in range(len(matrix[0])):
                if matrix[x][y] == 0:
                    rowset.add(x)
                    columnset.add(y)
        print(rowset,columnset)
        print(matrix)

        # 改行
        for i in rowset:
            for j in range(len(matrix[0])):
                matrix[i][j] = 0
        # 改列
        for i in columnset:
            for j in range(len(matrix)):
                matrix[j][i] = 0

        print(matrix)

    """
    74. 搜索二维矩阵：给你一个满足下述两条属性的 m x n 整数矩阵：
                       每行中的整数从左到右按非严格递增顺序排列。
                       每行的第一个整数大于前一行的最后一个整数。
                   给你一个整数 target ，如果 target 在矩阵中，返回 true ；否则，返回 false 。
        示例 1：输入：matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3，输出：true
        标签：数组，二分查找，矩阵
        https://leetcode.cn/problems/search-a-2d-matrix/description/
    """

    def searchA2dMatrix_74(self,matrix: list[list[int]],target:int) -> bool:
        # 思路：既然matrix内容已经是排序了的，那就是个变相的二分查找啊
        rows, columns = len(matrix), len(matrix[0])
        # 把矩阵当成一维列表展开，记录头尾序号
        begin,end = 0, rows*columns-1
        # 循环二分查找
        while begin < end:
            mid = (begin + end + 1)//2
            print('[',begin,end,']','mid:',mid,'mid//rows',mid//rows,'mid%rows',mid%rows,matrix[mid//rows][mid%rows])
            if matrix[mid//rows][mid%rows] == target:
                print('True')
                return True
            elif matrix[mid//rows][mid%rows] > target:
                end = (begin + end)//2
            elif matrix[mid//rows][mid%rows] < target:
                begin = (begin + end)//2
        print('False')
        return False

    """
    75. 颜色分类：给定一个包含红色、白色和蓝色、共 n 个元素的数组 nums ，原地 对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
                我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。必须在不使用库内置的 sort 函数的情况下解决这个问题。
        示例 1：输入：nums = [2,0,2,1,1,0]，输出：[0,0,1,1,2,2]
        标签：数组，双指针，排序
        https://leetcode.cn/problems/sort-colors/description/
    """
    def sortColors_75(self, nums: list[int]) -> None:
        # 思路：由于只有3种值，所以做个简单的排序：遍历每个列表元素，如果是0就抽出来插入第一位，如果是2就抽出来插入最末一位，遍历完即可
        # 过程中需要记录一下往后仍的次数，作为结束循环的判断
        i, count2 = 0,0
        while i < len(nums)-count2:
            print(i,count2)
            if nums[i] == 0:
                nums.pop(i)
                nums.insert(0,0)
                i = i + 1
            elif nums[i] == 2:
                nums.pop(i)
                nums.append(2)
                count2 = count2 + 1
            else:
                i = i + 1
            print(nums)

        print(nums)

    """
    77. 组合：给定两个整数 n 和 k，返回范围 [1, n] 中所有可能的 k 个数的组合。你可以按 任何顺序 返回答案。
        示例 1：输入：n = 4, k = 2，输出：[  [2,4],  [3,4],  [2,3],  [1,2],  [1,3],  [1,4],]
        标签：回溯
        https://leetcode.cn/problems/combinations/description/
    """

    def combinations_77(self, n: int, k: int) -> list:
        # 经典排列组合，关键是不需要算有多少种排列组合，而是需要给出结果，我们从k=1开始算：
        # 从n个数里取1个，有n种方法，然后从剩下来的书中再取一个，加入到原队列，如此一直到k
        # 先初始化只包含一个元素的队列
        resultlist = []
        for i in range(1,n+2-k):
          resultlist.append([i])
        print(resultlist)
        # 因为每个组合都有k个元素，且已初始化一个元素，所以循环k-1遍
        for i in range(k-1):
            # 初始化resultlist当前的长度，因为后面要append所以不能变化
            row = len(resultlist)
            # 针对当前resultlist里的每个元素，
            for j in range(row):
                # 因为都是向后追溯，没必要向前，否则就会出现重复组合，所以要从resultlist当前元素的组合的最后一位数开始即可，确保k不在resultlist[0]中
                for k in range(resultlist[0][len(resultlist[0])-1]+1,n+1):
                    tmpl = resultlist[0].copy()
                    #if k not in tmpl:
                    tmpl.append(k)
                    resultlist.append(tmpl)
                # 组合元素全部添加完后，把第一个元素抛弃掉，下一循环处理下一个元素
                resultlist.pop(0)
                print(resultlist)

        print(resultlist)
        return resultlist

        # 官解方法一递归实现的，方法二通过二进制表示选还是不选某个元素穷举所有组合情况，总体来说性能不比上述思路更好，就不实现了

    """
    78. 子集：给你一个整数数组 nums ，数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。
        解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
        示例 1：输入：nums = [1,2,3]，输出：[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
        标签：位运算，数组，回溯
        https://leetcode.cn/problems/subsets/description/
    """

    def subsets_78(self, nums: list[int]) -> list[list[int]]:
        # 思路：和77题有相似之处，可以直接循环调，这里用二进制的思路重写
        # 假如nums的长度是n，所有可能的组合，相当于所有n位二进制数的组合
        resultlist = []
        # 对所有n位二进制数判断
        for i in range(2 ** len(nums)):
            tmplist = []
            # 判断每一位是否应该纳入组合
            for j in range(len(nums)):
                if ((2 ** j) & i) >= 1:
                    tmplist.append(nums[len(nums)-j-1])
            resultlist.append(tmplist)

        print(resultlist)
        return resultlist

    """
    79. 单词搜索：给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中，返回 true ；否则，返回 false 。
        单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
        标签：深度优先搜索，数组，字符串，回溯，矩阵
        https://leetcode.cn/problems/word-search/description/
    """

    def wordSearch_79(self, board: list[list[str]], word: str) -> bool:
        # 思路：2个指针，分别记录矩阵当前要匹配的元素和单词当前要匹配的字母；另设一个列表记录已匹配到的矩阵元素坐标防止重复使用
        # 从矩阵最左上角开始遍历，从word第一个字符开始逐个匹配，如果能匹配完到word最后一个字符则返回True

        flag = False
        vistedlist = []
        directions = [[-1,0],[0,1],[1,0],[0,-1]] # 上右下左的方向

        # tobematched:上个元素上下左右的待匹配坐标列表；subword：待匹配的子串
        def recursion(tobematched: list[list[int]], subword: str) -> bool:
            print('递归函数入口：',tobematched,subword)
            if subword == '':
                flag = True
                return flag
            else:
                for i in tobematched:
                    # print('待判断矩阵元素坐标',tobematched,'当前矩阵坐标',i,'已走通路径',vistedlist,'待判断子串',subword)
                    if board[i[0]][i[1]] == subword[0]:
                        print('待判断矩阵元素坐标', tobematched, '当前矩阵坐标', i, '已走通路径', vistedlist,
                              '待判断子串', subword)
                        # 将匹配到的坐标放入路径
                        vistedlist.append(i)
                        # 把下一个待匹配的（当前元素上下左右）放入tmplist，进入下一轮匹配
                        tmplist = []
                        # 把矩阵当前元素上下左右4个元素加入下一轮判断队列，剔除已经访问过的
                        for x,y in directions:
                            if 0<=i[0]+x<len(board) and 0<=i[1]+y<len(board[0]) and [i[0]+x,i[1]+y] not in vistedlist:
                                tmplist.append([i[0]+x,i[1]+y] )
                        # 递归调用下一个待匹配元素和subword子串
                        subsubword = subword[1:len(subword)]
                        flag = recursion(tmplist, subsubword)
                        print('flag',flag)
                        if flag:
                            return True
                flag = False
                return flag

        for i in range(len(board)*len(board[0])):
            tbm = [[i//len(board[0]), i%len(board[0])]]
            vistedlist = []
            word1 = word
            flag = recursion(tbm,word1)
            if flag:
                break

        print(flag)
        return flag

    """
    80. 删除有序数组中的重复项 II：给你一个有序数组 nums ，请你 原地 删除重复出现的元素，使得出现次数超过两次的元素只出现两次 ，返回删除后数组的新长度。
        不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。
        示例 1：输入：nums = [1,1,1,2,2,3]，输出：5, nums = [1,1,2,2,3]
        标签：数组，双指针
        https://leetcode.cn/problems/remove-duplicates-from-sorted-array-ii/description/
    """

    def removeDuplicates_80(self, nums: list[int]) -> int:
        # 思路：遍历数组，判断当前元素是否跟下面两个相同，是的话直接删
        i = 0
        while i < len(nums)-2:
            if nums[i] == nums[i+1] == nums[i+2]:
                nums.pop(i)
            else:
                i = i + 1
        print(len(nums), nums)
        return len(nums)

    """
    81. 搜索旋转排序数组 II：已知存在一个按非降序排列的整数数组 nums ，数组中的值不必互不相同。
        在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转 ，
        使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。
        例如， [0,1,2,4,4,4,5,6,6,7] 在下标 5 处经旋转后可能变为 [4,5,6,6,7,0,1,2,4,4] 。
        给你 旋转后 的数组 nums 和一个整数 target ，请你编写一个函数来判断给定的目标值是否存在于数组中。
        如果 nums 中存在这个目标值 target ，则返回 true ，否则返回 false 。你必须尽可能减少整个操作步骤。
        标签：数组，二分查找
        https://leetcode.cn/problems/search-in-rotated-sorted-array-ii/description/
    """

    def searchInRotatedSortedArray_81(self, nums: list[int], target: int) -> bool:
        # 先定位数组在什么位置“断掉”的，然后分别对两条子序列做二分查找。
        #
        # if nums[0] == nums[len(nums)-1]:
        #     if nums[0] == target:
        #         print(True)
        #         return True
        #     else:
        #         print(False)
        #         return False
        # 此题与33题相似，但本题中nums可能包含重复元素

        # 如果不是极端情况，那必然存在中间一个“断掉”的位置，对于极端情况数组所有元素都一样，那idx必然等于len(nums)-1，也不影响后续判断
        idx = 0
        for i in range(len(nums)-1):
            if nums[i] > nums[i+1]:
                idx = i
                break
        print('idx',idx)

        # 先定位target在哪一段
        begin,end = 0,len(nums)-1
        if nums[0] <= target <= nums[idx]:
            begin, end = 0, idx
        elif nums[idx+1] <= target <= nums[len(nums)-1]:
            begin, end = idx+1, len(nums)-1
        else:
            print(False)
            return False

        # 进行二分法查找
        while begin < end:
            mid = (begin + end) // 2
            print('begin,end,mid', begin, end, mid)
            if nums[mid] == target or nums[begin] == target or nums[end] == target:
                print(True)
                return True
            elif nums[mid] < target:
                begin = mid
            elif nums[mid] > target:
                end = mid
            # 当begin和end之间只差1时，也没必要循环了
            if begin + 1 == end :
                if (nums[begin] == target or nums[end] == target):
                    print(True)
                    return True
                else:
                    print(False)
                    return False

        print(False)
        return False
        # 这题官解也是搞笑，干脆就轮询一遍得了，并没有很精妙

    """
    82. 删除排序链表中的重复元素 II：给定一个已排序的链表的头 head ， 删除原始链表中所有重复数字的节点，只留下不同的数字 。返回 已排序的链表 。
        示例 1：输入：head = [1,2,3,3,4,4,5]，输出：[1,2,5]
        标签：链表，双指针
        https://leetcode.cn/problems/remove-duplicates-from-sorted-list-ii/description/
    """

    def removeDuplicates_82(self, head: list[int]) -> list:
        # 思路：轮询列表，每个循环中判断后面的元素是否连续相等，记录起止位置，删掉
        i = 0
        while i < len(head)-1:
            end = i+1
            while head[end] == head[i]:
                end = end+1
            print(i,end)
            if i == end-1:
                i = i + 1
            else:
                for j in range(i, end):
                    head.pop(i)

        print(head)
        return head

    """
    86. 分隔链表：给你一个链表的头节点 head 和一个特定值 x ，请你对链表进行分隔，使得所有 小于 x 的节点都出现在 大于或等于 x 的节点之前。
        你应当 保留 两个分区中每个节点的初始相对位置。
        示例 1：输入：head = [1,4,3,2,5,2], x = 3，输出：[1,2,2,4,3,5]
        标签：链表，双指针
        https://leetcode.cn/problems/partition-list/description/
    """

    def partitionList_86(self, head: list, x:int) -> list:
        # 思路：设立begin、end两个坐标指针，begin表示下个下雨x的插入位置，end遍历列表
        begin, end = 0, 1
        while end < len(head):
            print('before:', begin, end, head)
            # 兼容0坐标，当begin小于x的话，begin右移一位
            if head[begin] < x:
                begin += 1
            # 当end小于x的话，把end位置的元素插到begin位置，原end位置的元素剔除，begin和end均右移一位
            if head[end] < x:
                head.insert(begin, head[end])
                head.pop(end+1)
                end += 1
                begin += 1
            # 其余情况（也就是当前end值大于x，不做调整）仅对end右移一位
            else:
                end += 1
            print('after:', begin, end, head)

        print(head)
        return head

    """
    89. 格雷编码：n 位格雷码序列 是一个由 2**n 个整数组成的序列，其中：
                每个整数都在范围 [0, 2n - 1] 内（含 0 和 2n - 1）；第一个整数是 0；一个整数在序列中出现 不超过一次；
                每对 相邻 整数的二进制表示 恰好一位不同 ，且 第一个 和 最后一个 整数的二进制表示 恰好一位不同
        给你一个整数 n ，返回任一有效的 n 位格雷码序列 。
        示例 1：输入：n = 2，输出：[0,1,3,2]
        标签：位运算，数学，回溯
        https://leetcode.cn/problems/gray-code/description/
    """

    def grayCode_89(self, n: int) -> list[int]:
        # 本来的思路是：初始化两个列表，一个是0-2**n-的所有数字，一个是最终列表，头尾2个元素，初始化固定为0和1，
        #            其他的遍历初始列表，符合条件的就逐个往最终列表里插
        # 实践证明这个思路是错的，当n>=5时，最终初始列表中总会剩余若干元素不符合条件插入最终列表
        originallist = list(range(2,2**n))
        resultlist = [0,1]
        for i in range(len(originallist)):
            for i in originallist:
                # 把原始列表中某个元素和结果列表倒数第二个比较，如果符合条件，就把该元素插入结果列表倒数第二位，同时从原始列表中删除
                x = i ^ (resultlist[-2])
                # 两个二进制数恰好有一位不同，说明这两个数异或操作后，一定是个2的次方数，以此作为判断条件
                if (x & (x-1)) == 0:
                    resultlist.insert(-1, i)
                    originallist.remove(i)

        print('错误思路resultlist：',resultlist)
        print('错误思路originallist：',originallist)
        #return resultlist

        # 正确思路：格雷编码生成是有明确方式和证明的。
        # n+1位格雷码的集合 = n位格雷码集合(顺序)加前缀0 + n位格雷码集合(逆序)加前缀1
        resultlist = [0]
        for i in range(1, n + 1):
            for j in range(len(resultlist) - 1, -1, -1):
                # 1向左位移i-1位，再和resultlist[j]进行或操作，相当于加前缀1
                resultlist.append(resultlist[j] | (1 << (i - 1)))
            print(resultlist)
        print('正确思路resultlist：',resultlist)
        return resultlist

    """
    90. 子集 II：给你一个整数数组 nums ，其中可能包含重复元素，请你返回该数组所有可能的 子集（幂集）。
                解集 不能 包含重复的子集。返回的解集中，子集可以按 任意顺序 排列。
        示例 1：输入：nums = [1,2,2]，输出：[[],[1],[1,2],[1,2,2],[2],[2,2]]
        标签：位运算，数组，回溯
        https://leetcode.cn/problems/subsets-ii/description/
    """

    def susetsII_90(self, nums: list[int]) -> list[list[int]]:
        # 思路：和78题有相似之处，只是78不包含重复元素，本题可以，这里用二进制的思路重写，加一个判断避免重复
        # 假如nums的长度是n，所有可能的组合，相当于所有n位二进制数的组合
        resultlist = []
        # 对所有n位二进制数判断
        for i in range(2 ** len(nums)):
            tmplist = []
            # 判断每一位是否应该纳入组合
            for j in range(len(nums)):
                if ((2 ** j) & i) >= 1:
                    tmplist.append(nums[len(nums)-j-1])
            if tmplist not in resultlist:
                resultlist.append(tmplist)

        print(resultlist)
        return resultlist

    """
    91. 解码方法：一条包含字母 A-Z 的消息通过以下映射进行了 编码 ："1" -> 'A'
                                                        "2" -> 'B'
                                                        ...
                                                        "25" -> 'Y'
                                                        "26" -> 'Z'
                然而，在 解码 已编码的消息时，你意识到有许多不同的方式来解码，因为有些编码被包含在其它编码当中（"2" 和 "5" 与 "25"）。
                给你一个只含数字的 非空 字符串 s ，请计算并返回 解码 方法的 总数 。如果没有合法的方式解码整个字符串，返回 0。
        示例 1：输入：s = "12"，输出：2，解释：它可以解码为 "AB"（1 2）或者 "L"（12）。
        标签：字符串，动态规划
        https://leetcode.cn/problems/decode-ways/description/
    """

    def decodeWays_91(self, nums: str) -> int:
        # 思路：动态规划，先判断字符串头一位或者头两位，是否在1-25范围内，如是，方法数相当于后面的子串方法数，可以用递归实现
        # 这个思路是对的，但是代码写得比较蠢，边界条件if else写得太多了，相对来说，官解更优雅写
        def recursion(substr:str) ->int:
            print(substr)
            # 如果str是空的或者第一位为0，返回0
            if substr == '' or substr[0] == '0':
                return 0

            # 如果str就剩1位了，返回1
            if len(substr) == 1 :
                return 1

            # 如果str就剩2位了，看情况
            if len(substr) == 2 :
                # 情况1、如果该两位>26，那只能看后面情况
                if int(substr[0:2]) > 26:
                    return recursion(substr[1:])
                # 情况2：如果该两位<27，那可以取第一位也可以取前两位，再看后面情况
                elif int(substr[0:2]) < 27:
                    return 1 + recursion(substr[1:])

            # 如果str大于2位，看str前2位数的情况
            if len(substr)>2:
                # 情况1、如果前两位>26，那只能取第一位，再看后面情况
                if int(substr[0:2]) > 26:
                    return recursion(substr[1:])
                # 情况2：如果前两位<27，那可以取第一位也可以取前两位，再看后面情况
                elif int(substr[0:2])<27:
                    return  recursion(substr[1:]) + recursion(substr[2:])

        a = recursion(nums)
        print(a)
        # return a

        # 官解是从前往后推，不像递归那么别扭：设 fi表示字符串 s 的前 i 个字符 s[1..i] 的解码方法数，那么有两种情况：
        # 情况一、使用了一个字符，即 s[i] 进行解码，那么只要 s[i] !=0，它就可以被解码成 A∼I 中的某个字母，状态转移方程fi=f i−1
        # 情况二、使用了两个字符，即 s[i−1] 和 s[i] 进行编码，s[i−1] 不能等于 0，并且 s[i−1] 和 s[i] 组成的整数必须小于等于 26，这样它们就可以被解码成 J∼Z 中的某个字母。
        #        状态转移方程：fi=f i−2
        # 需要注意的是，只有当 i>1 时才能进行转移，否则 s[i−1] 不存在，边界条件为：f0=1
        n = len(nums)
        f = [1] + [0] * n
        for i in range(1, n + 1):
            if nums[i - 1] != '0':
                f[i] += f[i - 1]
            if i > 1 and nums[i - 2] != '0' and int(nums[i-2:i]) <= 26:
                f[i] += f[i - 2]
        print(f)
        return f[n]

    """
    92. 反转链表 II：给你单链表的头指针 head 和两个整数 left 和 right ，其中 left <= right 。请你反转从位置 left 到位置 right 的链表节点，返回 反转后的链表 。
        示例 1：输入：head = [1,2,3,4,5], left = 2, right = 4，输出：[1,4,3,2,5]
        标签：链表
        https://leetcode.cn/problems/reverse-linked-list-ii/description/
    """

    def reverseLinkedListII_92(self, head: list, left:int, right:int) -> list:
        # 思路：直接原地交换吧
        while left < right:
            tmp = head[left-1]
            head[left-1] = head[right-1]
            head[right-1] = tmp
            left += 1
            right -= 1
        print(head)
        return head

        # 官解的要求是，链表的操作问题，一般而言不允许我们修改节点的值，而只能修改节点的指向操作。这样的话就会复杂一些，先不写了

    """
    93. 复原 IP 地址：有效 IP 地址 正好由四个整数（每个整数位于 0 到 255 之间组成，且不能含有前导 0），整数之间用 '.' 分隔。
                    给定一个只包含数字的字符串 s ，用以表示一个 IP 地址，返回所有可能的有效 IP 地址，这些地址可以通过在 s 中插入 '.' 来形成。
                    你 不能 重新排序或删除 s 中的任何数字。你可以按 任何 顺序返回答案。
        示例 1：输入：s = "25525511135"，输出：["255.255.11.135","255.255.111.35"]
        标签：字符串，回溯
        https://leetcode.cn/problems/restore-ip-addresses/description/
    """

    def restoreIpAddresses_93(self, s: str) -> list:
        # 思路：有点像91题，用递归的方式穷尽所有可能的255以内数字的4个组合，应该算是深度遍历搜索
        resultlist = []
        # sub1是已经匹配模式的字符串，subs2是剩余待分割的字符串
        def recursion(subs1:str, subs2:str):
            print(subs1,subs2,resultlist)
            # 如果待分割的字符串是空的说明匹配完毕，返回
            if subs2=='' :
                # 只有当已经匹配模式的字符串，格式符合要求（4段符合要求的数字）时，才认为是正确的IP地址，加入到结果列表中，需要做除重判断
                if subs1.count('.')==4 and subs1[0:len(s)+3] not in resultlist:
                    resultlist.append(subs1[0:len(s)+3])
                return
            # 如果待匹配字符串已经超出4个点了说明格式不对，也返回，这里节省不必要的深度递归递归
            if subs1.count('.') > 4:
                return

            # substr的第一个字符无论如何都能符合条件，加入
            recursion(subs1+subs2[0]+'.', subs2[1:])
            # 如果substr的第一个字符不等于0 ，那么前两个也符合条件，加入
            if subs2[0]!='0':
                recursion(subs1+subs2[0:2]+'.', subs2[2:])
                # 进而，如果substr的前3个字符<256，那么前三个也符合条件，加入
                if int(subs2[0:3])<256:
                    recursion(subs1+subs2[0:3]+'.', subs2[3:])

        recursion('',s)

        print(len(resultlist),resultlist)
        return resultlist

    """
    95. 不同的二叉搜索树 II：给你一个整数 n ，请你生成并返回所有由 n 个节点组成且节点值从 1 到 n 互不相同的不同 二叉搜索树 。可以按 任意顺序 返回答案。
        示例 1：输入：n = 3，输出：[[1,null,2,null,3],[1,null,3,2],[2,1,3],[3,1,null,null,2],[3,2,null,1]]
        标签：树，二叉搜索树，动态规划，回溯，二叉树
        https://leetcode.cn/problems/unique-binary-search-trees-ii/description/
    """

    def uniqueBinarySearchTreeII_95(self, nums: int) -> list:
        # 二叉搜索树的定义是：左子树的所有节点值均小于根节点的值，右子树的所有节点值均大于根节点的值，左右子树也分别为二叉搜索树。
        # 思路：跟93题差不多，设计一个递归函数，参数是当前已安置节点和待安置的数字队列，把待安置数字队列里的元素分成大于当前节点和小于当前节点两部分，再递归执行左右两部分，直到待安置的数字队列为空
        # 上述思路尝试半天还是写不出来，以下代码作废。
        resultlist = []
        originlist = list(range(1,nums+1))

        # def recursion(processedlist:list[int], remainedlist:list[int], count:int) -> None:
        #     print('processedlist', processedlist, 'remainedlist', remainedlist, 'count', count)
        #
        #     # 当计数器count为0表示整个二叉搜索树都构建完毕，加入resultlist，返回
        #     if count == 0:
        #         resultlist.append(processedlist.copy())
        #         return
        #     # 当待安置数字队列都没了时，结束递归，返回，但这个时候不一定是整个二叉搜索树都构建完毕，所以不必加入resultlist
        #     if remainedlist==[] :
        #         return
        #
        #     # 把remainedlist切分成左子树和右子树
        #     i = 0
        #     while i < len(remainedlist):
        #         if remainedlist[i] < processedlist[-1]:
        #             i = i + 1
        #     leftlist,rightlist = remainedlist[0:i], remainedlist[i:len(remainedlist)]
        #     # 如果左子树为空但右子树还有内容，需要补个0
        #     if leftlist==[] and rightlist!=[]:
        #         processedlist.append(0)
        #     # 如果左子树还有内容，则优先遍历左子树
        #     if leftlist != [] :
        #         for i in leftlist:
        #             ll = leftlist.copy()
        #             ll.remove(i)
        #             pp = processedlist.copy()
        #             pp.append(i)
        #             print('  pp,ll', pp,ll)
        #             recursion(pp, ll,count-1)
        #     # 如果右子树还有内容，则优先遍历左子树
        #     if rightlist != []:
        #         for i in rightlist:
        #             rr = rightlist.copy()
        #             rr.remove(i)
        #             pp = processedlist.copy()
        #             pp.append(i)
        #             print('  pp,rr', pp, rr)
        #             recursion(pp, rr,count-1)
        #
        # recursion([], originlist, [])
        #
        # print(len(resultlist),resultlist)
        # return resultlist


        # 参考官网解答，发现树状的数据结构和算法，还是构造一个树节点TreeNode存储左右子树比较方便清楚
        def generateTrees(start, end):
            if start > end:
                return [None, ]

            allTrees = []
            for i in range(start, end + 1):  # 枚举可行根节点
                # 获得所有可行的左子树集合
                leftTrees = generateTrees(start, i - 1)

                # 获得所有可行的右子树集合
                rightTrees = generateTrees(i + 1, end)

                # 从左子树集合中选出一棵左子树，从右子树集合中选出一棵右子树，拼接到根节点上
                for l in leftTrees:
                    for r in rightTrees:
                        currTree = BinaryTreeNode(i,l,r)
                        allTrees.append(currTree)

            return allTrees

        resultlist = generateTrees(1, nums) if nums else []
        endlist = []
        for i in resultlist:
            endlist.append(i.breadthFirstTraversal())
        print(endlist)
        return endlist

    """
    96. 不同的二叉搜索树：给你一个整数 n ，求恰由 n 个节点组成且节点值从 1 到 n 互不相同的 二叉搜索树 有多少种？返回满足题意的二叉搜索树的种数。
        示例 1：输入：n = 3，输出：5
        标签：树，二叉搜索树，数学，动态规划，二叉树
        https://leetcode.cn/problems/unique-binary-search-trees/description/
    """

    def uniqueBinarySearchTree_96(self, nums: int) -> int:
        # 思路：这题比95就简单多了，只需要返回数量，不需要返回每个树长啥样。可以这么理解：如果确定了根节点，那么它的可能树种类是左子树种类*右子树种类。
        # 设计一个递归函数，参数是表示待排列的元素个数，对每个元素遍历，递归调用左子树和右子树相乘，再累计
        def recursion(n: int) -> int:
            if n == 0 or n == 1 :
                return 1
            else:
                totalMethods = 0
                for i in range(n):
                    leftmethods = recursion(i)
                    rightmethods = recursion(n-i-1)
                    totalMethods += leftmethods * rightmethods
                return totalMethods

        t = recursion(nums)
        print(t)
        return t

    # 官网解答是根据组合数学原理介绍了卡特兰数，直接用了二重循环，思路差不多，这里不再写了

    """
    97. 交错字符串：给定三个字符串 s1、s2、s3，请你帮忙验证 s3 是否是由 s1 和 s2 交错 组成的。
        示例 1：输入：s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"，输出：true
        示例 2：输入：s1 = "aabcc", s2 = "dbbca", s3 = "aadbbbaccc"，输出：false
        标签：字符串，动态规划
        https://leetcode.cn/problems/interleaving-string/description/
    """

    def interleavingString_97(self, s1:str, s2:str, s3:str) -> bool :
        # 思路：动态规划，从左向右找3个字符串的匹配模式，如果能匹配到，则继续下一段的判断，直至3个字符串任一结束
        # 设置3组坐标指针，分别记录3个字符串正在判断的子串的起始位置，初始化0,0
        idxlist = [[0,0,len(s1)],[0,0,len(s2)],[0,0,len(s3)]]
        strlist = [s1,s2,s3]
        # 标记每轮循环判断是s1还是s2，0表示s1，1表示s2
        side = 0
        # 标记连续匹配不上的次数，等于2说明s1和s2都匹配不上，就返回False
        count = 0
        while idxlist[0][1]<=idxlist[0][2] or idxlist[1][1]<=idxlist[1][2] or idxlist[2][1]<=idxlist[2][2]:
            # 判断最大能匹配的子串
            i = 1
            while s3[idxlist[2][0]:idxlist[2][0]+i] == strlist[side][idxlist[side][0]:idxlist[side][0]+i] \
                    and i <= idxlist[2][2] - idxlist[2][1]:
                print(i,side,s3[idxlist[2][0]:idxlist[2][0]+i],strlist[side][idxlist[side][0]:idxlist[side][0]+i],idxlist)
                i = i + 1
            # 如果当前不匹配，换一个字符串匹配，进入下个循环
            if i == 1 :
                # 还有种情况时匹配到了最后时成功的
                if idxlist[0][1]>=idxlist[0][2] and idxlist[1][1]>=idxlist[1][2] and idxlist[2][1]>=idxlist[2][2]:
                    print(idxlist)
                    return True
                else:
                    side = (side + 1) % 2
                    count = count + 1
                    # 如果已经2轮都不匹配的话，就说明匹配不上了，返回False
                    if count>1:
                        return False
            # 如果当前匹配，idxlist更新一遍，下个循环换一个字符串匹配
            else:
                # 计数器清零
                count = 0
                idxlist[side][1] = idxlist[side][0] + i - 1
                idxlist[side][0] = idxlist[side][0] + i - 1
                idxlist[2][1] = idxlist[2][0] + i - 1
                idxlist[2][0] = idxlist[2][0] + i - 1
                side = (side + 1) % 2

        # 上述思路按照官解的说法不算动态规划，而是双指针法。官解的思路比较复杂，暂时不想了，费脑子。

    """
    98. 验证二叉搜索树：给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。有效 二叉搜索树定义如下：
                     节点的左子树只包含 严格小于 当前节点的数。
                     节点的右子树只包含 严格大于 当前节点的数。
                     所有左子树和右子树自身必须也是二叉搜索树。
        示例 1：输入：root = [2,1,3]，输出：true
        示例 2：输入：root = [5,1,4,null,null,3,6]，输出：false，解释：根节点的值是 5 ，但是右子节点的值是 4 。
        标签：树，深度优先搜索，二叉搜索树，二叉树
        https://leetcode.cn/problems/validate-binary-search-tree/description/
    """

    def validateBinarySearchTree_98(self, root: list) -> bool:
        # 先把list格式改成二叉树BinaryTreeNode的格式，初始化该数据结构
        rootbtn = self.initiateBinaryTreeFromList(root)

        # 思路1：深度遍历每个节点，每个节点都需要判断：
        # 左子树的所有节点值都比该节点值小，右子树的所有节点值都比该节点值大
        # 改进BinaryTreeNode的内容，增加4个成员变量：leftmin\leftmax\rightmin\rightmax，分别表示节点下左子树的最大最小值、右子树的最大最小值
        # BinaryTreeNode类增加函数validateBinarySearchTree，遍历所有节点，计算上述4个值，并判断
        flag = rootbtn.validateBinarySearchTree()
        if flag == None:
            flag = True
        else:
            flag = False
        return flag

        # 思路2：参照官解方法二，如果中序遍历后形成的数组是升序的，说明是有效的二叉搜索树。
        # 当然，在中序遍历的过程中发现顺序不对就可以返回退出，能提高计算效率，
        # 这里不干预修改BinaryTreeNode类LDRTraversal方法了，整个序列拿回来再循环一遍判断
        l2 = rootbtn.LDRTraversal([])
        print(l2)
        for i in range(len(l2)-1):
            if l2[i] > l2[i+1]:
                return False
        return True

        # 思路3：参照官解方法一，设计一个递归函数 helper(root, lower, upper) 来递归判断，
        # 函数表示考虑以 root 为根的子树，判断子树中所有节点的值是否都在 (l,r) 的范围内（注意是开区间）。
        # 如果 root 节点的值 val 不在 (l,r) 的范围内说明不满足条件直接返回，否则我们要继续递归调用检查它的左右子树是否满足，如果都满足才说明这是一棵二叉搜索树。
        # 这个方法一开始不太好理解，也是想了蛮久
        def isValidBST(node: BinaryTreeNode,lower = float('-inf'), upper = float('inf') ) -> bool:
            if not node:
                return True

            val = node.val
            if val <= lower or val >= upper:
                return False

            if not isValidBST(node.left, lower, val):
                return False
            if not isValidBST(node.right, val, upper):
                return False
            return True

        flag = isValidBST(rootbtn,float('-inf'),float('inf'))
        print(flag)
        return flag

    """
    99. 恢复二叉搜索树：给你二叉搜索树的根节点 root ，该树中的 恰好 两个节点的值被错误地交换。请在不改变其结构的情况下，恢复这棵树 。
        示例 1：输入：root = [1,3,null,null,2]，输出：[3,1,null,null,2]，解释：3 不能是 1 的左孩子，因为 3 > 1 。交换 1 和 3 使二叉搜索树有效。
        标签：树，深度优先搜索，二叉搜索树，二叉树
        https://leetcode.cn/problems/recover-binary-search-tree/description/
    """

    def recoverBinarySearchTree_99(self, root: list) -> list:
        # 思路：遍历每个节点，必然存在：该节点左子树的最大值>节点值，或者该节点右子树的最小值<节点值。找出这2个节点，交换
        # 先把list格式改成二叉树BinaryTreeNode的格式，初始化该数据结构
        rootbtn = BinaryTreeNode(root[0], None, None)
        poplist = [rootbtn]
        i = 1
        while poplist != [] or i < len(root):
            currentrootbtn = poplist[0]
            if i < len(root) and root[i] != None:
                leftbtn = BinaryTreeNode(root[i],None,None)
            else:
                leftbtn = None
            currentrootbtn.left=leftbtn
            i = i + 1

            if i < len(root) and root[i] != None:
                rightbtn = BinaryTreeNode(root[i],None,None)
            else:
                rightbtn = None
            currentrootbtn.right = rightbtn
            i = i + 1

            if leftbtn != None:
                poplist.append(leftbtn)
            if rightbtn != None:
                poplist.append(rightbtn)

            poplist.pop(0)

        # 调用BinaryTreeNode.validateBinarySearchTree，计算每个节点是否符合二叉搜索树
        rootbtn.validateBinarySearchTree()

        # 调用广度遍历搜索，碰到的第一个左子树的最大值>节点值，或者该节点右子树的最小值<节点值，即是需要调换的节点
        def breadthFirstTraversal(node:BinaryTreeNode) -> BinaryTreeNode:
            queue = [node]
            while queue:
                n = len(queue)
                for i in range(n):
                    q = queue.pop(0)
                    if q.val < q.leftmax or q.val > q.rightmin:
                        return q
                    if q:
                        queue.append(q.left if q.left else None)
                        queue.append(q.right if q.right else None)
            return node

        newnode = breadthFirstTraversal(rootbtn)
        print(newnode.val, newnode.leftmin, newnode.leftmax,newnode.rightmin, newnode.rightmax)
        val1 = 0
        if newnode.val < newnode.leftmax:
            val1 = newnode.leftmax
        else:
            val1 = newnode.rightmin

        a,b = 0,0
        for i in range(len(root)):
            if root[i] == newnode.val:
                a = i
            if root[i] == val1:
                b = i
                break
        print(a,b)
        tmp = root[a]
        root[a] = root[b]
        root[b] = tmp

        print(root)
        return root

        # 以上经过了n轮递归和循环，肯定不是最优解。
        # 看了下官解，是将二叉树先中序遍历一遍得到结果list，然后判断哪2个位置的数字和相邻的不满足大小关系即可，好像也不很简洁
        # 就不写了


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
    # ma.maximumSubarray_53([-2,1,-3,4,-1,2,1,-5,4])
    # ma.spiralMatrix_54([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    # ma.jumpGame_55([3,2,1,0,4])
    # ma.mergeIntervals_56([[1,9],[2,5],[19,20],[10,11],[12,20],[0,3],[0,1],[0,2]])
    # ma.insertIntervals_57_standard(intervals = [[1,3],[6,9],[12,18],[20,25],[28,30]], newInterval = [32,35])
    # ma.spiralMatrix_59(5)
    # ma.rotateList_61([1,2,3,4,5,6], 2)
    # ma.uniquePaths_62(3,7)
    # ma.uniquePathsII_63([[0,0,0],[0,1,0],[0,0,0]])
    # ma.minimumPathSum_64([[1,3,1],[1,5,1],[4,2,1]])
    # ma.simplifyPath_71("/.../a/../b/c/../d/./")
    # ma.simplifyPath_71("/home/user/Documents/../Pictures")
    # ma.setMatrixZeroes_73([[0,1,2,0],[3,4,5,2],[1,3,1,5]])
    # ma.searchA2dMatrix_74([[1,3,5,7],[10,11,16,20],[23,30,34,60]],3)
    # ma.sortColors_75([2,0,2,1,1,0])
    # ma.combinations_77(5,3)
    # ma.subsets_78([1,2,3,4])
    # ma.wordSearch_79([["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]],"ABCB")
    # ma.removeDuplicates_80([0,0,1,1,1,1,2,3,3])
    # ma.searchInRotatedSortedArray_81([2,5,6,0,0,1,2],3)
    # ma.removeDuplicates_82([1,1,1,2,3])
    # ma.partitionList_86([2,1], 2)
    # ma.grayCode_89(6)
    # ma.susetsII_90([1,2,2])
    # ma.decodeWays_91("226")
    # ma.reverseLinkedListII_92( [5],1, 1)
    # ma.restoreIpAddresses_93('25525511135')
    # ma.uniqueBinarySearchTreeII_95(3)
    # ma.uniqueBinarySearchTree_96(8)
    # print(ma.interleavingString_97("aabcc",  "dbbca",  "aadbbcbcac"))
    # print(ma.interleavingString_97( "aabcc", "dbbca", "aadbbbaccc"))
    print(ma.validateBinarySearchTree_98([5,1,4,None,None,3,6]))
    print(ma.validateBinarySearchTree_98([2,1,3]))
    # ma.recoverBinarySearchTree_99([3,1,4,None,None,2])