"""
    力扣算法题Python实践：https://leetcode.cn/problemset/algorithms/，可用于中学编程教学
    DATE        AUTHOR        CONTENTS
    2025-08-03  Jerry Chang   Create
"""
import itertools
import math
import re
from medium.MediumAlgorithm0_99 import BinaryTreeNode, MediumAlgorithm0_99

# 一个树节点的类，一个节点可能包含多个子节点，子节点由一个list组成，list每个元素也是个TreeNode类
class TreeNode:
    def __init__(self, v:str, childlist = []):
        self.val = v
        self.childlist = childlist

    # 前序深度遍历该节点下的所有树结构，并返回一个所有元素的val列表。根-子
    def DLRTraversal(self,l:list)->list[int]:
        l.append(self.val)
        # 叶子节点，minvalue和maxvalue都是自己
        if self.childlist == [] :
            l.append(None)
        else:
            tmpl = []
            for i in self.childlist:
                tmpl.append(i.val)
            l.append(tmpl)
            for i in self.childlist:
                i.DLRTraversal(l)
        return l

# 主类，算法实现都在这里面
class HardAlgorithm0_199:
    """    构造函数，什么都不做    """

    def __init__(self):
        print('Hello World! I''m HardAlgorithm0_199''')

    """
    4. 寻找两个正序数组的中位数：给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。
        算法的时间复杂度应该为 O(log (m+n)) 。
        示例 1：输入：nums1 = [1,3], nums2 = [2]，输出：2，解释：合并数组 = [1,2,3] ，中位数 2
        示例 2：输入：nums1 = [1,2], nums2 = [3,4]，输出：2.50000，解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
        标签：数组，二分查找，分治
        https://leetcode.cn/problems/median-of-two-sorted-arrays/description/
    """

    def medianOfTwoSortedArrays_4(self, nums1:list, nums2:list) -> float:
        # 思路1：比较笨的办法：先将2个数列按顺序合并，再算出中位数。但合并的过程没法做到O(log (m+n))
        finallist = []
        i,j = 0,0
        # 双重循环将2个数据交错按顺序插入新队列
        while i < len(nums1):
            while j < len(nums2):
                if nums1[i] < nums2[j]:
                    finallist.append(nums1[i])
                    i += 1
                    break
                else:
                    finallist.append(nums2[j])
                    j += 1
            if j == len(nums2):
                finallist.append(nums1[i])
                i += 1
        # 上述只能确保把nums1插完，但可能nums2还有，一并插入
        while j < len(nums2):
            finallist.append(nums2[j])
            j += 1

        # 计算中位数，按照数据个数奇偶性分别计算
        median = 0
        if len(finallist)%2 == 0:
            median = (finallist[len(finallist)//2-1] + finallist[len(finallist)//2])/2
        else:
            median = finallist[len(finallist)//2]
        print(finallist,median)
        # return median

        # 思路2：参考官解方法一，对2个序列分别进行二分查找，排除前面若干不可能的元素。官解的解释比较详细，也能较难理解，这里不赘述。
        # 初始化2个起始位置
        idx1,idx2 = 0,0
        # 先计算中位数所属的位置mid，如果数列个数是奇数就是len((nums1)+len(nums2))//2，如果是偶数就是len((nums1)+len(nums2))//2-1
        mid = 0
        if (len(nums1) + len(nums2))%2 == 1:
            mid = (len(nums1)+len(nums2))//2
        else:
            mid = (len(nums1)+len(nums2))//2 - 1
        # 从总长度/2开始判断
        k = mid
        while idx1 + idx2 < mid:
            newidx1 = min((idx1 + k//2 + 1), len(nums1)-1)
            newidx2 = min((idx2 + k//2 + 1), len(nums2)-1)
            k = k//2
            # nums1[idx1]之前的序列排除
            if nums1[idx1] < nums2[idx2]:
                k = k - idx1 - 1
                idx1 = newidx1
            else:
                k = k - idx2 - 1
                idx2 = newidx2
            print('idx1, idx2, newidx1, newidx2,mid',idx1, idx2, newidx1, newidx2,mid)

        # 如果数列个数是奇数，中位数就是就是nums1[idx1]和nums2[idx2]较小的那个，如果是偶数就是nums1[idx1]和nums2[idx2]的均值
        if (len(nums1) + len(nums2))%2 == 1:
            median = min(nums1[idx1],nums2[idx2])
        else:
            median = (nums1[idx1]+nums2[idx2]) / 2
        print(median)
        #return median

        # 思路3：参考官解方法二，更精妙，也比较难于理解，核心要义是在于怎么找到2个数列各自的位置，使得它是合并后的数列的中间位置
        # 在 [0,m] 中找到最大的 i，使得：nums1[i−1]≤nums2[j]，其中 j= (m+n+1)/2 − i
        # 此时，nums1[i]、nums2[i]、nums1[j]、nums2[j]就是中位数，根据其他条件计算可得

        # 需要注意，判断顺序的排列，元素少的数组放第一个，否则循环到后面数据序列号会溢出
        if len(nums1)>len(nums2):
            nums1,nums2 = nums2,nums1
        # 另须注意，在2个数列前后都加上正负无穷，避免无从比较
        infinty = 2 ** 40
        m, n = len(nums1), len(nums2)
        left, right = 0, m
        # median1：前一部分的最大值； median2：后一部分的最小值
        median1, median2 = 0, 0
        # 下面这段循环我没看懂
        while left <= right:
            # 前一部分包含 nums1[0 .. i-1] 和 nums2[0 .. j-1]； 后一部分包含 nums1[i .. m-1] 和 nums2[j .. n-1]
            i = (left + right) // 2
            j = (m + n + 1) // 2 - i
            # nums_im1, nums_i, nums_jm1, nums_j 分别表示 nums1[i-1], nums1[i], nums2[j-1], nums2[j]
            nums_im1 = (-infinty if i == 0 else nums1[i - 1])
            nums_i = (infinty if i == m else nums1[i])
            nums_jm1 = (-infinty if j == 0 else nums2[j - 1])
            nums_j = (infinty if j == n else nums2[j])
            print('left, right',left, right)
            print('nums_im1, nums_jm1,nums_i, nums_j', nums_im1, nums_jm1, nums_i, nums_j)
            # 对比num1和num2对应元素，如果num1<num2，说明整体中位数坐标还要往num1的右边移动
            # 这里median1, median2的处理是比较奇妙的，我不可能自己想到，也一直没太想明白
            if nums_im1 <= nums_j:
                median1, median2 = max(nums_im1, nums_jm1), min(nums_i, nums_j)
                left = i + 1
            else:
                right = i - 1

        result = (median1 + median2) / 2 if (m + n) % 2 == 0 else median1
        print('result',result, median1, median2)
        return result

    """
    10. 正则表达式匹配：给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。
                    '.' 匹配任意单个字符，'*' 匹配零个或多个前面的那一个元素，所谓匹配，是要涵盖 整个 字符串 s 的，而不是部分字符串。
        示例 1：输入：s = "aa", p = "a"，输出：false，解释："a" 无法匹配 "aa" 整个字符串。
        示例 2：输入：s = "aa", p = "a*"，输出：true，解释：因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。
        标签：递归，字符串，动态规划
        https://leetcode.cn/problems/regular-expression-matching/description/
    """

    def regularExpressionMatching_10(self, s: str, p: str) -> bool:
        # 思路：动态规划，分别给s、p设立2个指针，针对. 、* 和具体字符匹配逐步向后，如果能匹配到最后，返回true，如果中途都不匹配，返回false
        idxs,idxp = 0,0
        while idxs < len(s) and idxp < len(p):
            if p[idxp] == '.':
                idxs += 1
                idxp += 1
            elif p[idxp] == '*':
                if p[idxp-1] == '.':
                    return True
                else:
                    while idxs < len(s) and s[idxs] == p[idxp-1]:
                        idxs += 1
                    idxp += 1
            else:
                if s[idxs] == p[idxp]:
                    idxs += 1
                    idxp += 1
                else:
                    return False
        print('idxs,idxp',idxs,idxp)
        if idxs == len(s) and idxp == len(p):
            return True
        else:
            return False

    """
    23. 合并 K 个升序链表：给你一个链表数组，每个链表都已经按升序排列。请你将所有链表合并到一个升序链表中，返回合并后的链表。
        示例 1：输入：lists = [[1,4,5],[1,3,4],[2,6]]，输出：[1,1,2,3,4,4,5,6]
        标签：链表，分治，堆（优先队列），归并排序
        https://leetcode.cn/problems/merge-k-sorted-lists/description/
    """

    def mergeKSortedLists_23(self, lists: list[list[int]]) -> list[list[int]]:
        # 思路1：给每个数组设置指针，判断顺序，从小到大逐步将元素推入新队列
        # 初始化二维数组，每个元素包含2个数：数组当前的指针和数组长度，便于后续判断，totalcount表示总共需要循环多少次，就是所有元素个数
        idxlist,resultlist = [],[]
        totalcount = 0
        for i in lists:
            idxlist.append([0,len(i)])
            totalcount += len(i)
        print('idxlist',idxlist)
        # 开始循环，每个循环挑出本轮最小值，放入结果队列
        i = 0
        while i < totalcount:
            # minval是一个数组，包含2个元素，第一个记录本轮最小值，第二个记录本轮最小值是在哪个数组里
            # 为防止超出个数组边界，需要判断一下，从第一个未超的队列开始比较
            for k in range(len(idxlist)):
                if idxlist[k][0] < idxlist[k][1]:
                    minval = [lists[k][idxlist[k][0]], k]
            # 每个数组判断，如果没过节且值小于minval，则更新minval值
            for j in range(len(lists)):
                if idxlist[j][0]<idxlist[j][1] and lists[j][idxlist[j][0]] < minval[0]:
                    minval = [lists[j][idxlist[j][0]],j]
            print(minval)
            # 将本轮minval值写入结果数组中
            resultlist.append(minval[0])
            # 更新最小值所在数据的指针挪到下一个
            idxlist[minval[1]][0] = idxlist[minval[1]][0] + 1
            i += 1

        print('resultlist',resultlist)
        return resultlist

        # 官解是用链表数据结构实现的，采用两两分治合并的方式，感觉也挺麻烦，就不写了

    """
    25. K 个一组翻转链表：给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。
                       k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
                       你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。
        示例 1：输入：head = [1,2,3,4,5], k = 2，输出：[2,1,4,3,5]
        标签：递归，链表
        https://leetcode.cn/problems/reverse-nodes-in-k-group/description/
    """

    def reverseNodesInKGroup_25(self, head:list, k:int) -> list[int]:
        # 思路：len//k个循环，每个循环内把数组值翻转

        cycleindex = len(head)//k
        beginidx = 0
        for i in range(cycleindex):
            for j in range(k//2):
                tmp = head[beginidx]
                head[beginidx+j] = head[beginidx+k-j-1]
                head[beginidx+k-j] = tmp
            beginidx += k

        print(head)
        return head

        # 官解思路还是用链表指针的数据结构做的，这里就不写了

    """
    30. 串联所有单词的子串：给定一个字符串 s 和一个字符串数组 words。 words 中所有字符串 长度相同。
                        s 中的 串联子串 是指一个包含  words 中所有字符串以任意顺序排列连接起来的子串。
        示例 1：输入：s = "barfoothefoobarman", words = ["foo","bar"]，输出：[0,9]
        标签：哈希表，字符串，滑动窗口
        https://leetcode.cn/problems/substring-with-concatenation-of-all-words/description/
    """

    def substringWithConcatenationOfAllWords_30(self, s: str, words: list) -> list[int]:
        # 思路：先把words里所有的组合都列出来，然后和s匹配。
        # 先定义一个递归函数，把所有可能的组合都出来
        wordlist = []
        def recursion(headword:str, alternative:list[str]):
            if alternative == []:
                wordlist.append(headword)
                return
            else:
                for i in alternative:
                    tmpl = alternative.copy()
                    tmpl.remove(i)
                    tmpw = headword + i
                    recursion(tmpw,tmpl)

        recursion('', words)
        print(wordlist)

        # 把每个候选字符串跟s匹配，并把结果放入结果集中
        resultlist = []
        l = len(wordlist[0])
        for j in wordlist:
            i = 0
            while i < len(s):
                idx = s[i:].find(j)
                # print(s[i:],j,idx)
                if idx != -1:
                    resultlist.append(idx)
                    i = idx + l
                else:
                    break
        print(resultlist)
        # return resultlist

        # 上述思路是比较中规中矩刻板的，我忘了一个条件是 words 中所有字符串 长度相同，所以性能上肯定不是最优，但通用性好。
        # 官解是这样的：既然words中所有字符串长度相同，那么对s做滑动窗口判断，一直滑到最后。感觉性能上也不是很优
        idx = 0
        wordlen = len(words[0])
        wordscount = len(words)
        resultlist = []
        # 以idx为起始，每个循环判断后面wordlen*wordscount个字符是否符合条件，
        # 如果符合，把idx插入结果；如果不符合，idx向右滑动一位，继续下一轮判断
        while idx < len(s)-wordlen*wordscount:
            tmplist = words.copy()
            for i in range(wordscount):
                # print(idx,i,s[idx+i*wordlen:idx+i*wordlen+wordlen],tmplist)
                if s[idx+i*wordlen:idx+i*wordlen+wordlen] in tmplist:
                    tmplist.pop( tmplist.index(s[idx+i*wordlen:idx+i*wordlen+wordlen]))
                else:
                    break
            if tmplist == []:
                resultlist.append(idx)
            idx += 1
        print(resultlist)
        return resultlist

    """
    32. 最长有效括号：给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。
        示例 1：输入：s = "(()"，输出：2，解释：最长有效括号子串是 "()"
        示例 2：输入：s = ")()())"，输出：4，解释：最长有效括号子串是 "()()"
        标签：栈，字符串，动态规划
        https://leetcode.cn/problems/longest-valid-parentheses/description/
    """

    def longestValidParenthesis_32(self,s:str) -> int:
        # 思路：设置一个空字符串，一个指针读取s，从0开始，
        # 如果是左括号，压入栈中；如果是右括号，和栈中的最右字符比较，如果能形成()，则从栈中pop，结果数量加2，继续判断，直至读完s
        tmps = '.'
        idx = 0
        result = 0
        while idx < len(s):
            # print(tmps,len(tmps))
            if s[idx] == '(':
                tmps += s[idx]
            elif s[idx] == ')' and tmps[len(tmps)-1] == '(':
                tmps = tmps[0:len(tmps)-1]
                result += 2
            elif s[idx] == ')' and tmps[len(tmps)-1] != '(':
                tmps += s[idx]
            idx += 1
        print(result)
        return result

        # 官解反而没太看明白，似乎性能也不是很突出，暂时不做了

    """
    37. 解数独：编写一个程序，通过填充空格来解决数独问题。
        示例 1：输入：board = [ ["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],
                             ["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],
                             [".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
               输出：[["5","3","4","6","7","8","9","1","2"],["6","7","2","1","9","5","3","4","8"],["1","9","8","3","4","2","5","6","7"],
                     ["8","5","9","7","6","1","4","2","3"],["4","2","6","8","5","3","7","9","1"],["7","1","3","9","2","4","8","5","6"],
                     ["9","6","1","5","3","7","2","8","4"],["2","8","7","4","1","9","6","3","5"],["3","4","5","2","8","6","1","7","9"]]
        标签：数组，哈希表，回溯，矩阵
        https://leetcode.cn/problems/sudoku-solver/description/
    """

    def sudokuSolver_37(self, board: list[list[int]]) -> list[list[int]]:
        # 思路：按照游戏规则，把每个空白格子中，可能的数字都列出来。最终总会有至少一个格子，其可能的值只有1个，该格子值确定后，将该值在其他格子可能的备选中删除。
        # 依次循环迭代，直到每个格子的值都有唯一答案。

        # 设计两个dict类型的数据。
        # dict1存储需要填充格子的hashmap，key值是空格序号，value值是不定长的list，记录备选的数字，初始化为[]
        # dict2存储需要填充格子的hashmap，key值是空格序号，value值int，记录该坐标归属的小九宫格最左上角的坐标
        dict1, dict2 = {},{}
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == None:
                    l = i*9+j
                    dict1[l] = []
                    dict2[l] = (i-i%3)*9 + (j-j%3)
        print('空格序号：',dict1)
        print('空格所在小九宫格左上角的序号：',dict2)

        # 对每个空白格子，判断可以填的数字
        for key in dict1.keys():
            tmplist = [1,2,3,4,5,6,7,8,9]
            # 排除所在行不可填的数字
            i = key//9
            for j in range(9):
                if board[i][j] != None and board[i][j] in tmplist:
                    tmplist.remove(board[i][j])
            # 排除所在列不可填的数字
            n = key%9
            for m in range(9):
                if board[m][n] != None and board[m][n] in tmplist:
                    tmplist.remove(board[m][n])
            # 排除所在九宫格不可填的数字
            beginkey = dict2[key]
            for x in range(3):
                for y in range(3):
                    if board[beginkey//9+x][beginkey%9+y] != None and board[beginkey//9+x][beginkey%9+y] in tmplist:
                        tmplist.remove(board[beginkey//9+x][beginkey%9+y])
            dict1[key] = tmplist

        # 此时每个空格备选可以填的数字都以已算出，
        print('空格可选的数字：',dict1)

        # 现在需要先找出有唯一解的，把该解从其他备选清单中去掉，循环迭代，直到每个空格有唯一的解
        # 循环判断直到dict1中没有待判断的元素
        while len(dict1) > 0:
            keylist = list(dict1.keys())
            for key1 in keylist:
                # 如果该座标的候选只有一个，就说明是唯一的
                if len(dict1[key1]) == 1:
                    # 填入原始列表
                    board[key1//9][key1%9] = dict1[key1][0]
                    # 该行的所有备选列表中删除该元素
                    # print('dict1[key1][0]',dict1[key1][0])
                    for i in range(9):
                        if (key1 - key1 % 9 + i != key1) and ((key1 -key1%9 + i) in dict1.keys()) \
                                and (dict1[key1][0] in dict1[key1 -key1%9 + i]):
                            dict1[key1 - key1 % 9 + i].remove(dict1[key1][0])
                    # 该列的所有备选列表中删除该元素
                    for j in range(9):
                        if (j*9 + key1 % 9 != key1) and (j*9 + key1 % 9 in dict1.keys()) \
                                and (dict1[key1][0] in dict1[j*9 + key1 % 9]):
                            dict1[j*9 + key1 % 9].remove(dict1[key1][0])
                    # 该小九宫格的所有备选列表中删除该元素
                    for i in range(3):
                        for j in range(3):
                            if (dict2[key1] + i*9 + j != key1) and (dict2[key1] + i*9 + j in dict1.keys()) \
                                    and (dict1[key1][0] in dict1[dict2[key1] + i*9 + j]):
                                dict1[dict2[key1] + i * 9 + j].remove(dict1[key1][0])
                    # dict1中删除该元素
                    dict1.pop(key1)
        print('结果：', board)
        return board

        # 官解方法一、方法二用递归回溯的方式，硬尝试每个可能的数字组合，感觉性能也不一定好，只是代码量简洁些
        # 官解方法三枚举优化，有点类似于我的方法，只是还是用递归实现了，性能应该会好些

    """
    41. 缺失的第一个正数：给你一个未排序的整数数组 nums ，请你找出其中没有出现的最小的正整数。请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。
        示例 1：输入：nums = [1,2,0]，输出：3，解释：范围 [1,2] 中的数字都在数组中。
        示例 2：输入：nums = [3,4,-1,1]，输出：2，解释：1 在数组中，但 2 没有。
        标签：数组，哈希表
        https://leetcode.cn/problems/first-missing-positive/description/
    """

    def firstMissingPositive_41(self,nums:list[int]) -> int:
        # 思路1：有个专属python的骚操作，可能不符合性能要求。目标数据肯定在1~len(nums)之间，轮询判断是否在数组中就好了
        for i in range(1, len(nums) + 1):
            if i not in nums:
                print(i)
                # return i

        # 思路2：我自己是想不到能满足时间复杂度为 O(n)、只使用常数级别额外空间的方法，看了官解也是个鬼才，可以号称“原地哈希”
        # 第一步：如果nums长度为l，把所有负数变为l+1的正数
        for i in range(len(nums)):
            if nums[i] <= 0:
                nums[i] = len(nums)+1
        print('置正后：',nums)
        # 此时数列所有值都是正数了，遍历每个元素，对于值小于l的，将坐标为该值-1的元素反转为负数
        for i in range(len(nums)):
            absval = abs(nums[i]) - 1
            if absval < len(nums):
                nums[absval] = -abs(nums[absval])
        print('计算后：',nums)
        # 此时，第一个值不为负的元素，其下标+1即为未出现的元素
        for i in range(len(nums)):
            if nums[i] > 0:
                print(i+1,nums)
                return i+1

    """
    42. 接雨水：给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。图示更直观一点，看原网页。
        示例 1：输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]，输出：6
        示例 2：输入：height = [4,2,0,3,2,5]，输出：9
        标签：栈，数组，双指针，动态规划，单调栈
        https://leetcode.cn/problems/trapping-rain-water/description/
    """

    def trappingRainWater_42(self,height:list[int]) -> int:
        # 朴素的思路1：判断一个空位置是否能存水，条件是左边和右边是否都有“壁垒”，先找出数列中最高的位置，从height[1]开始判断，双重循环
        totalwater = 0
        # 先找出数列中最高的位置
        maxleval = 0
        for i in range(len(height)):
            if height[i] > maxleval:
                maxleval = height[i]
        # 从从height[1]开始判断，每个柱子上方是否能“存水”
        for i in range(1,len(height)-1):
            for j in range(height[i],maxleval):
                flag1,flag2 = False,False
                # 向左找壁垒，没有的话跳过
                for m in range(0,i):
                    if height[m] > j:
                        flag1 = True
                        break
                # 向右找壁垒，没有的话跳过
                for m in range(i+1, len(height)):
                    if height[m] > j:
                        flag2 = True
                        break
                # 如果左右都有壁垒，说明这个位置可以装水
                if flag1 and flag2:
                    totalwater += 1
        print(totalwater)
        # return totalwater

        # 官解版动态规划思路，性能要好一些。先计算出每个元素左边最大高度和右边最大高度，取其小减去本元素值，即可装的水
        totalwater = 0
        leftmax,rightmax = [0],[0]
        # 计算出每个元素左边最大高度
        for i in range(1,len(height)):
            leftmax.append(max(leftmax[i-1],height[i-1]))
        # 计算出每个元素右边最大高度
        for i in range(len(height)-2,-1,-1):
            tmp = rightmax[0]
            rightmax.insert(0,max(tmp,height[i+1]))
        print(leftmax,rightmax)

        # 累计水量
        for i in range(1,len(height)-1):
            totalwater += max((min(leftmax[i],rightmax[i]) - height[i]),0)
        print(totalwater)
        return totalwater

    """
    51. N 皇后：按照国际象棋的规则，皇后可以攻击与之处在同一行或同一列或同一斜线上的棋子。
               n 皇后问题 研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。
               给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。
        示例 1：输入：n = 4，输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]
        标签：数组，回溯
        https://leetcode.cn/problems/n-queens/description/
    """

    def nQueens_51(self,n:int) -> list[list[str]]:
        # 思路：硬递归，在每个格子一层一层向下尝试，效率可能不太好
        result = []

        # 定义一个递归函数，第一个参数是singleresult，第二个参数是递归深入的层次，表示本次从第几行开始
        def recursion(sr:list[str],level:int):
            # 如果到达了最底层，说明本次走通，将sr放入result中
            if level == n:
                a = sr.copy()
                result.append(a)
                print('层',level,'递归成功，result',result)
                return True
            # 如果未到最底层，需要继续判断
            # ssrr = sr.copy()

            # 本行循环判断
            flag = False
            for i in range(n):
                print('层',level,'列',i, 'ssrr befor:',sr,'点位',sr[level][i],'flag',flag)
                # 如果本次位置为.，说明可以放Q，置为Q
                if sr[level][i] == '.':
                    sr[level] = sr[level][:i] + 'Q' + sr[level][i+1:]
                    # print('ssrr[level]',ssrr[level])
                    flag = True
                    # 把该位置所有竖向下、斜向下位置都置为‘N’表示后面不可放置Q
                    x,y = i,i
                    for j in range(level+1, n):
                        x = x - 1
                        if 0<=x<n:
                            sr[j] = sr[j][:x] + 'N' + sr[j][x+1:]
                        sr[j] = sr[j][:i] + 'N' + sr[j][i+1:]
                        y = y + 1
                        if 0<=y<n:
                            sr[j] = sr[j][:y] + 'N' + sr[j][y+1:]
                        # print('j,x', j, x,'  j,i',j,i, '  j,y',j,y)
                        # print('ssrr mid:',ssrr)
                    # 下一层递归
                    print('层',level,'列',i,'竖向下、斜向下位置都置为‘N’后的ssrr:',sr)
                    # 如果下一层递归失败，需要把本层改回去，,注意，这里不能直接根据本层Q的位置改它的影响，而是要从第0层开始重新置N，这里比较费性能
                    # b = level+1
                    ff = recursion(sr,level+1)
                    if not ff:
                        print('层',level,'列',i,'向下递归失败,ssrr:',sr)
                        for l in range(n):
                            for m in range(n):
                                if sr[l][m] != 'Q':
                                    sr[l] = sr[l][:m] + '.' + sr[l][m+1:]
                        # 当前位置的Q也要改回成.
                        sr[level] = sr[level][:i] + '.' + sr[level][i+1:]
                        print('层', level, '列', i, '向下递归失败,清零后的ssrr:', sr)
                        for l in range(level):
                            for m in range(n):
                                if sr[l][m] == 'Q':
                                    x, y = m, m
                                    for j in range(l + 1, n):
                                        x = x - 1
                                        if 0 <= x < n :
                                            sr[j] = sr[j][:x] + 'N' + sr[j][x + 1:]
                                        sr[j] = sr[j][:m] + 'N' + sr[j][m + 1:]
                                        y = y + 1
                                        if 0 <= y < n :
                                            sr[j] = sr[j][:y] + 'N' + sr[j][y + 1:]
                        print('层', level, '列',i,'向下递归失败,修正后的ssrr:', sr)

            # 如果本行都循环完了，还是没法安置Q，说明该方案行不通，返回False不要再递归了
            if not flag:
                return False

        # 单个矩阵初始化为全部.
        singleresult = ['.'*n for _ in range(n)]
        print(singleresult)
        # 递归调用
        recursion(singleresult, 0)
        # 把结果中标记临时状态的N恢复成.
        for i in range(len(result)):
            for j in range(len(result[i])):
                result[i][j] = result[i][j].replace('N','.')

        print('共有',len(result),'个方案',result)
        return result

        # 官解方法的确要巧妙些，和我的思路是反向的，它在判断每个格子是否可以放置皇后时，就看该格子所在列、两条斜线上是否已有皇后即可，
        # 不需要像上述方法一样回溯判断并记录整个棋盘上哪个可以放皇后，哪个不可以放。代码暂时不写了，这个题做了2天有点累了。

    """
    60. 排列序列：给出集合 [1,2,3,...,n]，其所有元素共有 n! 种排列。按大小顺序列出所有排列情况，并一一标记，当 n = 3 时, 所有排列如下：
                "123"，"132"，"213"，"231"，"312"，"321"，给定 n 和 k，返回第 k 个排列。
        示例 1：输入：n = 3, k = 3，输出："213"
        示例 2：输入：n = 4, k = 9，输出："2314"
        标签：递归，数学
        https://leetcode.cn/problems/permutation-sequence/description/
    """

    def permutationSequence_60(self,n:int,k:int) -> str:
        # 思路：递归算出所有的排列组合，再取出第k个

        resultlist = []

        def recursion(s:str, candidates:list[int]) -> None:
            if candidates == []:
                resultlist.append(s)
                return
            else:
                for i in candidates:
                    cc = candidates.copy()
                    ss = s + str(i)
                    cc.remove(i)
                    recursion(ss, cc)

        c = [i for i in range(1,n+1)]
        recursion('', c)
        print('所有排列组合：',resultlist,'，其中第k个：',resultlist[k-1])
        # return resultlist[k-1]

        # 官解思路倒是也挺有意思的，完全不同，通过数学推导出，第 k 个排列的首个元素是有公式推导出来的，依此类推，使用相似的思路，确定下一个元素
        # 这样的话，就不用递归，直接循环完事，时空性能都优于前面的递归
        c = [i for i in range(1, n + 1)]
        nn = n
        kk = k
        s = ''
        while len(c) > 1:
            m = (kk-1)//math.factorial(nn-1)
            s = s + str(c[m])
            c.pop(m)
            kk = kk % math.factorial(nn-1)
            nn = nn - 1
            print('after',c,kk,s,nn)
        s = s + str(c[0])
        print(s)
        return s

    """
    65. 有效数字：给定一个字符串 s ，返回 s 是否是一个 有效数字。
                例如，下面的都是有效数字："2", "0089", "-0.1", "+3.14", "4.", "-.9", "2e10", "-90E3", "3e+7", "+6e-1", "53.5e93", "-123.456e789"，
                而接下来的不是："abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53"。
        示例 1：输入：s = "0"，输出：true
        示例 2：输入：s = "e"，输出：false
        示例 3：输入：s = "."，输出：false
        标签：字符串
        https://leetcode.cn/problems/valid-number/description/
    """

    def validNumber_65(self,s:str) -> bool:
        print('待匹配字符串',s)
        # 思路：假如能用正则表达式的话，这题就很好解
        pattern = '(^([+-]?[0-9]*[.]?[0-9]*)$)|(^([+-]?[0-9]*[.]?[0-9]+(e|E)[+-]?[0-9]+)$)'
        flag = re.match(pattern,s)
        print(flag)
        # if flag == None:
        #     return False
        # else:
        #     return True

        # 但显然题目的意思不是用现成的
        i = 0

        # 当首位是'+','-'时，符合要求，
        if s[i] not in ('+','-','0','1','2','3','4','5','6','7','8','9','.'):
            return False

        if s[i] in ('+','-'):
            i = i + 1

        # 下一步后面应该跟数字或'.'，且.只能出现一次
        tmp = ''
        j = 0
        while i+j < len(s):
            if s[i+j] in ('0','1','2','3','4','5','6','7','8','9','.'):
                j = j + 1
            else:
                break
        tmp = tmp + s[i:i + j]
        if tmp.count('.') > 1:
            return False
        print('tmp',tmp)
        i = i + j
        print('i',i)
        # 此时如果i到底了就返回True
        if i == len(s):
            return True
        # 此时如果s后面还有内容的话，应该是指数部分
        if s[i] in ('e', 'E'):
            i = i + 1
        else:
            return False
        # 如果e|E后面没了，那也是部队的
        if i == len(s):
            return False
        if s[i] in ('+', '-'):
            i = i + 1
        # 如果e|E和=|-后面也没了，那也是不对的
        if i == len(s):
            return False
        # 后面应当全是数字，否则也是不对的
        while i < len(s):
            if s[i] not in ('0','1','2','3','4','5','6','7','8','9'):
                return False
            i = i + 1
        return True

        # 官解使用有限状态自动机的概念，其实跟上述思想差不多，这里就不写了

    """
    68. 文本左右对齐：给定一个单词数组 words 和一个长度 maxWidth ，重新排版单词，使其成为每行恰好有 maxWidth 个字符，且左右两端对齐的文本。
                   你应该使用 “贪心算法” 来放置给定的单词；也就是说，尽可能多地往每行中放置单词。必要时可用空格 ' ' 填充，使得每行恰好有 maxWidth 个字符。
                   要求尽可能均匀分配单词间的空格数量。如果某一行单词间的空格不能均匀分配，则左侧放置的空格数要多于右侧的空格数。
                   文本的最后一行应为左对齐，且单词之间不插入额外的空格。
        示例 1：输入: words = ["This", "is", "an", "example", "of", "text", "justification."], maxWidth = 16
               输出:[   "This    is    an",
                       "example  of text",
                       "justification.  "
                    ]
        标签：数组，字符串，模拟
        https://leetcode.cn/problems/text-justification/description/
    """

    def textJustification_68(self,words:list[str], maxWidth:int) -> list[str]:
        # 思路：先根据maxWidth的限制，确定应该把单词组分成几部分，再根据每组单词的长度确定单词间应该留几个空格
        # 先根据maxWidth的限制，确定应该把单词组分成几部分，结果放进groupwords中
        groupwords, unitwords,groupworscount = [],[],[]
        grouplength = 0
        for w in words:
            if grouplength + len(w)  + len(unitwords) < maxWidth:
                # print('if',w, grouplength, len(w), unitwords, len(unitwords))
                unitwords.append(w)
                grouplength += len(w)
            else:
                # print('else', w, grouplength, len(w), unitwords, len(unitwords))
                # 把每一组单词字符数量也记下来，便于后续算空格数
                groupworscount.append(grouplength)
                grouplength = len(w)
                groupwords.append(unitwords.copy())
                unitwords = [w]
        groupwords.append(unitwords.copy())
        groupworscount.append(grouplength)

        print(groupwords,groupworscount)

        # 确定每组单词中间应该空几格，放入最终结果result中
        result = []
        s = ''
        for i in range(len(groupwords)-1):
            s = ''
            minblanks = (maxWidth-groupworscount[i])//(len(groupwords[i])-1)
            model = (maxWidth-groupworscount[i])%(len(groupwords[i])-1)
            for j in range(len(groupwords[i])-1):
                s = s + groupwords[i][j] + ' ' * minblanks
                if j < model:
                    s = s + ' '
            # 最后一个单词后面不用加空格
            s = s + groupwords[i][-1]
            result.append(s)

        # 处理最后一行，末尾追加空格
        s = ''
        for j in range(len(groupwords[-1])):
            s = s + groupwords[-1][j] + ' '
        model = maxWidth - len(s)
        s = s + ' ' * model
        result.append(s)
        print(result)
        return result

    """
    76. 最小覆盖子串：给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。
                   注意：对于 t 中重复字符，我们寻找的子字符串中该字符数量必须不少于 t 中该字符数量。如果 s 中存在这样的子串，我们保证它是唯一的答案。
                   进阶：你能设计一个在 o(m+n) 时间内解决此问题的算法吗？
        示例 1：输入：s = "ADOBECODEBANC", t = "ABC"，输出："BANC"，解释：最小覆盖子串 "BANC" 包含来自字符串 t 的 'A'、'B' 和 'C'。
        标签：哈希表，字符串，滑动窗口
        https://leetcode.cn/problems/minimum-window-substring/description/
    """

    def minimumWindowSubstring_76(self,s:str,t:str) -> str:
        # 思路：如果有答案的话，那么结果子串的长度一定>=t的长度，我们从t长度开始，一直到s长度，逐步滑动窗口的方式判断每个字串内容是否都包含t
        # 但是这个思路的性能不满足o(m+n)
        lent = len(t)
        while lent<=len(s):
            for i in range(len(s)-lent+1):
                finalstr,tmpstr = s[i:i+lent],s[i:i+lent]
                # print('本轮待匹配字符串', tmpstr)
                # 这里正式判断tmpstr是否包含t的所有字母
                for j in range(len(t)):
                    p = tmpstr.find(t[j])
                    # print('    待匹配字符串', tmpstr,'本轮待匹配t的字符',t[j],'是否匹配',p)
                    if p == -1:
                        break
                    else:
                        tmpstr = tmpstr[0:p] + tmpstr[p+1:]
                # 此时，tmpstr中能匹配得上得字符都剔除掉了，剩下来的字符长度应该是lent - len(t)，否则说明t中有没匹配上的
                if len(tmpstr) == lent - len(t):
                    print('有匹配的',finalstr)
                    return finalstr
            lent += 1
        print('没有匹配的')
        return ''

    def minimumWindowSubstring_76_map(self, s: str, t: str) -> str:
        # 思路2：创建2个dict类型的hashmap，一个存放t中每个字符出现的次数，一个存放t中每个字符出现在s中的坐标，
        # 经过一轮判断后，取hashmap中坐标的最大最小值，既是最小匹配字符串，当同一个字母出现多次的话，需要额外判断取哪个最优
        # 最后做下来，感觉这个方法也挺麻烦，思路比上一个还要绕
        # 初始化countmap,idxmap
        countmap,idxmap = {},{}
        for i in range(len(t)):
            if t[i] not in countmap:
                countmap[t[i]] = 1
            else:
                countmap[t[i]] += 1
            idxmap[t[i]] = set()

        # 记录t中每个字符在s中的坐标
        for i in range(len(t)):
            j = 0
            while j<len(s):
                p = s.find(t[i],j)
                if p == -1:
                    break
                idxmap[t[i]].add(p)
                j = p + 1
        print(countmap,idxmap)

        # 走到这里发现也挺难的，如果是一个字母出现多次，特别是t中有重复字母的话，还需要继续把所有可能的排列组合列出来，找个最短的
        # 这里偷懒，用个现成的排列组合函数
        newmap = {}
        for i in idxmap.keys():
            l = []
            for u in itertools.combinations(idxmap[i],countmap[i]):
                l.append(list(u))
            newmap[i] = l
        print(newmap,list(newmap.keys()))
        # 到这里，newmap是个dict（即hashmap），key是t中的每个字母，value是字母在s中所有坐标的可能组合，value也是个list
        def recursion(l:list, keys:list,result:list) -> int:
            if keys ==[]:
                minl, maxl =min(l), max(l)
                if (maxl-minl) < (result[1] - result[0]):
                    result[1], result[0] = maxl, minl
                return 0
            for i in keys:
                for j in newmap[i]:
                    # print('i,j', i,j,newmap[i])
                    ll = l.copy()
                    ll.extend(j)
                    kk = keys.copy()
                    kk.remove(i)
                    # print('ll,kk',ll,kk)
                    recursion(ll,kk,result)
        rr = [0,len(s)]
        recursion([],list(newmap.keys()),rr)
        print(rr,s[rr[0]:rr[1]+1])
        return s[rr[0]:rr[1]+1]

        # 官解是用滑动窗口思想，在滑动窗口类型的问题中都会有两个指针，一个用于「延伸」现有窗口的 r 指针，和一个用于「收缩」窗口的 l 指针。
        # 在任意时刻，只有一个指针运动，而另一个保持静止。我们在 s 上滑动窗口，通过移动 r 指针不断扩张窗口。
        # 当窗口包含 t 全部所需的字符后，如果能收缩，我们就收缩窗口直到得到最小窗口。
        # 如何判断当前的窗口包含所有 t 所需的字符呢？我们可以用一个哈希表表示 t 中所有的字符以及它们的个数，用一个哈希表动态维护窗口中所有的字符以及它们的个数，
        # 如果这个动态表中包含 t 的哈希表中的所有字符，并且对应的个数都不小于 t 的哈希表中各个字符的个数，那么当前的窗口是「可行」的。
        # 应该性能好点，暂时不写了。

    """
    84. 柱状图中最大的矩形：给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。求在该柱状图中，能够勾勒出来的矩形的最大面积。
        示例 1:输入：heights = [2,1,5,6,2,3]，输出：10，解释：最大的矩形为图中红色区域，面积为 10
        标签：栈，数组，单调栈
        https://leetcode.cn/problems/largest-rectangle-in-histogram/description/
    """

    def largestRectangleinHistogram_84(self,heights:list[int]) -> int:
        # 思路：一个笨办法：从底向上一层一层算，每层的最大面积，最终取最大的。每层的最大面积，取决于该层上最大的连续柱子数量

        # 先把原列表去重排序，找出列表中的最小最大值，以此为搜索范围开始，最小值如果是0没有意义，忽略，从1开始
        tidylist = sorted(list(set(heights)))
        if 0 in tidylist:
            tidylist.remove(0)
        print('去重排序后的列表',tidylist)

        # 从小到大每个元素值，判断该元素周围不小于它的最大连续宽度，计算面积并保留最大的
        maxacreage = 0
        for i in tidylist:
            b = 0
            while True:
                try:
                    b = heights.index(i,b,len(heights))
                except:
                    break
                # 以b坐标为中心，分别向左、向右延伸寻找边界
                else:
                    width = 0
                    # 向左
                    left = b-1
                    while left>=0 and heights[left] >= i:
                        left = left-1
                        width = width+1
                    # print('  b',b,'left', left,'width',width)
                    # 向右
                    right = b
                    while right < len(heights) and heights[right] >= i:
                        right = right+1
                        width = width+1
                    # print('  b',b,'right', right, 'width', width)
                    # b坐标向后挪一位，以防i值有重复，要全部判断到
                    b = b + 1
                # 保留面积最大的
                if i*width > maxacreage:
                    maxacreage = i*width
                print('高度为', i, '坐标',b-1,'最大连续宽度为', width,'最大面积', maxacreage)

        print('最大面积', maxacreage)
        return maxacreage

        # 官解用了单调栈，性能应该更优，但是比较难理解，暂时理解不了，先放一放

    """
    85. 最大矩形：给定一个仅包含 0 和 1 、大小为 rows x cols 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。
        示例 1：输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]，输出：6
        标签：栈，数组，动态规划，矩阵，单调栈
        https://leetcode.cn/problems/maximal-rectangle/description/
    """

    def maximalRectangle_85(self,matrix:list[list[str]]) -> int:
        # 思路：遍历每个格子，碰到1，从这里开始向右下方拓展计算能到达的最大矩形，记下来
        maxrectangle = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == "1":
                    print('当前坐标', i, j, '坐标值', matrix[i][j])
                    # 先看该位置向右或者向下能拓展的最大边界
                    maxi,maxj = i,j
                    while maxi<len(matrix) and matrix[maxi][j] == '1':
                        maxi = maxi+1
                    while maxj<len(matrix[0]) and matrix[i][maxj] == '1':
                        maxj = maxj+1
                    print('向下、向右能拓展的最大边界',maxi-1,maxj-1)
                    # 逐步向下试探最大矩阵面积
                    m = i
                    while m < maxi:
                        n = j
                        while n < maxj:
                            # print('m,n',m,n,'matrix[m][n]',matrix[m][n])
                            if matrix[m][n] == '1':
                                if maxrectangle < (m-i+1)*(n-j+1):
                                    maxrectangle = (m-i+1)*(n-j+1)
                                n = n + 1
                            else:
                                maxj = n
                                break
                        # print('maxi,maxj',maxi,maxj)
                        m = m + 1
                    print('maxrectangle',maxrectangle)
        print(maxrectangle)
        return maxrectangle

        # 官解本质思路应该跟我的上述思路差不多，并且补充说明，该方法本质上是「84. 柱状图中最大的矩形」题中优化暴力算法的复用，这个我倒是没想到

    """
    87. 扰乱字符串：使用下面描述的算法可以扰乱字符串 s 得到字符串 t ：
                     如果字符串的长度为 1 ，算法停止；如果字符串的长度 > 1 ，执行下述步骤：
                     在一个随机下标处将字符串分割成两个非空的子字符串。即，如果已知字符串 s ，则可以将其分成两个子字符串 x 和 y ，且满足 s = x + y 。
                     随机 决定是要「交换两个子字符串」还是要「保持这两个子字符串的顺序不变」。即，在执行这一步骤之后，s 可能是 s = x + y 或者 s = y + x 。
                     在 x 和 y 这两个子字符串上继续从开始递归执行此算法。
                 给你两个 长度相等 的字符串 s1 和 s2，判断 s2 是否是 s1 的扰乱字符串。如果是，返回 true ；否则，返回 false 。
        示例 1：输入：s1 = "great", s2 = "rgeat"，输出：true
        示例 2：输入：s1 = "abcde", s2 = "caebd"，输出：false
        标签：字符串，动态规划
        https://leetcode.cn/problems/scramble-string/description/
    """

    def scrambleString_87(self,s:str,t:str) -> bool:
        # 思路：将s按照上述算法暴力递归得到所有的可能，如果命中t则返回True，
        resultlist = []
        # 建立递归函数，ss是每次待分割组合的字符串，返回值是个list，表示ss分割组合后所有的结果列表
        def recursion(ss:str) -> list:
            print('ss',ss,len(ss))
            # 初始化：当ss长度为0/1/2时的返回值
            if len(ss) == 0:
                return []
            elif len(ss)==1:
                return [ss]
            elif len(ss)==2:
                return [ss,ss[1]+ss[0]]
            # 当ss长度超过2时就要递归分割，首先有len(ss)-1中分割可能，依次循环
            tmptmplist = []
            for i in range(1,len(ss)):
                tmplist = []
                print('i',i,ss[:i],ss[i:])
                # 递归左边字符串
                leftlist = recursion(ss[:i])
                #递归右边字符串
                rightlist = recursion(ss[i:])
                # print('leftlist',leftlist,'rightlist',rightlist)
                # 两边所有字符串的可能组合，是左+右，或者右+左，都放到备选列表里
                for l in leftlist:
                    for r in rightlist:
                        tmplist.append(l + r)
                        tmplist.append(r + l)
                # 除重
                tmplist = list(set(tmplist))
                # print('tmplist',tmplist)
                # 如果左右长度和s一致，放到最终结果列表里
                if (len(leftlist[0])+len(rightlist[0]))==len(s):
                    resultlist.extend(tmplist)
                tmptmplist.extend(tmplist)
                # print('tmptmplist',tmptmplist)
            return tmptmplist

        ts = recursion(s)
        resultset = set(resultlist)
        print(len(resultset), resultset)
        if t in resultset or t in ts:
            return True
        else:
            return False

        # 官解的动态规划方法实在没看明白，暂时放弃，貌似时空性能是更优的

    """
    115. 不同的子序列：给你两个字符串 s 和 t ，统计并返回在 s 的 子序列 中 t 出现的个数。
        示例 1：输入：s = "rabbbit", t = "rabbit"，输出：3
        标签：字符串，动态规划
        https://leetcode.cn/problems/distinct-subsequences/description/
    """

    def distinctSubsequences_115(self,s:str,t:str) -> int:
        # 思路：一个笨办法，先确定t一共有哪些划分方法：理论上可以切成1~len(t)-1段，当切成n段时，有C(len(t)-1,n)中切法
        # 所以，每种切法都会形成一个子串列表，判断这些子串是否都在s中即可，同时可以加一个约束条件是，当t切成n段时，s的长度必须大于len(t)+n-1，因为每段之间至少要有一个字母的间隔

        # 先定义一个递归函数，实现动态规划，原理后面讲
        # 参数：idx：当前判断第几个子串；nn：总共子串的数量；dvdl:子串内容；idxl:子串在s中匹配到的坐标。以上参数都是只读的
        #      rl：存储最终结果，虽然是个list，但是只存储一个元素值，因为要代入到递归深层次中做全局累计，所以用了list传址
        def recursion(idx:int, nn:int, dvdl:list, idxl:list, rl:list) :
            # print('recursion:',idx,nn,dvdl,idxl)
            if idx == nn-1:
                rl[0] = rl[0] + 1
                # print('recursion result',rl[0])
            else:
                for i in idxl[idx]:
                    f = False
                    for j in idxl[idx+1]:
                        # print('  recursion:',i,len(dvdl[idx]),j)
                        # 子串之间在s中至少要有1个字母的间隔，
                        if i+len(dvdl[idx]) < j:
                            f = True
                            recursion(idx+1,nn,dvdl,idxl,rl)
                        else:
                            break
                    if not f:
                        return


        # 先确定n的上限，以及初始化t的坐标列表idxoroginlist，以及最终划分方案结果列表：
        maxn = len(s)-len(t) + 1
        idxoroginlist = list(range(len(t)-1))
        resultlist = [0]
        print('idxoroginlist',idxoroginlist)
        # 从把t分成1段开始，一步一步往上加。分成n段，需要在0~len(t)-2坐标中找出n-1个
        n = 1
        while n <= maxn:
            idxcomb = itertools.combinations(idxoroginlist,n-1)
            for i in idxcomb:
                l = list(i)
                if l ==[] and s.find(t) != -1:
                    resultlist[0] = resultlist[0] + 1
                else:
                    # 先把坐标变成子串列表
                    dividelist = []
                    tmpidx = 0
                    for j in l:
                        dividelist.append(t[tmpidx:j+1])
                        tmpidx = j+1
                    dividelist.append(t[tmpidx:len(t)+1])
                    # 对子串列表中的每个子串判断是否在s中，按照题目要求，先把每个子串在s中所有的匹配位置找出来，存在一个二维数组中
                    flag = True
                    idxslist = []
                    for divides in dividelist:
                        tmplist = []
                        tmpidx = s.find(divides)
                        while tmpidx != -1:
                            tmplist.append(tmpidx)
                            tmpidx = s.find(divides,tmpidx+1)
                        # 如果当前子串一个都没在s中匹配到，那么整个dividelist全部放弃
                        if tmplist == []:
                            flag = False
                            break
                        else:
                            idxslist.append(tmplist)
                    # 当该dividelist中所有字串都能在s中匹配到时，才满足条件，进一步判断有几种匹配组合方式
                    if flag:
                       # 此时，对dividelist、idxslist做动态规划的判断，是否满足要求，注意子串之间在s中至少要有1个字母的间隔，否则可能会出现重复结果
                       recursion(0,n,dividelist,idxslist,resultlist)
                       print('分成', n, '段的分法', dividelist, '匹配s的t子串坐标', idxslist, 'resultlist', resultlist)

            n = n + 1

        print('result',resultlist[0])
        # return resultlist[0]

        # 官解直接用公式推导s和t匹配的动态规划转移方程，还是比较抽象难于理解的，代码确实很简洁，根据对算法本身的理解再写一遍

        # 初始化一个dp二维矩阵，dp[i][j]表示字符串s[i:]包含字符串t[j:]的个数，i<=len(s),j<=len(t)，
        # 矩阵最右一列为1，表示t是空字符串时是任意字符串的子序列，其他置为0
        dp = [[0 for _ in range(len(t)+1)] for _ in range(len(s)+1)]
        for i in range(len(s)+1):
            dp[i][len(t)] = 1

        # 自下而上，自右向左，逐步迭代计算dp[i][j]
        for i in range(len(s)-1,-1,-1):
            for j in range(len(t)-1,-1,-1):
                if s[i] == t[j]:
                    dp[i][j] = dp[i+1][j+1] + dp[i+1][j]
                else:
                    dp[i][j] = dp[i+1][j]
        print('dp',dp)
        return dp[0][0]

    """
    123. 买卖股票的最佳时机 III：给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。
                             注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
        示例 1:输入：prices = [3,3,5,0,0,3,1,4]，输出：6
        解释：在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。
             随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3 。
        标签：数组，动态规划
        https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iii/description/
    """

    def bestTimeToBuyAndSellStockIII_123(self,prices:list) -> int:
        # 这道题自己想实在没什么好的思路，参考官解和网友的思路，自己重写的。

        # 思路1网友的：由于用到了多次的、多重循环，性能远不及官解，但是比较好理解
        # 先算一个长度为n的数组，第i个元素表示第i天及第i天之前完成第一笔交易后，最大利润。
        # 再算一个长度为n的数组，第i个元素表示第i天之后，购买第二笔交易，能获得的最大利润。
        # 然后遍历两个数组，找到 vec1[i] + vec2[i] 最大的就是解了。
        # 这里需要注意，网友说算最大利润“只要找到0 ~ i 中间的最大和最小值就行” ，这是不对的，还要考虑买卖是有先后顺序的，如果最大值在最小值前面就不成立

        # 先单独定义一个函数，计算一个序列中单次买卖利润最大值
        def maxrevenue(subprices:list) -> int:
            maxval = 0
            for i in range(len(subprices)):
                for j in range(i+1,len(subprices)):
                    if maxval < subprices[j] - subprices[i]:
                        maxval = subprices[j] - subprices[i]
            return maxval

        # 计算每个子序列的最大利润，记入vec1,vec2中
        vec1,vec2 = [],[]
        for i in range(len(prices)):
            leftlist = prices[:i+1]
            vec1.append(maxrevenue(leftlist))
            rightlist = prices[i+1:]
            vec2.append(maxrevenue(rightlist))
        print(vec1, vec2)

        # 比较vec1[i]+vec2[i]的最大值，选取最大的
        maxrevenue = 0
        for i in range(len(vec1)):
            if maxrevenue < vec1[i]+vec2[i]:
                maxrevenue = vec1[i]+vec2[i]

        print('最大收益',maxrevenue)
        # return maxrevenue

        # 思路2：官解动态规划，似乎股票收益类的算法都可以用动态规划，有个122题也是，比这个简单点，没做过。
        # 这个动态规划算法比122题麻烦的是，限定最多交易两次，所以有4个核心状态：
        # 第一次买、第一次卖、第二次买、第二次卖，需要在每个prices[i]中判断这4个状态的转移关系。详细逻辑看网页这里不写了，只是重写下代码。
        # 先初始化原始状态的收益：
        buy1,sell1,buy2,sell2 = -prices[0],0,-prices[0],0
        for i in range(1,len(prices)):
            buy1 = max(buy1,-prices[i])
            sell1 = max(sell1,buy1+prices[i])
            buy2 = max(buy2,sell1-prices[i])
            sell2 = max(sell2, buy2+prices[i])
        print('动态规划最大收益',buy1,sell1,buy2,sell2)
        return sell2

    """
    124. 二叉树中的最大路径和：二叉树中的 路径 被定义为一条节点序列，序列中每对相邻节点之间都存在一条边。
                        同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。
                        路径和 是路径中各节点值的总和。给你一个二叉树的根节点 root ，返回其 最大路径和 。
        示例 2：输入：root = [-10,9,20,null,null,15,7],输出：42,解释：最优路径是 15 -> 20 -> 7 ，路径和为 15 + 20 + 7 = 42
        标签：树，深度优先搜索，动态规划，二叉树
        https://leetcode.cn/problems/binary-tree-maximum-path-sum/description/
    """

    def binaryTreeMaximumPathSum_124(self,root:list) -> int:
        # 思路：考虑到二叉树是左右对称的，这里只看从左到右的路径。通过中序深度遍历的方式，定义：每个节点的最大路径值=从其他左边的节点到达该节点的最大路径值
        # 1、如果节点是左叶子，记为：本节点值；
        # 2、如果节点是右叶子，记为：max(父节点最大路径值 + 本节点值, 本节点值)；
        # 3、如果节点不是叶子，且节点本身是左/根节点，记为：max(左子节点最大路径值 + 本节点值, 本节点值)
        # 4、如果节点不是叶子，且节点本身是右节点，记为：max(左子节点最大路径值 + 本节点值, 父节点最大路径值 + 本节点值,  本节点值)

        # 先将root列表初始化为BinaryTreeNode
        ma = MediumAlgorithm0_99()
        rootbtn = ma.initiateBinaryTreeFromList(root)
        print('rootbtn',rootbtn.val,rootbtn.left.val,rootbtn.right.val)

        # 定义一个中序深度遍历的递归函数，计算每个节点的最大路径值,direct说明mode节点是左还是右
        # 这里借用BinaryTreeNode的leftmax成员变量，其实不是这么用的
        def LDRTraversal(node:BinaryTreeNode, paranode:BinaryTreeNode, direct:str, result:list):
            # 叶子节点的情况
            if node.left == None and node.right == None:
                if direct == 'L':
                    node.leftmax = node.val
                elif direct == 'R':
                    node.leftmax = max(paranode.leftmax+node.val, node.val)
                if result[0] < node.leftmax:
                    result[0] = node.leftmax
                return
            # 非叶子节点
            if node.left != None:
                LDRTraversal(node.left, node, 'L',result)

            if direct == 'L':
                node.leftmax = max(node.left.leftmax+node.val, node.val)
            elif direct == 'R':
                node.leftmax = max(node.left.leftmax+node.val, paranode.leftmax+node.val, node.val)
            if result[0] < node.leftmax:
                result[0] = node.leftmax

            if node.right != None:
                LDRTraversal(node.right, node, 'R',result)

        # 执行递归函数，计算最大路径
        resultlist = [float('-inf')]
        LDRTraversal(rootbtn, None, 'L', resultlist)

        print(resultlist[0])
        return resultlist[0]

    """
    126. 单词接龙 II：给你两个单词 beginWord 和 endWord ，以及一个字典 wordList 。
                    请你找出并返回所有从 beginWord 到 endWord 的 最短转换序列 ，如果不存在这样的转换序列，返回一个空列表。
                    每个序列都应该以单词列表 [beginWord, s1, s2, ..., sk] 的形式返回。
        示例 1：输入：beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
               输出：[["hit","hot","dot","dog","cog"],["hit","hot","lot","log","cog"]]
               解释：存在 2 种最短的转换序列："hit" -> "hot" -> "dot" -> "dog" -> "cog"，"hit" -> "hot" -> "lot" -> "log" -> "cog"
               标签：广度优先搜索，哈希表，字符串，回溯
        https://leetcode.cn/problems/word-ladder-ii/description/
    """

    def wordLadderII_126(self,beginWord:str,endWord:str,wordList:list) -> list:
        # 思路：从beginword开始，通过wordList构建一棵树，所有通向节点endWord的路径就是结果
        if endWord not in wordList:
            print('endWord not in wordList')
            return []

        # 先构建一个哈希表，每个单词是一个key，value是个列表，列表中都是跟key只有一个字母之差的单词
        wordDict = {}
        wordDict[beginWord] = []
        for word in wordList:
            c = 0
            for i in range(len(beginWord)):
                if beginWord[i] != word[i]:
                    c = c + 1
            if c == 1:
                wordDict[beginWord].append(word)
        for word1 in wordList:
            wordDict[word1] = []
            for word2 in wordList:
                c = 0
                for i in range(len(word1)):
                    if word1[i] != word2[i]:
                        c = c + 1
                if c == 1:
                    wordDict[word1].append(word2)
        print(wordDict)

        # 用广度遍历算法构建一个TreeNode树，当发现某个节点的childlist包含endWord的时候，遍历到该层结束，不需要更下一层了
        root = TreeNode(beginWord)
        queue = [root]
        layer = 0
        while queue:
            n = len(queue)
            layer = layer + 1
            # 先找出queue中所有val单词是否包含endWord，如果包含，就不要往queue里塞入下一层元素了
            flag = False
            for i in range(n):
                if queue[i].val == endWord:
                    flag = True

            # 如果本层queue中所有val单词不包含endWord，则把下一层读进来继续判断，否则跳出循环
            # 另外需考虑，假如就是没有路径，当遍历层数超过wordList元素个数+1时，也需要停止循环，因为这里没有判断路径中是否出现重复元素有可能死循环
            if not flag and layer <= len(wordList)+1:
                for j in range(n):
                    node = queue.pop(0)
                    childl = wordDict[node.val]
                    tmplist = [].copy()
                    for childword in childl:
                        node2 = TreeNode(childword)
                        tmplist.append(node2)
                    node.childlist = tmplist
                    queue.extend(tmplist)
            else:
                break
        print('广度遍历最大层数：',layer)
        print('深度遍历树：', root.DLRTraversal([]))

        resultlist = []
        # 用深度遍历算法找到所有叶子节点为endWord的路径
        # path:已经有的路径；currentword：当前待判断的节点；
        def recursion(path: list, currentnode: TreeNode):
            path.append(currentnode.val)
            if currentnode.val == endWord:
                resultlist.append(path.copy())
            elif currentnode.childlist != None and currentnode.childlist != []:
                for node in currentnode.childlist:
                    recursion(path, node)
            path.pop(-1)

        recursion([], root)
        print('最终结果：',resultlist)
        return resultlist

    """
    127. 单词接龙：给你两个单词 beginWord 和 endWord 和一个字典 wordList ，返回 从 beginWord 到 endWord 的 最短转换序列 中的 单词数目 。如果不存在这样的转换序列，返回 0 。
        示例 1：输入：beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
               输出：5
        标签：广度优先搜索，哈希表，字符串
        https://leetcode.cn/problems/word-ladder/description/
    """

    def wordLadder_127(self,beginWord:str,endWord:str,wordList:list) -> int:
        # 思路：这一题比126容易一些，去掉最后深度遍历获取路径的步骤，用广度遍历获取layer即可。126的广度遍历其实还有个缺陷，就是没有判断路径上是否有重复元素。
        # 这次改进一下，先构造一个单词节点之间存在连线的“图”，再根据图进行广度遍历，

        if endWord not in wordList:
            print('endWord not in wordList')
            return []

        # 先构建一个图，该图是一个list，每个list包含2个单词，表示这2个单词有一个字母之差，即这个图包含了所有的“边”。
        wordEdges = []
        tmplist = wordList.copy()
        tmplist.insert(0,beginWord)
        for i in range(len(tmplist)-1):
            for j in range(i+1,len(tmplist)):
                c = 0
                for k in range(len(beginWord)):
                    if tmplist[i][k] != tmplist[j][k]:
                        c = c + 1
                if c == 1:
                    wordEdges.append([tmplist[i],tmplist[j]])

        print(wordEdges)

        # 用广度遍历算法寻找图，当发现某条“边”节点的childlist包含endWord的时候，结束，不需要继续遍历了。
        # 这个算法还是有缺陷。tmpEdges存储了所有单词之间的“边”，已经用到的“边”就删掉避免死循环，
        # 但是从广度搜索算法本身来讲还不能回溯地判断单条路径是否有重复的边，所以这样构建出来的路径不全。仅仅可以得到正确到达endWord的“层数”而已。
        # 要想完整准确无回路地构造路径，还是需要深度遍历
        queue = [beginWord]
        layer = 0
        tmpEdges = wordEdges.copy()
        while queue:
            print(queue,tmpEdges)
            n = len(queue)
            layer = layer + 1
            # 先找出queue中所有val单词是否包含endWord，如果包含，就不要往queue里塞入下一层元素了
            flag = False
            for i in range(n):
                if queue[i] == endWord:
                    flag = True

            # 如果本层queue中所有val单词不包含endWord，则把下一层读进来继续判断，否则跳出循环
            # 另外需考虑，假如就是没有路径，当遍历层数超过wordList元素个数+1时，也需要停止循环，因为这里没有判断路径中是否出现重复元素有可能死循环
            if not flag and layer <= len(wordList)+1:
                for j in range(n):
                    node = queue.pop(0)
                    k = 0
                    while k <len(tmpEdges):
                        if node in tmpEdges[k]:
                            tmpEdges[k].remove(node)
                            queue.extend(tmpEdges[k])
                            tmpEdges.pop(k)
                        else:
                            k = k + 1
            else:
                break
        print('广度遍历最大层数：',layer)
        return layer

    """
    132. 分割回文串 II：给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是回文串。返回符合要求的 最少分割次数 。
         示例 1：输入：s = "aab"，输出：1，解释：只需一次分割就可将 s 分割成 ["aa","b"] 这样两个回文子串。
         标签：字符串，动态规划
         https://leetcode.cn/problems/palindrome-partitioning-ii/description/
   """

    def palindromePartitioningII_132(self,s:str)->int:
        # 思路1：笨办法，假如s的长度是n，那么最多有n-1个分法，也即分割成两个字母就是回文串了。
        # 分头从1试到n-1，看看是否符合条件，思路神似115题。只是这个方法时间复杂度较长。

        # 先定义一个子函数判断字符串是否回文串
        def isPalindromeString(ss:str)->bool:
            for i in range (len(ss)//2):
                if ss[i] != ss[len(ss)-i-1]:
                    return False
            return True

        # 先判断s是否回文：
        if isPalindromeString(s):
            return 0

        # 如果s不是回文，继续往下走
        # 先确定n的上限，以及初始化t的坐标列表idxoroginlist，以及最终划分方案结果列表：
        maxn = len(s) - 1
        idxoroginlist = list(range(maxn))
        resultlist = []
        # print('idxoroginlist', idxoroginlist)

        # 从把t分成1段开始，一步一步往上加。分成n段，需要在0~len(t)-2坐标中找出n-1个
        n = 1
        divideList = []
        while n <= maxn:
            idxcomb = itertools.combinations(idxoroginlist, n)
            for i in idxcomb:
                divideList.append(list(i))
            n = n + 1
        print(divideList)

        # 从小到大对每种分法判断是否回文
        for i in divideList:
            resultlist = []
            flag = True
            j = 0
            for k in range(len(i)):
                tmps = s[j:i[k]+1]
                resultlist.append(tmps)
                flag = flag and isPalindromeString(tmps)
                if not flag:
                    break
                else:
                    j = i[k] + 1
            tmps = s[j:len(s)]
            resultlist.append(tmps)
            flag = flag and isPalindromeString(tmps)
            if flag:
                break

        print(len(resultlist)-1,resultlist)
        # return len(resultlist)-1

        # 思路2：官解用的动态规划，性能好，但比较难于理解，尝试理解了很久……
        # 先用一个矩阵f[i][j]表示s[i:j]是否回文串
        n = len(s)
        f = [[True] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                f[i][j] = (s[i] == s[j]) and f[i + 1][j - 1]
        print(f)

        ret = list()
        ans = list()
        # 类似于深度遍历，对于每个s[i,n]中的字符j，如果s[i:j]也是回文串，就把它加入临时列表中，
        def dfs(i: int):
            # print('i',i,'ans',ans)
            if i == n:
                ret.append(ans[:])
                return

            for j in range(i, n):
                if f[i][j]:
                    ans.append(s[i:j + 1])
                    dfs(j + 1)
                    ans.pop()

        dfs(0)
        # 此时，ret里装的是所有的回文分割方案的组合，需要从中找出最小的分割方案，不一定是最后一个，所以要判断一下，这么看的话性能也不很高啊
        c = len(s)
        result = []
        for re in ret:
            if len(re) < c:
                c = len(re)
                result = re[:]

        print(result)
        return len(result)

    """
    135. 分发糖果：n 个孩子站成一排。给你一个整数数组 ratings 表示每个孩子的评分。你需要按照以下要求，给这些孩子分发糖果：
                 每个孩子至少分配到 1 个糖果。相邻两个孩子中，评分更高的那个会获得更多的糖果。请你给每个孩子分发糖果，计算并返回需要准备的 最少糖果数目 。
        示例 1：输入：ratings = [1,0,2]，输出：5，解释：你可以分别给第一个、第二个、第三个孩子分发 2、1、2 颗糖果。
        示例 2：输入：ratings = [1,2,2]，输出：4，解释：你可以分别给第一个、第二个、第三个孩子分发 1、2、1 颗糖果。第三个孩子只得到 1 颗糖果，这满足题面中的两个条件。
        标签：贪心，数组
        https://leetcode.cn/problems/candy/description/
    """

    def candyRatings_135(self,ratings:list)->int:
        # 思路1：笨办法，从评分最低的开始分糖果，如果周围有高分的，周围加一个糖果。只是需要多重循环，性能比较低

        # 先把ratings评价份去重排序，存到另一个列表中
        newratings = list(set(ratings))
        newratings.sort()

        # candyList存储每个孩子最终分到的糖果，初始化全部置为1
        n = len(ratings)
        candyList = [1] * n

        # 最低值开始，如果周围有高分的，周围加一个糖果
        for rating in newratings:
            # 每个rating列表元素都判断一遍
            for i in range(n):
                if ratings[i] == rating:
                    # print(max(0,i-1),i,min(i+1, n-1))
                    if ratings[max(0,i-1)] > ratings[i]:
                        candyList[max(0,i-1)] = candyList[i] + 1
                    if ratings[min(i+1, n-1)] > ratings[i]:
                        candyList[min(i+1, n-1)] = candyList[i] + 1

        print('思路1',sum(candyList),candyList)
        # return sum(candyList)

        # 思路2：贪心，依次判断ratings的每个评分，当判断到第i个元素时，看左边的元素分情况：
        # 若ratings[i-1] > ratings[i]：candyList[i-1]加1，并且向左一直判断，如果左边大于右边，左边加1，直到左边小于右边
        # 若ratings[i-1] < ratings[i]：candyList[i]等于candyList[i-1]加1
        # 这个方法跟思路1比性能，要看情况，假如极端情况ratings完全是倒序排列的，那就一样
        candyList = [1] * n
        for i in range(n):
            if ratings[max(0, i - 1)] > ratings[i]:
                candyList[max(0, i - 1)] = candyList[i] + 1
                for j in range(max(0, i - 1), 0, -1):
                    if ratings[j - 1] > ratings[j]:
                        candyList[j-1] = candyList[j] + 1
            if ratings[max(0, i - 1)] < ratings[i]:
                candyList[i] = candyList[max(0, i - 1)] + 1
        print('思路2', sum(candyList), candyList)
        return sum(candyList)

        # 官解能看懂，思路比较好，性能比上述都要高一些，我智商有限很难想到，暂不写了。

    """
    140. 单词拆分 II：给定一个字符串 s 和一个字符串字典 wordDict ，在字符串 s 中增加空格来构建一个句子，使得句子中所有的单词都在词典中。
                    以任意顺序 返回所有这些可能的句子。注意：词典中的同一个单词可能在分段中被重复使用多次。
        示例 1：输入:s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]，输出:["cats and dog","cat sand dog"]
        标签：字典树，记忆化搜索，数组，哈希表，字符串，动态规划，回溯
        https://leetcode.cn/problems/word-break-ii/description/
    """

    def wordBreakII_140(self,s:str,wordDict:list)->list:
        # 思路：类似于深度遍历搜索，设计一个递归函数，
        # 从s的第0个字符开始，依次往后加字符，判断单词是否在wordDict中，如果在，就把前面的单词记下来，继续调用函数判断后面的字符串

        resultlist = []
        def recursion1(path:list,ss:str):
            if ss == '':
                resultlist.append(path[:])
                return
            for i in range(1,len(ss)+1):
                if ss[:i] in wordDict:
                    path.append(ss[:i])
                    print(ss[:i], ss[i:], path, resultlist)
                    recursion1(path,ss[i:])
                    path.pop()

        # recursion1([],s)
        print(len(resultlist),resultlist)
        # return resultlist

        # 我就知道这个题没那么简单被我做出来，当s,wordDict特别长的情况下，如下述超长aaaaa的测试用例，时间性能就会无法容忍了。
        # 加一个剪枝策略，如果s[i:]字符串都没有匹配上的话，后续就不用匹配了，把s[i:]放到一个列表里，加入判断
        # 但跑了一遍下述超长aaaaa的测试用例，这个也很慢无法容忍。
        resultlist = []
        mismatchlist = []
        def recursion2(path:list,ss:str):
            if ss == '':
                resultlist.append(path[:])
                return
            flag = False
            for i in range(1,len(ss)+1):
                if ss[:i] in wordDict:
                    flag = True
                    path.append(ss[:i])
                    print(ss[:i], ss[i:], path, resultlist,'mismatchlist',mismatchlist)
                    if ss[i:] not in mismatchlist:
                        recursion2(path,ss[i:])
                        path.pop()
                    else:
                        break
            if not flag:
                mismatchlist.append(ss)

        if s in wordDict:
            resultlist.append(s)
        else:
            recursion2([],s)
        print('mismatchlist',mismatchlist)
        print(len(resultlist),resultlist)
        return resultlist


if __name__ == "__main__":
    ha = HardAlgorithm0_199()

    ha.wordBreakII_140("catsanddog", ["cat","cats","and","sand","dog"])
    ha.wordBreakII_140("pineapplepenapple", ["apple","pen","applepen","pine","pineapple"])
    ha.wordBreakII_140("aaa...aaa", ["a", "aa", "aaa", ..., "aaa...aaa"])
    # ha.wordBreakII_140("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    #                      ["a", "aa", "aaa", "aaaa", "aaaaa", "aaaaaa", "aaaaaaa", "aaaaaaaa", "aaaaaaaaa", "aaaaaaaaaa"])

    # ha.candyRatings_135([1,0,2])
    # ha.candyRatings_135([3,1,2,2])
    # ha.candyRatings_135([1,2,3,4])
    # ha.candyRatings_135([4,1,3,2,1])

    # ha.palindromePartitioningII_132('aabaabcdfdcbu')
    # ha.palindromePartitioningII_132('aaba')

    # ha.wordLadder_127("hit", "cog", ["hot","dot","dog","lot","log","cog"])
    # ha.wordLadder_127("hit", "cog", ["hot","dot","dog","lot","log"])

    # ha.wordLadderII_126("hit", "cog", ["hot","dot","dog","lot","log","cog"])
    # ha.wordLadderII_126("hit", "cog", ["hot","dot","dog","lot","log"])

    # ha.binaryTreeMaximumPathSum_124([1, 2, 3])
    # ha.binaryTreeMaximumPathSum_124([-10,9,20,None,None,15,7])

    # ha.bestTimeToBuyAndSellStockIII_123([3,3,5,0,0,3,1,4])

    # ha.distinctSubsequences_115("rabbbit", "rabbit")
    # ha.distinctSubsequences_115("babgbag",  "bag")

    # print(ha.scrambleString_87("great", "rgeat"))
    # print(ha.scrambleString_87("abcde", "caebd"))
    # print(ha.scrambleString_87("a", "a"))

    # ha.maximalRectangle_85([["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]])
    # ha.maximalRectangle_85([["0"]])
    # ha.maximalRectangle_85([["1"]])

    # ha.largestRectangleinHistogram_84( [2,1,5,6,2,3])

    # ha.minimumWindowSubstring_76("ADOBECODEBANC", "ABC")
    # ha.minimumWindowSubstring_76("a", "aa")
    # ha.minimumWindowSubstring_76("ADOBECODEBANC", "ABAC")
    # ha.minimumWindowSubstring_76_map("ADOBECODEBANC", "ABAC")

    # ha.textJustification_68(["This", "is", "an", "example", "of", "text", "justification."],16)

    # print(ha.validNumber_65('+3.14'))
    # print(ha.validNumber_65('-.9'))
    # print(ha.validNumber_65('3e+7'))
    # print(ha.validNumber_65('+6e-1'))
    # print(ha.validNumber_65('-123.456e789'))

    # print(ha.validNumber_65('1a'))
    # print(ha.validNumber_65('e3'))
    # print(ha.validNumber_65('99e2.5'))
    # print(ha.validNumber_65('--6'))
    # print(ha.validNumber_65('95a54e53'))

    # ha.permutationSequence_60(3,2)
    # ha.permutationSequence_60(5,13)

    # ha.nQueens_51(4)

    # ha.trappingRainWater_42([0,1,0,2,1,0,1,3,2,1,2,1])
    # ha.trappingRainWater_42([4,2,0,3,2,5])

    # ha.firstMissingPositive_41([1,2,0])
    # ha.firstMissingPositive_41([3, 4, -1, 1])

    # ha.sudokuSolver_37([ [5,3,None,None,7,None,None,None,None],[6,None,None,1,9,5,None,None,None],[None,9,8,None,None,None,None,6,None],
    #                      [8,None,None,None,6,None,None,None,3],[4,None,None,8,None,3,None,None,1],[7,None,None,None,2,None,None,None,6],
    #                      [None,6,None,None,None,None,2,8,None],[None,None,None,4,1,9,None,None,5],[None,None,None,None,8,None,None,7,9]])

    # ha.longestValidParenthesis_32("(()")
    # ha.longestValidParenthesis_32(")()())")
    # ha.longestValidParenthesis_32("")

    # ha.substringWithConcatenationOfAllWords_30("barfoothefoobarman",  ["foo","bar"])
    # ha.substringWithConcatenationOfAllWords_30("wordgoodgoodgoodbestword", ["word","good","best","word"])
    # ha.substringWithConcatenationOfAllWords_30("barfoofoobarthefoobarman", ["bar","foo","the"])

    # ha.reverseNodesInKGroup_25([1,2,3,4,5], 2)

    # ha.mergeKSortedLists_23([[1,4,5],[1,3,4],[2,6]])

    # print(ha.regularExpressionMatching_10(  "aa",  "a"))
    # print(ha.regularExpressionMatching_10(  "aa",  "a*"))
    # print(ha.regularExpressionMatching_10(  "fghior6",  ".*"))

    # ha.medianOfTwoSortedArrays_4([1,3], [2])
    # ha.medianOfTwoSortedArrays_4([1, 2], [3, 4])
    # ha.medianOfTwoSortedArrays_4([8,9,10], [1,2,3,4,5,6,7])