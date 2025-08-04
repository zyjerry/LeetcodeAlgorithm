"""
    力扣算法题Python实践：https://leetcode.cn/problemset/algorithms/，可用于中学编程教学
    DATE        AUTHOR        CONTENTS
    2025-08-03  Jerry Chang   Create
"""
from idlelib.debugger_r import restart_subprocess_debugger


# 主类，算法实现都在这里面
class HardAlgorithm0_199:
    """    构造函数，什么都不做    """

    def __init__(self):
        print('Hello World!')

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
        return resultlist









if __name__ == "__main__":
    ha = HardAlgorithm0_199()

    ha.substringWithConcatenationOfAllWords_30("barfoothefoobarman",  ["foo","bar"])
    ha.substringWithConcatenationOfAllWords_30("wordgoodgoodgoodbestword", ["word","good","best","word"])
    ha.substringWithConcatenationOfAllWords_30("barfoofoobarthefoobarman", ["bar","foo","the"])

    # ha.reverseNodesInKGroup_25([1,2,3,4,5], 2)

    # ha.mergeKSortedLists_23([[1,4,5],[1,3,4],[2,6]])

    # print(ha.regularExpressionMatching_10(  "aa",  "a"))
    # print(ha.regularExpressionMatching_10(  "aa",  "a*"))
    # print(ha.regularExpressionMatching_10(  "fghior6",  ".*"))

    # ha.medianOfTwoSortedArrays_4([1,3], [2])
    # ha.medianOfTwoSortedArrays_4([1, 2], [3, 4])
    # ha.medianOfTwoSortedArrays_4([8,9,10], [1,2,3,4,5,6,7])