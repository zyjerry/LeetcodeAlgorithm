"""
    力扣算法题Python实践：https://leetcode.cn/problemset/algorithms/，可用于中学编程教学
    DATE        AUTHOR        CONTENTS
    2023-08-18  Jerry Chang   Create
"""
import math


class EasyAlgorithm200_299:
    """    构造函数，什么都不做    """

    def __init__(self):
        print('Hello World!')

    """
        205. 同构字符串：给定两个字符串 s 和 t ，判断它们是否是同构的。
            同构的定义：如果 s 中的字符可以按某种映射关系替换得到 t ，那么这两个字符串是同构的。
                      每个出现的字符都应当映射到另一个字符，同时不改变字符的顺序。
                      不同字符不能映射到同一个字符上，相同字符只能映射到同一个字符上，字符可以映射到自己本身。
            标签：哈希表，字符串
            https://leetcode.cn/problems/isomorphic-strings/
    """

    def isOmorphicStrings_205(self, s='', t=''):
        # 思路：逐个把字符串映射关系存到HashMap里，如果发现key已在HashMap里存在，但值不一样，则返回false
        # 需要注意的是，得用两个HashMap，分别存双向映射关系
        flag = True
        hm1 = {}
        hm2 = {}
        for i in range(len(s)):
            if s[i] not in hm1.keys():
                hm1[s[i]] = t[i]
            elif hm1[s[i]] != t[i]:
                flag = False
                break
            if t[i] not in hm2.keys():
                hm2[t[i]] = s[i]
            elif hm2[t[i]] != s[i]:
                flag = False
                break
        print(flag)

    """
        228. 汇总区间：给定一个  无重复元素 的 有序 整数数组 nums 。
             返回 恰好覆盖数组中所有数字 的 最小有序 区间范围列表 。
             也就是说，nums 的每个元素都恰好被某个区间范围所覆盖，并且不存在属于某个范围但不属于 nums 的数字 x 。
             列表中的每个区间范围 [a,b] 应该按如下格式输出："a->b" ，如果 a != b，"a" ，如果 a == b
             示例 1：输入：nums = [0,1,2,4,5,7]输出：["0->2","4->5","7"]
             标签：数组
             https://leetcode.cn/problems/summary-ranges/
    """

    def summaryRanges_228(self, nums=[]):
        # 思路：本质是判断每个元素是否连续，如果是连续则继续判断，如果不连续，就生成一个区间，下一步生成下一个区间
        finalList = []
        everyStr = str(nums[0]) + '->'
        for i in range(1, len(nums)):
            if nums[i] != (nums[i - 1] + 1):
                everyStr = everyStr + str(nums[i - 1])
                finalList.append(everyStr)
                everyStr = str(nums[i]) + '->'
            if i == (len(nums) - 1):
                everyStr = everyStr + str(nums[i])
                finalList.append(everyStr)
            print(i, everyStr)
        print(finalList)

    """
        231. 2 的幂：给你一个整数 n，请你判断该整数是否是 2 的幂次方。如果是，返回 true ；否则，返回 false 。
            标签：位运算，递归，数学
            https://leetcode.cn/problems/power-of-two/
    """

    def powerOfTwo_231(self, n=0):
        # 思路1、循环模2，只有每次模出来的结果都为0，才认为是2的幂次方，直至整除到1
        i = n
        flag = 0
        while i > 1:
            if (i % 2) != 0:
                flag = 1
                break
            else:
                i = i // 2
        if flag == 0:
            print(n, '是2的幂次方。')
        else:
            print(n, '不是2的幂次方。')

        # 思路2、通过位运算：
        # 心路历程一、2的幂次方数转换成2进制的话一定是1开头后面全是0，也就是说，它减1的话，转换成二进制就全部是1，
        # 所以，将它减1再按位取反，结果如果是0，则是2的幂次方，如果不是0则不是2的幂次方
        # 但我忘了，计算机中负数是以补码的形式存储的。那么，一个正数按位取反，就变成了一个负数，反之亦然，~x = -x-1，无法达到上述纯原码的效果
        # 例如：7（0000……111）按位取反后是1111……000，这是某个负数的补码，该补码-1（为1111……0111）再按位取反后为0000……1000，即为-8
        # 参考https://blog.csdn.net/zouxiaolv/article/details/99545024
        #    https://blog.csdn.net/love_gzd/article/details/85085084
        #    https://zhuanlan.zhihu.com/p/47719434
        # 心路历程二、参考leetcode官方答案：如果n是2的幂次方，那么n & (n - 1)必定为0，或者，n & (-n) 必定为 n，推导过程略
        i = ~(n - 1)
        if n & (n - 1) == 0 and n & (-n) == n:
            print(n, '是2的幂次方。')
        else:
            print(n, '不是2的幂次方。')

        # 思路3、假如n最大取2**31的话，如果n能够被2**31整除，那么也说明n是2的幂次方
        if (2 ** 31) % n == 0:
            print(n, '是2的幂次方。')
        else:
            print(n, '不是2的幂次方。')

    """
    242. 有效的字母异位词：给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。
        注意：若 s 和 t 中每个字符出现的次数都相同，则称 s 和 t 互为字母异位词。
        进阶: 如果输入字符串包含 unicode 字符怎么办？你能否调整你的解法来应对这种情况？
        标签：哈希表，字符串，排序
        https://leetcode.cn/problems/valid-anagram/
    """

    def validAnagram_242(self, s='', t=''):
        # 思路：用2个HashMap分别存储2个字符串，key是字符，value是字符出现的次数，再比较2个HashMap是否一样
        hms = {}
        hmt = {}
        # 先把s、t变成HashMap
        for i in range(len(s)):
            if s[i] not in hms.keys():
                hms[s[i]] = 1
            else:
                hms[s[i]] = hms[s[i]] + 1
            if t[i] not in hmt.keys():
                hmt[t[i]] = 1
            else:
                hmt[t[i]] = hmt[t[i]] + 1
        # 判断2个HashMap是否想等
        if hms == hmt:
            print(s, '和', t, '是异位词。')
        print(hms, hmt)

    """
    258. 各位相加：给定一个非负整数 num，反复将各个位上的数字相加，直到结果为一位数。返回这个结果。
        示例 1:输入: num = 38，输出: 2 ，
        解释: 各位相加的过程为：38 --> 3 + 8 --> 11，11 --> 1 + 1 --> 2，由于 2 是一位数，所以返回 2。
        进阶：你可以不使用循环或者递归，在 O(1) 时间复杂度内解决这个问题吗？
        标签：数学，数论，模拟
        https://leetcode.cn/problems/add-digits
    """

    def addDigits_258(self, num):
        # 思路1、正常循环判断
        if num == 0:
            print(0)
        else:
            flag = True
            tmpNum = num
            while flag:
                j = 0
                l = int(math.log10(tmpNum)) + 1
                for i in range(l):
                    j = j + tmpNum % 10
                    tmpNum = tmpNum // 10
                if (j // 10) == 0:
                    flag = False
                tmpNum = j
            print(tmpNum)

        # 思路2、需要用到数论中的同余理论。
        # num 与其各位相加的结果模 999 同余。重复计算各位相加的结果直到结果为一位数时，该一位数即为 num的数根，num与其数根模 9 同余。
        # 我们对 num 分类讨论： num 不是 9 的倍数时，其数根即为 num 除以 9 的余数。
        #                    num 是 9 的倍数时：
        #                        如果 num=0，则其数根是 0；
        #                        如果 num>0，则各位相加的结果大于 0，其数根也大于 0，因此其数根是 9。
        if num % 9 != 0:
            print(num % 9)
        elif num == 0:
            print(0)
        else:
            print(9)
        # 进一步，简洁一点，数根即为(num-1)%9+1。
        # 下面写法的意思是，如果num>0，则取(num - 1) % 9 + 1，否则如果num=0，则取0。
        # 因为假如num=0，那么(-1%9+1)等于9，不合题意
        print((num - 1) % 9 + 1 if num else 0)


if __name__ == "__main__":
    ea = EasyAlgorithm200_299()
    # ea.isOmorphicStrings_205('egg', 'add')
    # ea.isOmorphicStrings_205('foo', 'bar')
    # ea.isOmorphicStrings_205('paper', 'title')
    # ea.summaryRanges_228([0, 1, 2, 4, 5, 7])
    # ea.summaryRanges_228([0, 2, 3, 4, 6, 8, 9])
    # ea.powerOfTwo_231(63)
    # ea.powerOfTwo_231(64)
    # ea.validAnagram_242('absa', 'sbaa')
    ea.addDigits_258(48975620)
    ea.addDigits_258(38)
    ea.addDigits_258(0)
