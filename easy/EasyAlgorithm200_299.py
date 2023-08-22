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

    """
        263. 丑数：丑数 就是只包含质因数 2、3 和 5 的正整数。
             给你一个整数 n ，请你判断 n 是否为 丑数 。如果是，返回 true ；否则，返回 false 。
             标签：数学，数论
             https://leetcode.cn/problems/ugly-number/
    """

    def uglyNumber_263(self, n=0):
        # 思路：如果一个数只包含质因数 2、3、5，就是说，不包含其他的质因数，那么该数一定可以写成：(2**a)(3**b)(5**c)
        # 只能对n反复除以2, 3, 5，直到n不再包含质因数。若剩下的数等于1，则说明n不包含其他质因数，是丑数；否则，不是丑数。
        l235 = [2, 3, 5]
        num = n
        for i in l235:
            while num % i == 0:
                num = num // i
        if num == 1:
            print(n, '是丑数。')
        else:
            print(n, '不是丑数。')

    """
        268. 丢失的数字：给定一个包含 [0, n] 中 n 个数的数组 nums ，找出 [0, n] 这个范围内没有出现在数组中的那个数。
            进阶：你能否实现线性时间复杂度、仅使用额外常数空间的算法解决此问题?
            相关标签：位运算，数组，哈希表，数学，二分查找，排序
            https://leetcode.cn/problems/missing-number/
            吐槽：官方解答中前2个方法又是排序又是哈希的没必要搞这么复杂
    """

    def missingNumber_268(self, nums=[]):
        # 思路1、暴力循环：先初始化一个包含0~n共n+1个元素的列表，然后循环判断nums中元素，匹配到的从初始列表中删除，最后剩的那个就是丢失的数
        # 时间复杂度：O(2n)
        totalL = []
        for i in range(len(nums) + 1):
            totalL.append(i)
        for i in range(len(nums)):
            totalL.remove(nums[i])
        print(totalL[0])

        # 思路2、一次循环，看看数字是否在nums里，这样看的话上面那个算法好蠢，时间复杂度：O(n+1)
        for i in range(len(nums) + 1):
            if i not in nums:
                print(i)
                break

        # 思路3、将0~n这n+1个数字再次加到数组里，那么只有丢失的那个数出现了1次，其他数都出现2次。
        # 根据异或的计算特征，把所有数据异或起来结果就是丢失的数字，参考136题
        doubleNums = nums.copy()
        for i in range(len(nums) + 1):
            doubleNums.append(i)
        xor = doubleNums[0]
        for i in range(1, len(doubleNums)):
            xor = xor ^ doubleNums[i]
        print(xor)

        # 思路4、把0~n这n+1个数字累积相加，再把nums中的数据累计相加，两个数字相减，就是丢失的数字
        totalNum = len(nums) * (len(nums) + 1) / 2
        realNum = 0
        for i in range(1, len(nums)):
            realNum = realNum + nums[i]
        print(int(totalNum - realNum))

    """
        283. 移动零：给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
            请注意 ，必须在不复制数组的情况下原地对数组进行操作。
            示例 1:输入: nums = [0,1,0,3,12]，输出: [1,3,12,0,0]
            进阶：你能尽量减少完成的操作次数吗？
            标签：数组，双指针
            https://leetcode.cn/problems/move-zeroes/
    """

    def moveZeroes_283(self, nums=[]):
        # 思路：按顺序，碰到0，就append，再把当前的0删掉
        # 力扣官方解答，是用2个指针完成的，这里忽略
        i = 0
        numofz = 0
        while i < len(nums) - numofz:
            if nums[i] == 0:
                nums.append(0)
                nums.remove(0)
                numofz = numofz + 1
            else:
                i = i + 1
        print(nums)

    """
        290. 单词规律：给定一种规律 pattern 和一个字符串 s ，判断 s 是否遵循相同的规律。
            这里的 遵循 指完全匹配，例如， pattern 里的每个字母和字符串 s 中的每个非空单词之间存在着双向连接的对应规律。
            示例1:输入: pattern = "abba", s = "dog cat cat dog"，输出: true
            标签：哈希表，字符串
            https://leetcode.cn/problems/word-pattern/
    """

    def wordPattern_290(self, pattern='', s=''):
        # 思路：先把s切割成列表，再新建一个HashMap，存储pattern每个字母和s中单词的映射关系，循环诸葛判断，假如都符合映射关系，即True
        # 先把s切割成列表
        strList = []
        while len(s) > 0 and s.find(' ') != -1:
            strList.append(s[:s.find(' ')])
            s = s[s.find(' ') + 1:]
        strList.append(s)

        # 新建一个HashMap，循环判断
        hm = {}
        flag = True
        i = 0
        while flag and i < len(pattern):
            if pattern[i] not in hm.keys():
                hm[pattern[i]] = strList[i]
            elif hm[pattern[i]] != strList[i]:
                flag = False
            i = i + 1
        print(flag)

    """
        292. Nim 游戏：你和你的朋友，两个人一起玩 Nim 游戏：桌子上有一堆石头。你们轮流进行自己的回合， 你作为先手 。
            每一回合，轮到的人拿掉 1 - 3 块石头。拿掉最后一块石头的人就是获胜者。假设你们每一步都是最优解。
            请编写一个函数，来判断你是否可以在给定石头数量为 n 的情况下赢得游戏。如果可以赢，返回 true；否则，返回 false 。
            标签：数学，博弈
            https://leetcode.cn/problems/nim-game/
    """

    def nimGame_292(self, n=1):
        # 思路：先找找规律：假设从头开始：
        # 桌上有1~3个石头：甲必胜；
        # 桌上有4个石头：甲必输；
        # 桌上有5个石头：甲先拿1个，那么无论乙拿1~3个，甲必胜。
        # 桌上有6个石头：甲需要思考，自己拿过之后，确保桌上还剩4个，那么乙无论拿几个自己都能赢，那么，甲只能拿2个必胜。
        # 桌上有7个石头：甲需要思考，自己拿过之后，确保桌上还剩4个，那么乙无论拿几个自己都能赢，那么，甲只能拿3个必胜。
        # 桌上有8个石头：无论甲拿几个，只要乙拿完后桌上剩4个石头，那么甲必输。
        # 桌上有9个石头：甲需要思考，当自己拿过一次石头后，桌上千万不能剩5、6、7个，此时乙先手的话必胜。那么甲只能拿1个必胜，对于乙来说8个石头必输。
        # 桌上有10个石头：甲需要思考，当自己拿过一次石头后，桌上千万不能剩5、6、7、9个，此时乙先手的话必胜。那么甲只能拿2个必胜，对于乙来说8个石头必输。
        # ……
        # 经过上述摸索，可以得出规律，当有n个石头时，甲能否获胜的关键是，甲分别拿掉1~3个石头后，是否有乙必输的情况，有的话，甲必胜，没有的话，甲必输。
        # 换成程序的表达，即：初始化一个列表，从1开始判断甲是否必胜，到n；
        l = [True, True, True, True, False]  # 先初始化0~4个石头
        for i in range(5, n + 1):
            if l[i - 1] and l[i - 2] and l[i - 3]:
                l.append(False)
            else:
                l.append(True)
        print(l[n])

        # 进一步推理，基本可以看出如果堆里的石头数目为 4 的倍数时，你一定会输掉游戏。
        # 因为无论你取多少石头，对方总有对应的取法，让剩余的石头的数目继续为 4 的倍数。
        # 对于你或者你的对手取石头时，显然最优的选择是当前己方取完石头后，让剩余的石头的数目为 4 的倍数。
        # 假设当前的石头数目为 x，如果 x 为 4 的倍数时，则此时你必然会输掉游戏；
        # 如果 x 不为 4 的倍数时，则此时你只需要取走 x % 4 个石头时，则剩余的石头数目必然为 4 的倍数，从而对手会输掉游戏。
        # 所以可以直接用程序表达为：
        print(n % 4 != 0)


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
    # ea.addDigits_258(48975620)
    # ea.addDigits_258(38)
    # ea.addDigits_258(0)
    # ea.uglyNumber_263(15)
    # ea.missingNumber_268([0, 2, 3])
    # ea.moveZeroes_283([0, 1, 0, 3, 12])
    # ea.moveZeroes_283([0, 0, 0, 3, 5, 0, 9, 12])
    # ea.wordPattern_290('abba', 'dog cat cat dog')
    ea.nimGame_292(8)
