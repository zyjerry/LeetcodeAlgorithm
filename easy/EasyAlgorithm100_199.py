"""
    力扣算法题Python实践：https://leetcode.cn/problemset/algorithms/，可用于中学编程教学
    DATE        AUTHOR        CONTENTS
    2023-08-17  Jerry Chang   Create
"""


class EasyAlgorithm100_199:
    """    构造函数，什么都不做    """

    def __init__(self):
        print('Hello World!')

    """
        118. 杨辉三角：给定一个非负整数 numRows，生成「杨辉三角」的前 numRows 行。在「杨辉三角」中，每个数是它左上方和右上方的数的和。
            标签：数组，动态规划
            https://leetcode.cn/problems/pascals-triangle/
    """

    def pascalsTriangle_118(self, numRows=1):
        # 思路：只有动态规划一种方法了
        # 第一层数组包含numRows个元素
        rows = [[1]]
        for i in range(1, numRows):
            lines = [1]
            for j in range(1, i - 1):
                lines.append(rows[i - 1][j - 1] + rows[i - 1][j])
            lines.append(1)
            rows.append(lines)
        print(rows)

    """
        121. 买卖股票的最佳时机：给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
            你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
            返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 0 。
            标签：数组，动态规划
            https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/
    """

    def bestTimeToBuyAndSellStock_121(self, prices=[]):
        # 思路1：野蛮粗暴法，双重循环轮询所有元素，找出差值最大的那个
        maxprice = 0
        beststart = 0
        bestend = 0
        for i in range(len(prices)):
            for j in range(i, len(prices)):
                if maxprice < (prices[j] - prices[i]):
                    maxprice = prices[j] - prices[i]
                    beststart = i
                    bestend = j
        print('最佳买卖点和价格差为：', beststart, bestend, maxprice)

        # 思路2：动态规划，从0开始逐步判断，只需要一重循环。但这种情况只能判断出最大获益，不能定位到具体买卖点
        maxprice = 0
        beststart = 0
        bestend = 0
        for i in range(1, len(prices)):
            if (prices[i] - prices[beststart]) > maxprice:
                bestend = i
                maxprice = prices[i] - prices[beststart]
            elif prices[i] < prices[beststart]:
                beststart = i
        print('最佳价格差为：', maxprice)

    """
        125. 验证回文串：如果在将所有大写字符转换为小写字符、并移除所有非字母数字字符之后，短语正着读和反着读都一样。
                       则可以认为该短语是一个 回文串 。字母和数字都属于字母数字字符。
                       给你一个字符串 s，如果它是 回文串 ，返回 true ；否则，返回 false 。
            标签：双指针，字符串
            https://leetcode.cn/problems/valid-palindrome/
    """

    def validPalindrome_125(self, s):
        # 思路1、简单粗暴，先把字符串反转，再比较
        str = ''
        for i in range(len(s)):
            str = str + s[len(s) - i - 1]
        if s == str:
            print('该字符串是回文串。')
        else:
            print('该字符串不是回文串。')

        # 思路2、模拟双指针，只需循环字符数一半就好了
        flag = 0
        for i in range(int(len(s) / 2)):
            if s[i] != s[len(s) - i - 1]:
                flag = 1
                break
        if flag == 0:
            print('该字符串是回文串。')
        else:
            print('该字符串不是回文串。')

    """
        136. 只出现一次的数字：给你一个 非空 整数数组 nums ，除了某个元素只出现一次以外，其余每个元素均出现两次。
             找出那个只出现了一次的元素。你必须设计并实现线性时间复杂度的算法来解决此问题，且该算法只使用常量额外空间。
             标签：位运算，数组
             https://leetcode.cn/problems/single-number/
    """

    def singleNumber_136(self, nums=[]):
        # 思路1：设计一个HashMap，依次循环判断每个元素，如果在HashMap的key中，就把该key删除，最后剩下来的那个key就是结果
        singleDict = {}
        for i in range(len(nums)):
            if nums[i] in singleDict:
                singleDict.pop(nums[i])
            else:
                singleDict[nums[i]] = 0
        print(singleDict.keys())

        # 思路2：这个牛叉了，利用异或运算的几个特点：
        # 一个数和 0 做 XOR 运算等于本身：a⊕0 = a
        # 一个数和其本身做 XOR 运算等于 0：a⊕a = 0
        # XOR 运算满足交换律和结合律：a⊕b⊕a = (a⊕a)⊕b = 0⊕b = b
        # 故而在以上的基础条件上，将所有数字按照顺序做异或运算，最后剩下的结果即为唯一的数字
        xor = nums[0]
        for i in range(1, len(nums)):
            xor = xor ^ nums[i]
        print(xor)

    """
        168. Excel表列名称：给你一个整数 columnNumber ，返回它在 Excel 表中相对应的列名称。
            例如：A -> 1，B -> 2，C -> 3...Z -> 26，AA -> 27，AB -> 28 ...
            标签：数学，字符串
            https://leetcode.cn/problems/excel-sheet-column-title/
    """

    def excelSheetColumnTitle_168(self, columnNumber):
        # 思路：就相当于算26进制呗，先定一个HashMap，key是1~26，value是A~Z，然后不停地模26
        tbl = {0: 'Z', 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J',
               11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T',
               21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z'
               }
        tmpNum = columnNumber
        s = ''
        if tmpNum < 27:
            s = tbl[tmpNum]
        else:
            while tmpNum > 0:
                s = tbl[tmpNum % 26] + s
                tmpNum = int((tmpNum - 1) / 26)
        print(s)

    """
        169. 多数元素：给定一个大小为 n 的数组 nums ，返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。
             你可以假设数组是非空的，并且给定的数组总是存在多数元素。
             进阶：尝试设计时间复杂度为 O(n)、空间复杂度为 O(1) 的算法解决此问题。
             标签：数组，哈希表，分治，计数，排序
             https://leetcode.cn/problems/majority-element/
    """

    def majorityElement_169(self, nums=[]):
        # 思路：创建一个HashMap，记录每个元素出现的次数，当发现次数>1时，把该元素存到另一个列表里
        hashDict = {}
        for i in range(len(nums)):
            if nums[i] not in hashDict.keys():
                hashDict[nums[i]] = 1
            else:
                hashDict[nums[i]] = hashDict[nums[i]] + 1
            if hashDict[nums[i]] == int(len(nums) / 2):
                print(nums[i])
                return

    """
        190. 颠倒二进制位：颠倒给定的 32 位无符号整数的二进制位。输入是一个长度为 32 的二进制字符串，输出是十进制数字。
            进阶: 如果多次调用这个函数，你将如何优化你的算法？
            标签：位运算，分治
            https://leetcode.cn/problems/reverse-bits/solutions/685436/dian-dao-er-jin-zhi-wei-by-leetcode-solu-yhxz/
    """

    def reverseBinaryNum_190(self, binStr=''):
        # 思路1、简单粗暴循环一遍颠倒字符顺序，再转换成数字
        newStr = ''
        for i in range(len(binStr)):
            newStr = newStr + binStr[len(binStr) - i - 1]
        newStr = '0b' + newStr
        print(newStr, int(newStr, 2), '\n')

        # 思路2、分治方法：
        # 若要翻转一个二进制串，可以将其均分成左右两部分，对每部分递归执行翻转操作，然后将左半部分拼在右半部分的后面，即完成了翻转。
        # 由于左右两部分的计算方式是相似的，利用位掩码和位移运算，我们可以自底向上地完成这一分治流程。
        # 但是这里也有个局限，就是，输入字符串长度必须要是2的幂次方
        newNum = int(binStr, 2)
        M1 = 0x55555555  # 01010101010101010101010101010101
        M2 = 0x33333333  # 00110011001100110011001100110011
        M4 = 0x0f0f0f0f  # 00001111000011110000111100001111
        M8 = 0x00ff00ff  # 00000000111111110000000011111111
        print(bin(newNum), newNum)
        n1 = newNum >> 1 & M1
        n2 = (newNum & M1) << 1
        newNum = n1 | n2  # 奇偶位互换，这里用了位运算和位的左移右移，理解起来比较难
        print(bin(newNum), newNum)
        n1 = newNum >> 2 & M2
        n2 = (newNum & M2) << 2
        newNum = n1 | n2  # 两位两位互换，原理同上
        print(bin(newNum), newNum)
        n1 = newNum >> 4 & M4
        n2 = (newNum & M4) << 4
        newNum = n1 | n2  # 四位四位互换，原理同上
        print(bin(newNum), newNum)
        n1 = newNum >> 8 & M8
        n2 = (newNum & M8) << 8
        newNum = n1 | n2  # 八位八位互换，原理同上
        print(bin(newNum), newNum)
        n1 = newNum >> 16
        n2 = newNum << 16
        newNum = n1 | n2  # 十六位十六位互换，最后一次直接交换，就不需要位运算了。
        # 但是python右移的时候右边会补0，所以要除一下把补的0去掉
        print(bin(n1), bin(n2))


if __name__ == "__main__":
    ea = EasyAlgorithm100_199()
    # ea.pascalsTriangle_118(8)
    # ea.bestTimeToBuyAndSellStock([3, 5, 67, 1, 9, 80, 1, 1])
    # ea.validPalindrome_125('hgfiifgh')
    # ea.singleNumber_136([1, 5, 87, 9, 9, 5, 87, 7, 1])
    # ea.excelSheetColumnTitle_168(78)
    # ea.majorityElement_169([1, 2, 4, 56, 8, 8, 9, 9, 9, 9, 9, 9, 9])
    ea.reverseBinaryNum_190('10110101010000010101010000101010')
