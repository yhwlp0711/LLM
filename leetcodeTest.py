from typing import List


class Solution:
    def trap(self, height: List[int]) -> int:
        s = 0
        f = 1
        n = len(height)
        res = 0
        while f < n - 1:
            if height[f] < height[s]:
                res += height[s] - height[f]
            else:
                s = f
            f += 1
        return res


A = Solution()
print(A.trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))
