"""
implement of UnionFind

TODO: add test
"""
from typing import List

Index = int


class UnionFind:
    def __init__(self, n: int):
        self._nodes = [-1 for _ in range(n)]

    def add_node(self):
        self._nodes.append(-1)

    def find(self, key: Index) -> Index:
        if self._nodes[key] < 0:
            return key
        else:
            return self.find(self._nodes[key])

    def size(self, key: Index) -> int:
        key_rep = self.find(key)
        return -1 * self._nodes[key_rep]

    def same(self, a: Index, b: Index) -> bool:
        a_rep = self.find(a)
        b_rep = self.find(b)
        return a_rep == b_rep

    def merge(self, a: Index, b: Index):
        if self.same(a, b):
            return self.find(a)

        a_rep = self.find(a)
        b_rep = self.find(b)

        self._nodes[a_rep] -= self.size(b)
        self._nodes[b_rep] = a_rep
        return a_rep

    def groups(self) -> List[List[Index]]:
        groups = []
        for index in range(len(self._nodes)):
            if self._nodes[index] >= 0:
                continue
            # rep
            groups.append([index])

        for target in range(len(self._nodes)):
            if self._nodes[target] < 0:
                continue
            for group in groups:
                rep = group[0]
                if self.same(rep, target):
                    group.append(target)
                    break

        return groups
