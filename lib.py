import copy
from manim import *
import random 
from typing import Tuple

class Formula:
    def __init__(self, clauses: List[List[int]]):
        self.clauses = copy.deepcopy(clauses)
        self.tex = []
        self.math_tex = []
        self.literal_indices = []
        self.active = []
        for clause in clauses:
            self.tex.append([])
            self.literal_indices.append([])
            for id, literal in enumerate(clause):
                if id > 0:
                    self.tex[-1].append("\lor")
                self.tex[-1].append(("" if literal > 0 else "\lnot ") 
                                         + "x_{" + str(abs(literal)) + "}")
                self.literal_indices[-1].append(len(self.tex[-1]) - 1)
            self.math_tex.append(MathTex(*self.tex[-1]))
            self.active.append([True for _ in range(len(self.math_tex[-1]))])

    def find_clause(self, assignment):
        for id, clause in enumerate(self.clauses):
            if all(assignment[abs(lit)-1] == (1 if lit < 0 else 0) for lit in clause):
                   return id
        return None

class BinaryTree:
    def __init__(self, n : int, tp: str = "random", label_range: int = -1):
        self.n = n
        self.label_range = n // 2 if label_range == -1 else label_range
        label_range = self.label_range
        assert n % 2 == 1
        self.parent = [-1 for _ in range(n)]
        self.leaf_labels = dict()
        if tp == "random" or tp == "bamboo":
            leaves = [0]
            filled = 1
            random.seed(14323)
            while filled < self.n:
                if tp == "random":
                    t = random.randint(0, len(leaves) - 1)
                    leaves[t], leaves[-1] = leaves[-1], leaves[t]
                leaf = leaves.pop()
                for v in [filled, filled + 1]:
                    self.parent[v] = leaf
                    
                leaves.append(filled)
                leaves.append(filled + 1)
                filled += 2
            self.leaves = leaves

        if tp == "full":
            fullheight = 1
            while 2**(fullheight + 1) - 1 < self.n:
                fullheight += 1
            assert 2**(fullheight + 1) -1 == self.n
            self.leaves = list(range(2**fullheight, self.n))
            self.parent = [-1 if i == 0 else (i-1)//2 for i in range(self.n)]
        
        self.height = [-1 for _ in range(n)]
        self.bit = [-1 for _ in range(n)]
        self.height[0] = 0
        self.label = [random.randint(1, label_range) for _ in range(n)]
        self.layout = dict()
        self.layout[0] = np.array([0,0,0])
        for v in range(1, self.n):
            label_set = set()
            leaf = self.parent[v]
            cur = leaf
            while cur != -1:
                label_set.add(self.label[cur])
                cur = self.parent[cur]
            self.height[v] = self.height[leaf] + 1
            self.bit[v] = v & 1
            self.layout[v] = self.layout[leaf] + DOWN * 1.5 +\
                (LEFT if self.bit[v] == 0 else RIGHT) * (n * 0.53**self.height[v])
            while self.label[v] in label_set:
                self.label[v] = random.randint(1, max(label_range, self.height[v] + 1))
        if tp == "full":
            for i in range(self.n):
                self.layout[i][0] *= 0.2
        self.leaves = sorted(list(self.leaves),key=lambda s: self.layout[s][0])
        
    def lca(self, u, v):
        assert 0 <= u < self.n 
        assert 0 <= v < self.n
        if self.height[u] > self.height[v]:
            u,v = v,u
        while self.height[v] > self.height[u]:
            v = self.parent[v]
        while u != v:
            u = self.parent[u]
            v = self.parent[v]
        return u
    
    def get_assignment(self, v: int):
        assignment = [-1 for _ in range(self.n)]
        cur = v
        while cur > 0:
            assignment[self.label[cur]] = self.bit[cur]
            cur = self.parent[cur]
        return assignment
    
    def get_edges(self) -> List[Tuple[int, int]]:
        return [(i, self.parent[i]) for i in range(1, self.n)]
    
    def get_tex_labels(self):
        return {i: MathTex("x_{" + str(self.label[i]) + "}", color=BLACK) 
                if i not in self.leaves 
                else MathTex(str(self.leaf_labels[i]) 
                             if i in self.leaf_labels 
                             else "\\bot", color=BLACK) 
                for i in range(self.n)}
    
    def labels_for_graph(self, graph):
        for u, v in self.get_edges():
            t = self.bit[u]
            yield Tex("$" + str(t) + "$").move_to(graph.edges[u,v].get_center())\
                .scale(0.5).add_background_rectangle(BLACK, opacity=0.9)
	
