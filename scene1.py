from manim import *
import copy
import networkx as nx
from itertools import product, combinations
from lib import Formula, BinaryTree
import random
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService

def generate_girth(n, girth):
    outer_cycle = list(range(1, girth + 1))
    edges = [(i, i % girth + 1) for i in outer_cycle]
    cur_n = girth
    while cur_n < n:
        i = random.randint(0, len(outer_cycle) - 1)
        il = outer_cycle[(i + len(outer_cycle) - 1) % len(outer_cycle)]
        ir = outer_cycle[(i + len(outer_cycle) + 1) % len(outer_cycle)]
        inner = [cur_n + i + 1 for i in range(girth - 3)]
        cur_n += girth - 3
        edges.extend([(il, inner[0]), (ir, inner[-1])] + list(zip(inner[:-1], inner[1:])))
        outer_cycle = outer_cycle[:i] + inner + (outer_cycle[i+1:] if i < len(outer_cycle) - 1 else [])
    return cur_n, edges
        
def color_graph_greedily(n, edges): #assuming 1-indexing of the nodes
    colors = {i: -1 for i in range(1,n+1)}
    colors[1] = 0
    connect = [[] for i in range(n+1)]
    for u, v in edges:
        connect[u].append(v)
        connect[v].append(u)
    for i in range(2, n+1):
        excluded = set()
        for v in connect[i]:
            excluded.add(colors[v])
        for j in range(n):
            if j not in excluded:
                colors[i] = j
                break
        assert colors[i] != -1
    return colors
                

class ColoringSlide(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        caption = Tex("Chromatic number $\\chi(G)$")
        random.seed(12345432)
        n, edges = generate_girth(40, 5)
        print(edges)
        node_color = color_graph_greedily(n, edges)
        pallette = [RED_C, BLUE_C, GREEN_C, YELLOW_C]
        graph = Graph(list(range(1, n + 1)), edges, layout="kamada_kawai",
						vertex_config={x: {"fill_color": pallette[node_color[x]]} for x in range(1, n + 1)},
						layout_scale=2)
        with self.voiceover(""" 
                      The chromatic number of a graph is the smallest number of colors
                      one needs to color the nodes of the graph such that no edge connects
                      two nodes of the same color.
                      """):
            caption.move_to(UP * 3)
            graph.next_to(caption, direction=DOWN)
            self.play(Write(caption))
            self.add(graph)
            self.play(Wiggle(graph))
        with self.voiceover("""
							If the graph has chromatic number k, can we find a size big-O of k 
                            subgraph with the same chromatic number?
                            """):
            question = Tex("Is there a subgraph $H$ of $G$ with $O(\\chi(G))$ nodes and $\\chi(H) = \\chi(G)$?")\
                .scale(0.7).next_to(graph, direction=DOWN)
            question.add_background_rectangle(BLACK, opacity=0.9)
            self.add(question)
        with self.voiceover("""
							The answer is no, since graphs can have arbitrary large girth and chromatic number.
                            For example the graph on the screen has chromatic number 3 and girth 5, so
                            no graph of size 4 has cycles and thus have chromatic number 2.
                            """):
            answer = Tex("[Erd\\\"os, 1959] No, can have arbitrary large girth and chromatic number.").scale(0.7).next_to(question, direction=DOWN)
            answer.add_background_rectangle(BLACK, opacity=0.9)
            self.add(answer)
            for subgraph in [[1,2,3,4], [5,6,7,8], [18,19,20,21]]: 
                group = VGroup(*[graph[k] for k in subgraph], *[graph.edges[i,j] for i,j in edges if i in subgraph and j in subgraph])
                self.play(Indicate(group))
            
			
            
class VertexCoverSlide(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        with self.voiceover(""" 
                      A vertex cover is a set of nodes such that every edge in the graph is
                      incident to one of them
                      """):
            
            
            graph = Graph(list(range(10)), list(combinations(range(10), 2)), layout="circular")
            self.play(Create(text))
            self.add(graph)
            
    

################ OLD SLIDES FOR REFERENCE

class SATSlide(VoiceoverScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        with self.voiceover("""The guiding question in this talk is a very practical one.
                            How do we actually solve the SAT problem when we have to?
                            """):
            text = Text("How do we actually solve SAT?", font_size=30)
            self.play(FadeIn(text))
        arrow = Arrow(start=LEFT, end=RIGHT)
        phi = MathTex("\\varphi").next_to(arrow, LEFT)        
        bool_tex = MathTex("\\{","\\text{sat}",",", "\\text{unsat}", "\\}").next_to(arrow, RIGHT)
        with self.voiceover("""In the SAT problem we are given a boolean formula, let us
                            assume that it is a CNF.
                            """):
            self.play(text.animate.shift(UP * 2),  FadeIn(phi))
        with self.voiceover("""We are then asked if this formula is satisfiable or not,
                            that is, can one set the variables so all the clauses of the CNF
                            are satisfied.
                            """):
            self.play(GrowArrow(arrow))
            self.play(FadeIn(bool_tex))
        formula1 = Formula([[1, 2], [-1]])
        g_f1 = VGroup(*formula1.math_tex).arrange(direction=DOWN, buff=0.1).move_to(phi.get_center() + LEFT*0.5)
        formula2 = Formula([[1, 2, 3], [-1, -2, -3], [1, -2], [2, -3], [3, -1]])
        g_f2 = VGroup(*formula2.math_tex).arrange(direction=DOWN, buff=0.1).move_to(phi.get_center() + LEFT)
        with self.voiceover("""This formula is satisfiable: set x one to zero and x two to one.
                            Here the outer conjunction is implicit.
                            """):
            self.play(ReplacementTransform(phi, g_f1))
            self.play(Indicate(bool_tex[1]))
        with self.voiceover("""And this is unsatisfiable, no matter how we set the variables,
                            one of the clauses will be falsified.
                            """):
            self.play(ReplacementTransform(g_f1, g_f2))
            self.play(Indicate(bool_tex[3]))

class Experiment(Scene):
    def construct(self):
        s = [Square().scale(0.5) for _ in range(10)]
        strip = VGroup(*s).arrange(buff=0)
        self.play(Transform(strip, Circle()))


class TreeLikeResolutionDef(VoiceoverScene, MovingCameraScene):
    def construct(self):
        self.set_speech_service(GTTSService())
        #formula = Formula([[1, -2], [-1, 3], [-3, 4, -5], [-5], [-4, 5], [2, 5], [-3]])
        formula = Formula([[1, 2, 3], [-1, -2, -3], [1, -2], [2, -3], [3, -1]])
        n = 3
        assignments = list(product([0,1], repeat=n))
        texts = [MathTex(*list(map(str, reversed(assignment)))).scale(0.3) for assignment in assignments]
        group = VGroup(*texts)\
                 .arrange(buff=0.1, direction=DOWN)\
                 .scale_to_fit_height(5).shift(UP*0.5)
        formula_g = VGroup(*formula.math_tex).arrange(buff=1, direction=RIGHT).scale(0.5).move_to(UP*3.5)
        with self.voiceover("Let us see how we can check this formula on satisfiability."):
            self.play(Write(formula_g))
        with self.voiceover("The simplest way is to enumerate all possible assignments to the " +
                            "five variables of the formula."):
            self.add(group)
            self.play(group.animate.move_to(LEFT*5))
        to_play = []
        falsified_clause = []
        clauses_to_fade = []
        falsified_clause_exp = []
        for assignment, text in zip(assignments, texts):
            y = text.get_center()[1]
            cur_fals_clause = None
            cur_fals_clause_exp = None
            for clause, clause_tex in zip(formula.clauses, formula.math_tex):
                x = clause_tex.get_center()[0]
                p = np.array([x,y,0])
                is_sat = any(assignment[abs(lit) - 1] == (0 if lit < 0 else 1) for lit in clause)
                col_clause = clause_tex.copy()
                col_clause.stroke_color = col_clause.color = GREEN if is_sat else RED
                col_clause.move_to(p)
                to_play.append(col_clause)
                falsified = False
                if not is_sat and cur_fals_clause is None:
                    cur_fals_clause = col_clause
                    cur_fals_clause_exp = clause
                    falsified = True
                if not falsified:
                    clauses_to_fade.append(col_clause)
            falsified_clause.append(cur_fals_clause)
            falsified_clause_exp.append(cur_fals_clause_exp)
        to_play.sort(key=lambda x: x.get_center()[0] - x.get_center()[1])
        with self.voiceover("For every assignment we see what clauses are satisfied by it (in green)"
                             +" and falsified by it (in red)"):
            self.play(Create(VGroup(*to_play)))
        with self.voiceover("For this particular formula we see that for every assignment at least one of"
                            + " the clauses is falsified, so this formula is unsatisfiable."):
            self.play(*[Indicate(f) for f in falsified_clause])
        with self.voiceover("This outlook suggests considering the clause search relation or problem"
                            + " where given an assignment we are to find a clause of a CNF falsified by it"):
            self.play(*[Transform(a, a.copy().set_opacity(0.3)) for a in to_play])
            theor_rel = MathTex("\\mathrm{Search}_{\\varphi} = \{(x, C) \\mid x \in \\{0,1\\}^n;\; C \\in \\varphi;\; C(x) = 0\\}").scale(0.7)
            self.play(Write(theor_rel))
        with self.voiceover("Computational complexity of this relation in various models is the object of study of "
                            +" proof complexity."):
            self.play(FadeOut(theor_rel))
        ### Drawing the initial tree
        def get_height(x):
            cur = x
            t = 1
            while cur > 0:
                t += 1
                cur = (cur-1)//2
            return t
        def getssignment(x):
            cur = x
            result = [-1 for _ in range(n)]
            height = get_height(x) - 1
            while cur > 0:
                result[height - 1] = cur % 2
                height -= 1
                cur = (cur-1)//2
            return result
        def label_fn(x):
            t = get_height(x)
            if t <= n:
                return MathTex("x_{" + str(t) + "}", color=BLACK)
            return MathTex("\\bot", color=BLACK)
        g = Graph(list(range(2**(n+1)-1)),
                  [(x, (x-1)//2) for x in range(1, 2**(n+1)-1)],
                  labels={x: label_fn(x) for x in range(2**(n+1)-1)},
                  layout = "tree",
                  root_vertex=0,
                  layout_config={"vertex_spacing": (0.9, 1.5)}).scale_to_fit_height(4).shift(UP)
        print(g.edges)
        with self.voiceover("Back to our simple sat solving algorithm: "+
                            "If we are to implement the assignment enumeration recursively,"
                            +" our search is visualized as a full binary tree."):
            self.play(Create(g))
            edge_labels = []
            edge_labels_map = dict()
            for e in g.edges.keys():
                obj = g.edges[e]
                edge_labels.append(Text(str(max(e[0], e[1]) % 2), fill_opacity=1, color=BLACK).scale(0.6).move_to(obj.get_center()))
                edge_labels.append(BackgroundRectangle(edge_labels[-1], color=WHITE, fill_opacity=1, buff=.01))
                edge_labels[-1], edge_labels[-2] = edge_labels[-2], edge_labels[-1]
                edge_labels_map[e] = edge_labels[-1], edge_labels[-2]
            self.add(VGroup(*edge_labels))
        fals_clausesnim = []
        new_clauses = []
        clausey_leaf = dict()
        for id, f_clause in enumerate(falsified_clause):
            dest = g.vertices[2**(n+1)-2-id]
            f_clause_new = f_clause.copy()
            f_clause_new.set_opacity(1)
            f_clause_new.rotate(PI/2)
            f_clause_new.next_to(dest, direction=DOWN)
            new_clauses.append(f_clause_new)
            clausey_leaf[2**(n+1) - 2 - id] = f_clause_new
            fals_clausesnim.append(ReplacementTransform(f_clause, f_clause_new))
        with self.voiceover("For each leaf of the tree we find a clause that is falsified by"
                            + " the assignment of the branch ending in this leaf."):
            self.play(FadeOut(*clauses_to_fade))
            self.play(*fals_clausesnim)
        with self.voiceover("Say, for the highlighted branch, we falsify a clause X five or X two"):
            # Select one branch and indicate it.
            leaf = 2**n + 2**n // 3
            ids = [leaf]
            assignment = []
            while ids[-1] != 0:
                assignment.append(ids[-1] & 1)
                ids.append((ids[-1] - 1) // 2)
            assignment.reverse()
            edges = list(zip(ids[:-1], ids[1:]))
            self.play(*[g.edges[e].animate.set(color=YELLOW) for e in edges])
            srectssign = SurroundingRectangle(texts[sum(2**i * j for i,j in enumerate(assignment))])
            self.play(Indicate(clausey_leaf[leaf]),
                      Create(srectssign))
        ### Shrinking the tree
        nodes_to_remove = set()
        terminating_clause = [None for _ in range(2**(n+1)-1)]
        for node in g.vertices.keys():
            assignment = getssignment(node)
            found_clause = None
            found_clause_disp = None
            for clause, clause_disp in zip(falsified_clause_exp, falsified_clause):
                bad_clause = True
                for lit in clause:
                    vid = abs(lit) - 1
                    if assignment[vid] == -1 or assignment[vid] == (0 if lit < 0 else 1):
                        bad_clause = False
                if bad_clause:
                    found_clause = clause
                    found_clause_disp = clause_disp
            if found_clause is not None:
                terminating_clause[node] = (found_clause, found_clause_disp)
                for t in [node * 2 + 1, node * 2 + 2]:
                    if t < 2**(n+1)-1:
                        nodes_to_remove.add(t)
        for node in g.vertices.keys():
            if node in nodes_to_remove:
                for t in [node * 2 + 1, node * 2 + 2]:
                    if t < 2**(n+1)-1:
                        nodes_to_remove.add(t)
        mobj_to_fade = []
        for node in g.vertices.keys():
            if node in nodes_to_remove:
                mobj_to_fade.append(g.vertices[node])
        for u,v in g.edges.keys():
            if u in nodes_to_remove or v in nodes_to_remove:
                mobj_to_fade.append(g.edges[u,v])
                mobj_to_fade.extend(edge_labels_map[u,v])
        
        with self.voiceover("""
                            But it is easy to observe that our recursive search may be optimized
                            if we cut it as soon as the partial assignment we have falsifies some
                            clause.
                            """):
            self.play(FadeOut(VGroup(*mobj_to_fade)), Uncreate(srectssign), *[g.edges[e].animate.set(color=WHITE) for e in edges], run_time=0.5)
            self.play(FadeOut(*new_clauses))

        movement_to_play = []
        covering_dots = []
        for node, t_clause in enumerate(terminating_clause):
            if t_clause is None or node in nodes_to_remove:
                continue
            n_clause = t_clause[1].copy()
            n_clause.set_opacity(1)
            n_clause.move_to(g.vertices[node].get_center() + DOWN * (n_clause.height / 2 + g.vertices[node].height))
            covering_dots.append(Dot(radius=g.vertices[node].width/2).move_to(g.vertices[node].get_center()))
            movement_to_play.append(ReplacementTransform(t_clause[1].copy(), n_clause))
        with self.voiceover("""Such decision tree may serve as a more succinct certificate of "
                            unsatisfiability of the formula."""):
            self.add(*covering_dots)
            self.play(*movement_to_play)
        with self.voiceover("Notice that it is not necessary to query the variables in the same order"
                            +" in a decision tree."):
            nodes = [nd for id, nd in g.vertices.items() if id not in nodes_to_remove]
            self.play(*[Indicate(nd) for nd in nodes])
        
        tree = BinaryTree(11, tp="bamboo")
        print(tree.parent)
        tree.label = [[5,4,3,2,1][tree.height[i]] if i not in tree.leaves else 0 for i in range(tree.n)]    
        g2 = Graph(range(tree.n), [(i, tree.parent[i]) for i in range(1, tree.n)],
                   labels={i: MathTex(("x_{" + str(tree.label[i]) + "}") if i not in tree.leaves else "\\bot", color=BLACK) for i in range(tree.n)},
                   layout=tree.layout).scale_to_fit_height(4).center()
        with self.voiceover("In this case we could have queried the variables in a different order"
                            +" to get a much smaller tree"):
            back_g2 = BackgroundRectangle(g2)
            back_g2.set_opacity(0.95)
            self.add(back_g2)
            self.play(Create(g2))
        

class HornCNFAnimation(Scene):
    def construct(self):
        formula = Formula([[1, -2], [-1, 3], [-3, 4, -5], [-5], [-4, 5], [2, 5], [-3]])
        group = VGroup(*formula.math_tex).arrange(buff=0.3,direction=DOWN).move_to(UL)
        self.play(Write(group))
        self.wait()
        active_vars = set(range(1, 6))
        alive_clauses = [True for _ in range(len(formula.clauses))]
        assignment = [-1 for _ in range(6)]
        current_node = Dot().next_to(group, direction=RIGHT*9+UP*2.5)
        while len(active_vars) > 0:
            current_lit = 0
            clause_id = -1
            id_in_clause = -1
            for id, clause in enumerate(formula.clauses):
                if not alive_clauses[id]:
                    continue
                if (sum(1 if abs(lit) in active_vars else 0 for lit in clause) == 1):
                    current_lit = sum(lit if abs(lit) in active_vars else 0 for lit in clause)
                    id_in_clause = sum(lid if abs(lit) in active_vars else 0 for lid,lit in enumerate(clause))
                    clause_id = id
                    break
            if current_lit == 0:
                self.play(Write(Text("Satisfiable!")))
                break
            right_node = Dot().next_to(current_node, direction=RIGHT+DOWN*2)
            next_node = Dot().next_to(current_node, direction=DOWN*2+LEFT)
            dir = Line(current_node.get_center(), right_node.get_center())
            dir2 = Line(current_node.get_center(), next_node.get_center())            

            
            rectangle = SurroundingRectangle(formula.math_tex[clause_id][formula.literal_indices[clause_id][id_in_clause]])
            self.play(Create(current_node), Write(MathTex("x_{" + str(abs(current_lit))+"}").next_to(current_node, direction=LEFT)),
                      Create(rectangle), Create(dir), Create(dir2), Create(right_node))
            current_node = next_node
            for id, clause in list(filter(lambda a: a[0] != clause_id, 
                                          enumerate(formula.clauses))) + [(clause_id, formula.clauses[clause_id])]:
                candidates = []
                if not alive_clauses[id]:
                    continue
                color = RED
                if current_lit in clause:
                    candidates = list(range(len(formula.tex[id])))
                    color = GREEN
                    alive_clauses[id] = False
                elif -current_lit in clause:
                    for lit_id, lit in enumerate(clause):
                        if lit == -current_lit:
                            candidates = [formula.literal_indices[id][lit_id],
                                          formula.literal_indices[id][lit_id]+1,
                                          formula.literal_indices[id][lit_id]-1]
                candidates = list(filter(lambda x: x >= 0 and x < len(formula.tex[id]) and formula.active[id][x], candidates))
                if len(candidates) > 0:
                    to_show = []
                    for i in candidates:
                        final = formula.math_tex[id][i].copy()
                        final.stroke_color = color
                        final.color = color
                        to_show.append(Transform(formula.math_tex[id][i], final))
                    self.play(*to_show, run_time=0.7)
                for i in candidates:
                    formula.active[id][i] = False

            self.remove(rectangle)
            active_vars.remove(abs(current_lit))
            assignment[abs(current_lit)] = 1 if current_lit > 0 else 0  


