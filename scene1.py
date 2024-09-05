from manim import *
import copy
import networkx as nx
from itertools import product, combinations, permutations
from lib import Formula, BinaryTree
import random
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import math

from manim_slides import Slide

class TitleSlide(Slide):
    def construct(self):
        caption = Text("Hardness Condensation by Restriction").shift(UP)
        authors_l = [Tex("Mika G\\\"o\\\"os, ~~~\\texttt{EPFL}"), 
                     Tex("Artur Riazanov, ~~~\\texttt{EPFL}"), 
                     Tex("Ilan Newman, ~~~\\texttt{University of Haifa}"),
                     Tex("Dmitry Sokolov, ~~~\\texttt{EPFL}")]
        authors = VGroup(*authors_l).arrange(direction=DOWN).next_to(caption, direction=DOWN).shift(DOWN * 0.5)
        self.play(Write(caption), Create(authors))
        self.next_slide()


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
                

class ColoringSlide(Slide):
    def construct(self):

        caption = Tex("Chromatic number $\\chi(G)$")
        random.seed(12345432)
        n, edges = generate_girth(40, 5)
        print(edges)
        node_color = color_graph_greedily(n, edges)
        pallette = [RED_C, BLUE_C, GREEN_C, YELLOW_C]
        graph = Graph(list(range(1, n + 1)), edges, layout="kamada_kawai",
                        vertex_config={x: {"fill_color": pallette[node_color[x]]} for x in range(1, n + 1)},
                        layout_scale=2)
        # 
        #               The chromatic number of a graph is the smallest number of colors
        #               one needs to color the nodes of the graph such that no edge connects
        #               two nodes of the same color.
        #               """):
        caption.move_to(UP * 3.5)
        graph.next_to(caption, direction=DOWN)
        self.play(Write(caption), Create(graph))
    # """
    #                     If the graph has chromatic number k, can we find a size big-O of k 
    #                     subgraph with the same chromatic number?
    #                     """):
        self.next_slide()
        question = Tex("Is there a subgraph $H$ of $G$ with $O(\\chi(G))$ nodes and $\\chi(H) = \\chi(G)$?")\
            .scale(0.7).next_to(graph, direction=DOWN)
        question.add_background_rectangle(BLACK, opacity=0.9)
        self.play(Write(question))
    # """
    #                     The answer is no, since graphs can have arbitrary large girth and chromatic number.
    #                     For example the graph on the screen has chromatic number 3 and girth 5, so
    #                     no graph of size 4 has cycles and thus have chromatic number 2.
    #                     """):
        self.next_slide()
        for subgraph in [[1,2,3,4], [5,6,7,8], [18,19,20,21]]: 
            group = VGroup(*[graph[k] for k in subgraph], *[graph.edges[i,j] for i,j in edges if i in subgraph and j in subgraph])
            self.play(Indicate(group))
            self.next_slide()
        answer = Tex("[Erd\\\"os, 1959] \\textbf{No}, can have arbitrary large girth and chromatic number.")\
            .scale(0.7).next_to(question, direction=DOWN)
        answer.add_background_rectangle(BLACK, opacity=0.9)
        self.play(Write(answer))
        self.next_slide()
        
        

            
            

class VertexCoverSlide(Slide):
    def construct(self):
        random.seed(123323)
        caption = Text("Vertex Cover").move_to(UP * 3.5)
        nodes = range(1,41)
        edges = [(random.randint(1,7), random.randint(1,40)) for _ in range(80)]
        edges = [(x, y) for x,y in edges if x != y]
        graph = Graph(nodes, edges, layout="kamada_kawai").next_to(caption, direction=DOWN)
        for e in edges:
            graph.edges[e].fade(0.8)
            
        # with self.voiceover(""" 
        #               A vertex cover is a set of nodes such that every edge in the graph is
        #               incident to one of them.
        #               """):
        self.play(Write(caption))
        self.play(Create(graph))
        self.next_slide()
        # with self.voiceover("""
        #               For example here the nodes in the center form a vertex cover of the graph.
        #                     """):
        for v in range(1,8):
            graph[v].set_color(RED_C)
        animations = [graph[v].animate.move_to(RIGHT / 2 * math.cos(2*math.pi / 7 * v)
                                                            + UP / 2 * math.sin(2*math.pi / 7 * v)) for v in range(1,8)]
        animations.extend([graph[v].animate.move_to(RIGHT * 2 * math.cos(2*math.pi / 33 * (v - 8))
                                                            + UP * 2 * math.sin(2*math.pi / 33 * (v - 8))) for v in range(8, 41)])
            
        self.play(*animations) 
        self.next_slide()
        question = Tex("Is there a subgraph of size $f(k)$ that has a vertex cover $k$?").next_to(graph, direction=DOWN)
        # with self.voiceover("""
        #                     Can we condense (or locally witness) vervex cover? For example,
        #                     is there a subgraph of size depending only on k and the same vertex cover?
        #                     """):
        self.play(Write(question))
        self.next_slide()
        # with self.voiceover("""
        #                     The answer is yes. This is studied in the field of fixed-parameter tractability.
        #                     """):
        answer = Tex("\\textbf{Yes}: FPT kernelization.").next_to(question, direction=DOWN)
        self.play(Write(answer))
        self.next_slide()
        # with self.voiceover("""
        #                     An algorithmically efficient way to find a smaller graph that has vertex cover k'
        #                     if and only if the original graph had vertex cover k is called kernelization.
        #                     """):
        self.play(graph.animate.shift(LEFT * 3))
        lbrace = Tex("$\Big($").next_to(graph, direction=LEFT)
        rbrace = Tex("$,k\Big)$").next_to(graph, direction=RIGHT)
        arrow = Arrow(start=rbrace.get_right(), end=RIGHT)
        lbrace2 = Tex("$\Big($").next_to(arrow, direction=RIGHT)
        small_graph = Graph(list(range(7)), list(combinations(range(7), 2)), layout="circular")\
            .scale(0.3).next_to(lbrace2, direction=RIGHT)
        rbrace2 = Tex("$,k'\Big)$").next_to(small_graph, direction=RIGHT)
        self.play(FadeIn(arrow), FadeIn(lbrace, rbrace, lbrace2, rbrace2), Create(small_graph))
        self.next_slide()


class QueryComplexityIntroSlide(Slide):
    def construct(self):
        random.seed(123323)
        caption = Text("Query Complexity").move_to(UP * 3.5) 
        # with self.voiceover("""
        #                     The first computation complexity measure we will try to condense
        #                     is query complexity: the number of bit probes
        #                     one has to make in order to compute the value of the function.
        #                     Equivalently, the smallest depth of a decision tree computing 
        #                     the function.
        #                     """):
        q_def = Tex("""
                        \\textbf{Definition}. Query complexity of a function """, """$f\colon \\{0,1\\}^n \\to \mathcal{O}$ """,
                                            """is the \emph{depth} of decision tree computing $f$.
                        """).scale_to_fit_width(11)
        self.play(Write(q_def), FadeIn(caption))
        self.next_slide()
        # with self.voiceover("""Let us illustrate this notion with a simple example."""):
        self.play(q_def.animate.shift(UP * 2))
        self.next_slide()
        tree_or = BinaryTree(13, tp="bamboo")
        for i in range(tree_or.n):
            tree_or.label[i] = tree_or.height[i] + 1
        tree_or.leaf_labels = {i: int(any(j == 1 for j in tree_or.get_assignment(i))) for i in tree_or.leaves}
        g_or = Graph(range(tree_or.n), tree_or.get_edges(),
                  layout=tree_or.layout,
                  labels=tree_or.get_tex_labels()).scale_to_fit_width(8).center().shift(DOWN * 0.5)
        labels_or = [t for t in tree_or.labels_for_graph(g_or)]
        # with self.voiceover("This decision tree computes the function OR."):
        
        or_formula = []
        for i in range(tree_or.n):
            if i not in tree_or.leaves:
                if len(or_formula) > 0:
                    or_formula.append("\\lor")
                or_formula.append("x_{" + str(tree_or.label[i]) + "}")
        or_formula_disp = MathTex(*or_formula).next_to(g_or, direction=DOWN)
        # with self.voiceover("""The disjunction of all the variables."""):
        self.play(Write(or_formula_disp))
        self.next_slide()
        self.play(Create(g_or), *[Create(t) for t in labels_or])
        self.next_slide()
        
        statement_text = Tex(""" \\textbf{Fact}. Query complexity of ${\\rm OR}_n(x) := \\bigvee_{i=1}^n x_i$ 
                                                 is \emph{exactly} $n$.
                             """).scale(0.7).shift(UP)
        # with self.voiceover("""This is not very hard to show, that this depth is required, 
                            #  no matter what tree we use."""):
        self.play(g_or.animate.set_opacity(0.1), *[FadeOut(t) for t in labels_or],
                       FadeIn(statement_text))
        self.next_slide()
        # with self.voiceover("""
                            # Whatever bits we query, if we only get zeroes, the value
                            # of the function is not fixed until the very end, if it
                            # is zero, then the OR is zero, and if it is one, the OR is one.
                            # """):
        cur_text = ["*" for _ in range(6)]
        order = [5,2,3,0,1]
        last = list(set(range(6)).difference(set(order)))[0]
        cur_text_disp = Tex(*cur_text).scale_to_fit_width(5)
        self.play(FadeIn(cur_text_disp))
        for u in order:
            rect = SurroundingRectangle(cur_text_disp[u])
            self.play(Create(rect))
            cur_text[u] = "0"
            new_text_disp = Tex(*cur_text).scale_to_fit_width(5)
            self.play(Uncreate(rect), ReplacementTransform(cur_text_disp, new_text_disp))
            cur_text_disp = new_text_disp
            self.next_slide()
        
        for val in ["0", "1"]:
            cur_text[last] = val
            new_text_disp = Tex(*cur_text).scale_to_fit_width(5)
            self.play(ReplacementTransform(cur_text_disp, new_text_disp))
            cur_text_disp = new_text_disp
            self.next_slide()
 
    
class QueryComplexityCondensationSlide(Slide):
    def construct(self):
        random.seed(123325)
        caption = Text("Condensing Query Complexity").move_to(UP * 3) 
        # with self.voiceover("""
        #     Suppose a function f has query complexity k then can one fix all-but big-O of k
        #     input bits such that query complexity of the restricted function is still k?
        #                """):
        condensation_def = Tex("\\textbf{Question}: "
                                "$f\colon \\{0,1\\}^n \\to \\{0,1\\}$ has query complexity $k$."
                                " Find $\\rho \\subseteq \\{0,1, *\\}^n$ with $O(k)$ stars "
                                "such that $f|_{\\rho}$ retains query complexity $k$.")\
                                .scale_to_fit_width(11).next_to(caption, direction=DOWN)
        self.play(FadeIn(caption), Write(condensation_def))
        self.next_slide()
    # with self.voiceover("""
    #                     If query complexity is maximal, then the function is already condensed,
    #                     so let us look at some example of intermediate query complexity.
    #                     """):
    #     pass
    # with self.voiceover("""
    #                     One such function is called sink, the input encodes directions of edges
    #                     in a complete directed graph (known as tournament graph). 
    #                     """):
        sink_def = Tex("\\textbf{Example}: $\\textsc{Sink}\\colon \\{0,1\\}^{\\binom{n}{2}} \\to \\{0,1\\}$ is defined"
                        " such that $\\textsc{Sink}(G) = 1$ iff $G$ has a sink.")\
                        .scale_to_fit_width(11).next_to(condensation_def, direction=DOWN)
        edges = [(i, j) if (random.randint(1,2) == 1 or j == 3) and (not i == 3) else (j, i) 
                                    for i,j in combinations(range(7), 2)]
        not3 = [i for i in range(7) if i != 3]
        for i, j in zip(not3, not3[1:] + [not3[0]]):
            for t, e in enumerate(edges):
                if e == (j,i):
                    edges[t] = (i,j)

        sink_graph = DiGraph(list(range(7)), 
                                edges,
                                layout="circular", layout_scale=1.5, 
                                edge_config={"tip_config": {"tip_length": 0.35, "tip_width": 0.15}})\
                                .next_to(sink_def, direction=DOWN)
        self.play(Write(sink_def), Create(sink_graph))
        self.next_slide()
    
    # with self.voiceover("""
    #                     The function should return 1 if and only if the graph has a sink, i.e. 
    #                     there is a node with all edges directed towards it.
    #                     """):
        animations = []
        for i in range(7):
            if i == 3:
                continue
            edge_cp = sink_graph.edges[i,3].copy()
            edge_cp.stroke_color = RED_C
            edge_cp.color = RED_C
            animations.append(FadeOut(sink_graph.edges[(i,3)]))
            animations.append(FadeIn(edge_cp))
            sink_graph.edges[(i,3)] = edge_cp
        self.play(*animations)
        lbrace = MathTex("\\textsc{Sink}\\Big(").next_to(sink_graph, direction=LEFT)
        rbrace = MathTex("\\Big) = 1").next_to(sink_graph, direction=RIGHT)
        self.play(FadeIn(lbrace, rbrace))
        self.next_slide()
    # with self.voiceover("""
    #                     Query complexity of this function is n, which is about the square root
    #                     of the number of input bits. 
    #                     """):
        sink_qc = Tex("Query complexity of $\\textsc{Sink}$ is $\Theta(n)$.")\
            .next_to(sink_graph, direction=DOWN)
        
        self.play(Write(sink_qc))
        self.next_slide()
    # with self.voiceover("""
    #                     Then if we restrict all edges that are not incident to one of the nodes
    #                     such that there is a cycle among them, then the restricted function is 
    #                     1 if and only if all the edges are directed towards the fixed node.
    #                     """):
        ham_path = None
        for pp in permutations([i for i in range(7) if i != 3]):
            p = list(pp)
            if all(e in edges for e in zip([p[-1]] + p[:-1], p)):
                    ham_path = list(zip([p[-1]] + p[:-1], p))
                    break
        animations = []
        to_remove = []
        to_remove_later = []
        for e in ham_path:
            edge_cp = sink_graph.edges[e].copy()
            edge_cp.stroke_color = GREEN_C
            edge_cp.color = GREEN_C
            edge_cp.stroke_width *= 2
            to_remove.append(sink_graph.edges[e])
            to_remove_later.append(edge_cp)
            animations.append(ReplacementTransform(sink_graph.edges[e], edge_cp))
 
        for i in range(7):
            if i == 3:
                continue
            edge_cp = Line(sink_graph[i].get_center(), sink_graph[3].get_center(), stroke_color=YELLOW_C).set_opacity(0.9)
            edge_cp.z_index = -100
            to_remove.append(sink_graph.edges[i,3])
            to_remove_later.append(edge_cp)
            animations.append(ReplacementTransform(sink_graph.edges[(i,3)],edge_cp))
        one = Text("1")
        self.play(*animations, FadeOut(lbrace, rbrace), FadeIn(one.scale(0.7).next_to(sink_graph[3], direction=UP)))
        self.remove(*to_remove)
        self.next_slide()
    # with self.voiceover("""
    #                     The query complexity of the restricted function is exactly n minus one
    #                     since it is equivalent to the OR function up to flipping the signs of some bits.
    #                     """):
        answer = Tex("Can condense \\textsc{Sink} by restricting all edges non-incident to $1$.").scale_to_fit_width(10).next_to(sink_qc, direction=DOWN)
        self.play(Write(answer))
        self.next_slide()
        
        self.remove(*to_remove_later, *[e for e in sink_graph.edges], *[v for v in sink_graph], one)
        self.play(FadeOut(sink_def), FadeOut(sink_qc), answer.animate.next_to(condensation_def, direction=DOWN, buff=0.5))
        self.next_slide()
        sensitivity_def = Tex("""\\textbf{Definition.} Sensitivity ${\\rm s}(f)$ is the degree of graph $(\{0,1\}^n, \{(x, x + e_i) \mid f(x) \\neq f(x + e_i)\})$.
                              """).next_to(answer, direction=DOWN, buff=0.7).scale_to_fit_width(9)
        self.play(FadeIn(sensitivity_def))
        self.next_slide()
        condensing_s = Tex("$f$ can be condensed to exactly ${\\rm s}(f)$ variables.").next_to(sensitivity_def, direction=DOWN, buff=0.7)
        self.play(FadeIn(condensing_s))
        self.next_slide()
        condensing_d = Tex("$f$ can be condensed to exactly ${\\rm deg}(f)$ variables.").next_to(condensing_s, direction=DOWN, buff=0.7)
        self.play(FadeIn(condensing_d))
        self.next_slide()
        
        answer = Tex("\\textbf{Theorem}: \\\\"
                        "There exists $f\\colon \\{0,1\\}^n \\to \\{0,1\\}$ with query complexty $k$ "
                        "such that for all $\\rho$ with $|\\rho^{-1}(*)| = O(k)$ "
                        "the query complexity of $f|_{\\rho}$ is $\\tilde{O}(k^{2/3})$.")\
                            .center().scale_to_fit_width(12).add_background_rectangle(color=DARK_BLUE, opacity=1)
        self.play(Write(answer))
        self.next_slide()
        cheatsheets = Text("Cheatsheets", color=BLACK)
        cheatsheets.move_to(answer.get_top() + RIGHT * (answer.width / 2 - cheatsheets.width/2) + DOWN * cheatsheets.height / 2)\
            .add_background_rectangle(color=YELLOW_C, opacity=1)
        self.play(Write(cheatsheets))
        self.next_slide()
      
class CondensingWithParity(Slide):
    def construct(self):
        random.seed(123323)
        caption = Text("Condensing with Expander").move_to(UP * 3.5)  
        self.play(FadeIn(caption))
        self.next_slide()
        nodes_l = ["y_" + str(i) for i in range(1,10)]
        nodes_r = ["x_" + str(i) for i in range(1,6)]
        expander = Graph(nodes_l + nodes_r, [(nodes_l[random.randrange(0, len(nodes_l))], nodes_r[random.randrange(0, len(nodes_r))]) for _ in range(30)],
                         layout={u: LEFT * 4 + UP * 3.5 + DOWN * (int(u[2])-int('0')) * 0.8 + LEFT * (2 if u[0] == 'x' else 0) for u in nodes_l + nodes_r},
                         labels={u: MathTex(u if u[0] == 'x' else "\\oplus", color=BLACK) for u in nodes_l + nodes_r})
        self.play(Create(expander))
        self.next_slide()
        self.play(expander.animate.add_vertices("f", positions={"f": LEFT * 2}, labels={"f": MathTex("F", color=BLACK)}))
        self.play(expander.animate.add_edges(*[(node, "f") for node in nodes_l]))
        self.next_slide()
        theorem = Tex("""{\\bf Theorem.} \\texttt{[FPR 2022]} \\\\
                       If $\Pi$ is resolution refutation of \\\\
                       $F \circ \\oplus_G$ with small
                       enough width we have \\\\
                       ${\\rm width}(\\Pi) {\\rm depth}(\\Pi)$""", """$\,\ge\, {\\rm width}(F) / 2.$""")\
                       .scale_to_fit_width(8).next_to(expander, direction=RIGHT).shift(UP*2)
        self.play(Write(theorem))
        self.next_slide()
        self.play(Indicate(theorem[1]))
        self.next_slide()
        theorem2 = Tex("""{\\bf Theorem.} \\texttt{[this work]} \\\\
                       If $f$ has query complexity $k$, \\\\
                       then $f \circ \\oplus_G$ has query complexity\\\\
                        $\Omega(\\Delta(G) \cdot k)$.""")\
                       .scale_to_fit_width(8).next_to(theorem, direction=DOWN, buff=0.7)
        self.play(Write(theorem2))
        self.next_slide()
    
class NewmanTheorem(Slide):
    def construct(self):
        random.seed(123323)
        caption = Text("Domain Size is a Liability").move_to(UP * 3.5) 
        self.play(FadeIn(caption))
        self.next_slide()
        newman = Tex("""{\\bf Theorem.} \\texttt{[Newman]} Let $f\\colon (\\{0,1\\}^n)^2 \\to \\{0,1\\}$. 
                        Then $R_{1/3}^{\\rm private}(f) \\le R^{\\rm public}_{1/6}(f)$""", """$+\, \\log n + O(1).$
                     """).next_to(caption, direction=DOWN)
        self.play(Write(newman))
        self.next_slide()
        texts = [Tex("$t_" + str(i) + "$ :=", " \\texttt{" + ("".join(str(random.randint(0,1))
                                                                       for _ in range(10)))*2 + "}") for i in range(1,5)]
        for _, u in texts:
            u.set_color(GREEN_C)
        g_texts = VGroup(*texts).arrange(DOWN)
        self.play(Write(g_texts))
        self.next_slide()
        """
        First we globally pick at random k sufficiently long bitstrings t_1, ... t_k
        Then Alice picks an integer i from [k] uniformly at random, sends it to Bob and they
        run the public-coin protocol using the bits t_i as public random bits. 
        """
        
        alice = Text("Alice").move_to(DOWN + LEFT * 4)
        bob = Text("Bob").move_to(DOWN + RIGHT * 4)
        i = 2
        self.play(*[FadeOut(t) for j, t in enumerate(texts) if j != i],
                   FadeIn(alice), FadeIn(bob), texts[i].animate.next_to(newman, direction=DOWN, buff=1))
        self.next_slide()
        arrow = Arrow(start=alice.get_right(), end=bob.get_left())
        text = Text("3").next_to(arrow, direction=UP)
        self.play(Create(arrow), FadeIn(text))
        self.next_slide()
        rect = Rectangle(width=8, height=2.3).next_to(arrow, direction=DOWN)
        rect_capt = Text("public-coin protocol", color=BLUE_C).next_to(rect.get_top(), direction=DOWN)
        self.play(FadeIn(rect), FadeIn(rect_capt), texts[i][1].animate.next_to(rect_capt, direction=DOWN))
        self.next_slide()
        self.play(FadeOut(alice, bob, rect, rect_capt, arrow, text, texts[i]))
        self.next_slide()
        newman_new = Tex("""{\\bf Theorem.} \\texttt{[Newman]} Let $f\\colon (\\{0,1\\}^n)^2 \\to \\{0,1\\}$. 
                        Then $R_{1/3}^{\\rm private}(f) \\le R^{\\rm public}_{1/6}(f)$""", """$+\, \\log k.$
                     """).next_to(caption, direction=DOWN)
        self.play(Transform(newman[1], newman_new[1]))
        self.next_slide()
        prob = MathTex("\\Pr_{\\vec{\\bf t}}\\left[ \\frac{1}{k} \sum_{i \in [k]} [\Pi^{{\\bf t}_i}(x,y) = f(x,y)] \le 1/3\\right] \le \\exp(-\\Omega(k))")
        prob.next_to(newman_new.get_bottom(), direction=DOWN, buff=0.7)
        self.play(Write(prob))
        self.next_slide()
        union_bound = Tex("Union bound over all $x,y \in \\{0,1\\}^n$:\\\\ success with probability $2^{2n} \\exp(-\Omega(k))$.")\
            .next_to(prob, direction=DOWN, buff=0.7)
        self.play(Write(union_bound))
        self.next_slide()
        gromulz = Tex("{\\bf Theorem} \\texttt{[Gromulz 1997]} For $f\colon \\{0,1\\}^n\\to \\{0,1\\}$ we have $\|\hat{f}\|_{0, 1/3} \le O(\|\hat{f}\|^2_1 \cdot n).$ ")
        self.play(FadeOut(prob, union_bound), FadeIn(gromulz), FadeOut(newman[1]), FadeIn(MathTex("+\, \\log n + O(1)").move_to(newman_new[1].get_center())))
        self.next_slide()
        gromulz = Tex("{\\bf Theorem} \\texttt{[Lee, Shraibman 2015]} For $M\colon [N] \\times [N] \\to \{0,1\}$ we",
                      " have  ${\\rm rk}_{1/3}(M) \le O(\gamma_2^2(M) \cdot \\log N).$ ").next_to(gromulz, direction=DOWN, buff=0.7).scale_to_fit_width(12)
        self.play(FadeIn(gromulz))
        self.next_slide()
        
class CommunicationComplexityIntro(Slide):
    def construct(self):
        random.seed(123323)
        caption = Text("Deterministic Communication").move_to(UP * 3.5) 
        """
        The next measure we study is comminication complexity. Let me informally define it.                    
        """
        self.play(Write(caption))
        self.next_slide()
        """
        There are two players Alice and Bob who would like two compute a 2-argument function F,
        but each of players has only one of the arguments. The players exchange messages until 
        they both learn the value of the function.
        """
        fdef = MathTex("F\\colon [N] \\times [N] \\to \\{0,1\\}").next_to(caption, direction=DOWN)
        alice = Text("Alice").move_to(UP + LEFT * 4)
        bob = Text("Bob").move_to(UP + RIGHT * 4)
        self.play(Write(fdef), Write(alice), Write(bob))
        self.next_slide()
        
        path1 = CubicBezier(alice.get_bottom(), DOWN, LEFT, bob.get_left())
        path2 = CubicBezier(bob.get_bottom(), DOWN, RIGHT, alice.get_right())
        
        for curp, mlen in [(path1, 10), (path2, 15), (path1, 10)]:
            anims = []
            letters = []
            for d in range(mlen):
                t = Tex("\\texttt{" + str(random.randint(0,1)) + "}").set_color(GREEN_C)
                letters.append(t)
                anim = MoveAlongPath(t, curp)
                anims.append(anim)
            g = AnimationGroup(anims, run_time=1, lag_ratio=0.1)
            self.play(g)
            self.play(*[FadeOut(t, run_time=0.1) for t in letters])
            self.next_slide()
            
        """
        We think of the function as a matrix, Alice knows the number of the row and Bob knows the number
        of the column. 
        """
        N = 15
        rectangles = [[(0, N, 0, N)]]
        for i in range(6):
            new_layer = []
            for rect in rectangles[-1]:
                print(rect)
                if (rect[0] == rect[1] - 1 and i % 2 == 0) or (rect[2] == rect[3] - 1 and i % 2 == 1):
                    new_layer.append(rect)
                    continue
                thr = random.randrange(rect[0] + 1, rect[1]) if i % 2 == 0 else random.randrange(rect[2] + 1, rect[3])
                rect_a = (rect[0], thr, rect[2], rect[3]) if i % 2 == 0 else (rect[0], rect[1], rect[2], thr)
                rect_b = (thr, rect[1], rect[2], rect[3]) if i % 2 == 0 else (rect[0], rect[1], thr, rect[3])
                new_layer.extend([rect_a, rect_b])
            rectangles.append(new_layer)
        fval = [_ % 2 for _ in range(len(rectangles[-1]))]
        def F(i, j, r, fval):
            for id, rect in enumerate(r):
                if rect[0] <= i and i < rect[1] and rect[2] <= j and j < rect[3]:
                    return fval[id]
            return -1
        digit_matrix = [[Tex("\\tiny\\texttt{" + str(F(i, j, rectangles[-1], fval)) + "}")
                         .move_to(UP / 5 * (i - 5) + RIGHT / 5 * (j - 5) + DOWN) for j in range(N)] 
                         for i in range(N)]
        anims = []
        for i, j in product(range(N), repeat=2):
            anims.append(FadeIn(digit_matrix[i][j]))
        # alice_to_row = CubicBezier(alice.get_bottom(), DOWN, RIGHT, digit_matrix[N//2][0].get_left())
        # bob_to_column = CubicBezier(bob.get_left(), RIGHT, UP, digit_matrix[N-1][N//2].get_top())
        self.play(*anims)
        self.next_slide()
        """
        By communicating the first bit Alice partitions the rows into two subsets: the values for which she
        sends zero and the values for which she sends one, the bit sent by Bob further partitions.
        This process terminates when the matrix is partitioned into monochromatic rectangles.
        """
        #self.play(Uncreate(alice_to_row), Uncreate(bob_to_column))
        palette = [DARK_BLUE, GREEN_C, RED_E, BLACK, DARK_BROWN, DARK_GRAY, ORANGE]
        oanimb = []
        for i in range(1, 7):
            anima = []
            animb = []
            for id, r in enumerate(rectangles[i]):
                print("r=", r)
                corners = [(r[0], r[2]), (r[0], r[3] - 1), (r[1] - 1, r[3] - 1), (r[1] - 1, r[2])]
                directions = [DOWN + LEFT, DOWN + RIGHT, UP + RIGHT, UP + LEFT]
                clr = palette[id % len(palette)]
                gr = Polygon(*[digit_matrix[i][j].get_corner(d) for (i, j), d in zip(corners, directions)],
                             color=clr).set_stroke(opacity=0).set_fill(color=clr, opacity=0.95).scale(1.02).set_z_index(-100 + i)
                anima.append(FadeIn(gr).set_run_time(0.5))
                if i < 6: 
                    animb.append(gr.animate.fade(0.8))
            self.play(*anima, *oanimb)
            oanimb = copy.copy(animb)
            self.next_slide()
        """
        The number of rectangles is at most two to the number of bits communicated in the worst case. 
        Since each monochromatic rectangle has rank 1, we have that the deterministic communication 
        cost is at most the log of rank of the matrix.
        """
        M = MathTex("M=").next_to(digit_matrix[N//2][0], direction=LEFT)
        logrank = MathTex("{\\rm D}(M) \\ge \\log \\#\\text{rectangles} \\ge \\log{\\rm rk}(M)")\
                  .scale_to_fit_width(7).next_to(digit_matrix[0][N//2], direction=DOWN)
        self.play(Write(M), Write(logrank))
        self.next_slide()
        logrankconj = Tex("{\\bf Log-Rank Conjecture}: $\\exists C$ such that ${\\rm D}(M) \\le (\\log {\\rm rk} (M))^C$.")\
            .next_to(logrank, direction=DOWN).scale_to_fit_width(10)
        self.play(Write(logrankconj))
        self.next_slide()
        """
        We say that a communication complexity k matrix can be losslessly condensed if there is a 
        2 to the k by 2 to the k submatrix which retains communication complexity k.
        """
        condense_tex = Tex("{\\bf Definition}: $M$ with ${\\rm D}(M) = k$"
                           " can be losslessly condensed if $\\exists$ $2^{O(k)} \\times 2^{O(k)}$ submatrix $M'$"
                           " with ${\\rm D}(M') = \\Omega(k)$.")\
                            .scale_to_fit_width(12).next_to(digit_matrix[N-1][7], direction=UP)\
                            .add_background_rectangle(DARK_BROWN, opacity=1)
        rect = Rectangle(WHITE, height=1.5, width=1.5).center().shift(DOWN*0.3).set_fill(color=WHITE, opacity=0.6)
        mprime_label = MathTex("M'").move_to(rect.get_center()).set_color(BLACK)
        self.play(Write(condense_tex), Create(rect), Write(mprime_label))
        self.next_slide()
        """
        Notice that for rank this is true: for a rank-k matrix M there always exist a k-by-k submatrix M'
        with full rank. Hence, log rank conjecture implies that deterministic communication complexity is 
        condensed up to some polynomial.
        Indeed Hrubes independently confirms that communication complexity can be weakly condensed. 
        """
        hrubes_result = Tex("{\\bf Theorem} \\texttt{[Hrubes, 2024]} Can find $2^{\\sqrt{k}} \\times 2^{\\sqrt{k}}$ submatrix $M'$"
                            " with ${\\rm D}(M') = \\Omega(\\sqrt{k})$")\
                                .scale_to_fit_width(12).next_to(condense_tex,direction=DOWN)\
                                .add_background_rectangle(DARK_BLUE, opacity=1)
        self.play(Write(hrubes_result))
        self.next_slide()
        open_problem = Tex("{\\bf Open}: Is ${\\rm D}$ losslessly condensable?").scale_to_fit_width(11)\
                        .next_to(hrubes_result, direction=DOWN).add_background_rectangle(DARK_GRAY, opacity=1)
        self.play(Write(open_problem))
        self.next_slide()

class RandomizedCommunication(Slide):
    def construct(self):
        random.seed(123323)
        caption = Text("Randomized Communication").move_to(UP * 3.5) 
        """
        What if Alice and Bob are allowed to use randomness in their computations?                    
        """
        self.play(Write(caption))
        self.next_slide()
        """
        The goal is still to compute a function of two arguments
        """
        fdef = MathTex("F\\colon [N] \\times [N] \\to \\{0,1\\}").next_to(caption, direction=DOWN)
        alice = Text("Alice").move_to(UP + LEFT * 4)
        bob = Text("Bob").move_to(UP + RIGHT * 4)
        text = Tex("\\texttt{" + ("".join(str(random.randint(0,1)) for _ in range(500)))*3 + "}")\
            .next_to(fdef, direction=DOWN)
        text.shift(text.width / 3 * LEFT).set_color(GREEN_C)
        print(text.get_center())
        print(text.width)
        self.play(Write(fdef), Write(alice), Write(bob))
        self.play(FadeIn(text))
        self.next_slide(loop=True)
        self.play(text.animate.shift(RIGHT*text.width/6).set_rate_func(rate_functions.linear), duration=3)
        self.next_slide()
        N = 15
        digit_matrix = [[Tex("\\tiny\\texttt{" + str(int(i <= j)) + "}")
                         .move_to(UP / 5 * (i - 5) + RIGHT / 5 * (j - 5) + DOWN) for j in range(N)] 
                         for i in range(N)]
        gt_tex = MathTex("\\textsc{Gt}_N = ").next_to(digit_matrix[7][0], direction=LEFT)
        self.play(FadeIn(*[digit_matrix[i][j] for i,j in product(range(N), repeat=2)]), Write(gt_tex))
        self.next_slide()
        known = Tex("{\\bf Theorem:} \\texttt{[Nisan, 1993]} ${\\rm R}(\\textsc{Gt}_N) = O(\\log \\log N)$;~~~~"
                    "${\\rm D}(\\textsc{Gt}_N) = \\Theta(\\log N)$").scale(0.7).next_to(digit_matrix[0][7], direction=DOWN)
        self.play(Write(known))
        self.next_slide()
        rect = Rectangle(WHITE, height=1.5, width=1.5).center().shift(DOWN*0.3).set_fill(color=WHITE, opacity=0.8)
        mprime_label = MathTex("M'").move_to(rect.get_center()).set_color(BLACK)
        self.play(Create(rect), Write(mprime_label))
        self.next_slide()
        mprime_size = Tex("$|M'| = 2^{{\\rm R}({\\textsc{Gt}_N})} = O(\\log N)$")\
            .scale(.6).next_to(digit_matrix[N//2][N-1], direction=RIGHT)
        rcc_mprime = Tex("${\\rm R}(M') = O(\\log \\log |M'|) = O(\\log \\log \\log N)$").next_to(known, direction=DOWN)
        self.play(Write(mprime_size), Write(rcc_mprime))
        self.next_slide()
        hhh22 = Tex("[\\texttt{HHH 2022}]: ${\\rm R}(M) = (\\log N)^{0.9}$ and yet ${\\rm R}(M') = O(1)$.")\
            .add_background_rectangle(color=DARK_BROWN).scale_to_fit_width(12)
        self.play(Write(hhh22))
        self.next_slide()


class CondensingRcc(Slide):
    def construct(self):
        random.seed(123323)
        caption = Tex("When ${\\rm R}$ is condensable").move_to(UP * 3.5) 
        self.play(Write(caption))
        self.next_slide()
        larc = Tex("{\\bf Log Approximate Rank Conjecture}: ${\\rm R}(M) = (\\log \\tilde{\\rm rk}(M))^C$ for some $C$")\
            .scale_to_fit_width(11).next_to(caption, direction=DOWN)
        self.play(Write(larc))
        self.next_slide()
        cms2020 = Tex("[\\texttt{CMS 2020}]: False for $F(x,y) := \\textsc{Sink}_n(x \\oplus y)$.\\\\"
                       "$\\tilde{\\rm rk}(F) = O(n^4); ~~{\\rm R}(F) = \Omega(n)$.").next_to(larc, direction=DOWN)
        self.play(Write(cms2020))
        self.next_slide()
        sink = DiGraph(list(range(7)), [(i, j) if random.randint(0,1) == 0 else (j, i) for i,j in combinations(range(7), 2)], 
                       layout="circular",
                       edge_config={"tip_config": {"tip_length": 0.35, "tip_width": 0.15}}).\
            next_to(cms2020, direction=DOWN).shift(LEFT*3)
        rect = Rectangle(height=4.5, width=5).next_to(sink, direction=RIGHT, buff=0.7).set_fill(color=WHITE, opacity=0.1).shift(DOWN * 0.7)
        palette = [DARK_BLUE, GREEN_C, RED_E, YELLOW_C, BLUE_C, LIGHT_BROWN, ORANGE]
        subrects0 = [Rectangle(color=palette[i], width=0.9, height=1.1).set_fill(color=palette[i], opacity=0.5) for i in range(7)]
        subrects = [VGroup(r, Tex("\\textsc{Eq}").move_to(r.get_center())) for r in subrects0]
        subrects[0].move_to(rect.get_top()).shift(DOWN * subrects[0].height / 2 + LEFT * (rect.width / 2 - subrects[0].width/2) + 0.05 * (DOWN + RIGHT))
        for i in range(1, 4):
            subrects[i].next_to(subrects[i-1], direction=RIGHT, buff=0.3)
            if i < 3:
                subrects[i].shift(DOWN * random.random() / 10)
        subrects[4].next_to(subrects[3], direction=DOWN)
        for i in range(5, 7):
            subrects[i].next_to(subrects[i-1], direction=LEFT, buff=0.1).shift(DOWN * random.random() * 1.1)
        pi = list(range(7))
        random.shuffle(pi)
        self.play(Create(sink), Create(rect))
        self.next_slide()
        
        for j,i in enumerate(pi):
            animations = []
            to_remove_later = []
            for a,b in sink.edges.keys():
                if a == i or b == i:
                    edge_cp = sink.edges[a,b].copy()
                    edge_cp.stroke_color = palette[i]
                    edge_cp.color = palette[i]
                    edge_cp.stroke_width *= 2
                    edge_cp.z_index = sink.edges[a,b].z_index + 100
                    to_remove_later.append(edge_cp)
                    animations.append(FadeIn(edge_cp))
            self.play(*animations, Create(subrects[i]), duration=0.1)
            if (j < 4):
                self.next_slide()
            animations = []
            for e in to_remove_later:
                animations.append(FadeOut(e))
            self.play(*animations, duration=0.01)
        
        self.next_slide()
        result = Tex("{\\bf Theorem}: \\texttt{[this work]} There is a $2^{O(n)} \\times 2^{O(n)}$ submatrix $F'$ of $F$"
                     " that has $\\tilde{\\rm rk}(F') = O(n^3);$ and ${\\rm R}(F') = \Omega(n)$.")\
                     .add_background_rectangle(color=DARK_BROWN, opacity=1)\
                     .move_to(cms2020.get_center()).scale_to_fit_width(cms2020.width * 1.3)
        self.play(Write(result), FadeIn(Rectangle(height=3, width=3).set_fill(WHITE, opacity=0.5).
                                        move_to(rect.get_center())), 
                                FadeIn(MathTex("F'").move_to(rect.get_center())))
        self.next_slide()

class Conclusion(Slide):
    def construct(self):
        random.seed(123323)
        full_slide = Tex("""
                        {\\bf Open Problems}
                        \\begin{itemize}"""
                        """ \\item Is deterministic communication losslessly condensable? What about unambiguous certificate complexity?"""
                        """ \\item Construct explicit Shearer extractors."""
                        """\\end{itemize}
                        """).scale_to_fit_width(12).move_to(UP * 2.5)
        self.play(Write(full_slide))
        self.next_slide()
        shearer_def = Tex("""{\\bf Definition.} ${\\rm Ext}\\colon \\{0,1\\}^{cn} \\to \{0,1\}^m$ is $(k,\\epsilon)$-Shearer extractos wrt to $S_1, \dots, S_n \subseteq [m]$ if 
                          $$\Delta(({\\bf U}_n, {\\bf Y}), ({\\rm Ext}({\\bf X})_{S_{\\bf Y}}, {\\bf Y})) \le \epsilon$$
                          for every entropy-$k$ source ${\\bf X}$ where ${\\bf Y} \sim [n]$, ${\\bf U}_n \sim \{0,1\}^n.$""").scale_to_fit_width(12).next_to(full_slide, direction=DOWN)
        self.play(Write(shearer_def))
        self.next_slide()
        input_r = Rectangle(width=3, height=0.6)
        input_t = MathTex("cn").move_to(input_r.get_center())
        input = Group(input_r, input_t)
        outputs = [Group(Rectangle(width=1.6, height=0.6), MathTex("n")) for _ in range(5)]
        input.next_to(shearer_def, direction=DOWN)
        input.shift(LEFT * 5)
        outputs[0].next_to(input, direction=RIGHT, buff=0.7)
        for i in range(1, len(outputs)):
            outputs[i].next_to(outputs[i-1], direction=RIGHT, buff=0.05)
        arr = Arrow(input.get_right(), outputs[0].get_left())
        self.play(FadeIn(input), FadeIn(arr), *[FadeIn(o) for o in outputs])
        self.next_slide()
        self.play(*[outputs[i].animate.shift(i * LEFT * outputs[i].width / 4) for i in range(len(outputs))])
        self.next_slide()