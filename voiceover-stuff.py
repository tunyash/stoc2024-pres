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


class ColoringSlide0(VoiceoverScene):
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
            answer = Tex("[Erd\\\"os, 1959] \\textbf{No}, can have arbitrary large girth and chromatic number.")\
                .scale(0.7).next_to(question, direction=DOWN)
            answer.add_background_rectangle(BLACK, opacity=0.9)
            self.add(answer)
            for subgraph in [[1,2,3,4], [5,6,7,8], [18,19,20,21]]: 
                group = VGroup(*[graph[k] for k in subgraph], *[graph.edges[i,j] for i,j in edges if i in subgraph and j in subgraph])
                self.play(Indicate(group))

class VertexCoverSlide0(VoiceoverScene):
    def construct(self):
        random.seed(123323)
        self.set_speech_service(GTTSService())
        caption = Text("Vertex Cover").move_to(UP * 3)
        nodes = range(1,41)
        edges = [(random.randint(1,7), random.randint(1,40)) for _ in range(80)]
        edges = [(x, y) for x,y in edges if x != y]
        graph = Graph(nodes, edges, layout="kamada_kawai").next_to(caption, direction=DOWN)
        for e in edges:
            graph.edges[e].fade(0.8)
            
        with self.voiceover(""" 
                      A vertex cover is a set of nodes such that every edge in the graph is
                      incident to one of them.
                      """):
            self.play(Write(caption))
            self.play(Create(graph))
        with self.voiceover("""
                      For example here the nodes in the center form a vertex cover of the graph.
                            """):
            for v in range(1,8):
                graph[v].set_color(RED_C)
            animations = [graph[v].animate.move_to(RIGHT / 2 * math.cos(2*math.pi / 7 * v)
                                                            + UP / 2 * math.sin(2*math.pi / 7 * v)) for v in range(1,8)]
            animations.extend([graph[v].animate.move_to(RIGHT * 2 * math.cos(2*math.pi / 33 * (v - 8))
                                                            + UP * 2 * math.sin(2*math.pi / 33 * (v - 8))) for v in range(8, 41)])
            
            self.play(*animations) 
        question = Tex("Is there a subgraph of size $O(k^2)$ that has a vertex cover $k$?").next_to(graph, direction=DOWN)
        with self.voiceover("""
                            Can we condense (or locally witness) vervex cover? For example,
                            is there a subgraph of size depending only on k and the same vertex cover?
                            """):
            self.play(Write(question))
        with self.voiceover("""
                            The answer is yes. This is studied in the field of fixed-parameter tractability.
                            """):
            answer = Tex("\\textbf{Yes}: FPT kernelization.").next_to(question, direction=DOWN)
            self.play(Write(answer))
        with self.voiceover("""
                            An algorithmically efficient way to find a smaller graph that has vertex cover k'
                            if and only if the original graph had vertex cover k is called kernelization.
                            """):
            self.play(graph.animate.shift(LEFT * 3))
            lbrace = Tex("$\Big($").next_to(graph, direction=LEFT)
            rbrace = Tex("$,k\Big)$").next_to(graph, direction=RIGHT)
            arrow = Arrow(start=rbrace.get_right(), end=RIGHT)
            lbrace2 = Tex("$\Big($").next_to(arrow, direction=RIGHT)
            small_graph = Graph(list(range(7)), list(combinations(range(7), 2)), layout="circular")\
                .scale(0.3).next_to(lbrace2, direction=RIGHT)
            rbrace2 = Tex("$,k'\Big)$").next_to(small_graph, direction=RIGHT)
            self.play(FadeIn(arrow), FadeIn(lbrace, rbrace, lbrace2, rbrace2), Create(small_graph))

class QueryComplexityIntroSlide(VoiceoverScene):
    def construct(self):
        random.seed(123323)
        self.set_speech_service(GTTSService())
        caption = Text("Query Complexity").move_to(UP * 3) 
        with self.voiceover("""
                            The first computation complexity measure we will try to condense
                            is query complexity: the number of bit probes
                            one has to make in order to compute the value of the function.
                            Equivalently, the smallest depth of a decision tree computing 
                            the function.
                            """):
            q_def = Tex("""
                        \\textbf{Definition}. Query complexity of a function """, """$f\colon \\{0,1\\}^n \\to \mathcal{O}$ """,
                                            """is the \emph{depth} of decision tree computing $f$.
                        """).scale_to_fit_width(9)
            self.play(Write(q_def), FadeIn(caption))
        with self.voiceover("""Let us illustrate this notion with a simple example."""):
            self.play(q_def.animate.shift(UP * 2))
        tree_or = BinaryTree(13, tp="bamboo")
        for i in range(tree_or.n):
            tree_or.label[i] = tree_or.height[i] + 1
        tree_or.leaf_labels = {i: int(any(j == 1 for j in tree_or.get_assignment(i))) for i in tree_or.leaves}
        g_or = Graph(range(tree_or.n), tree_or.get_edges(),
                  layout=tree_or.layout,
                  labels=tree_or.get_tex_labels()).scale_to_fit_width(8).center().shift(DOWN * 0.5)
        labels_or = [t for t in tree_or.labels_for_graph(g_or)]
        with self.voiceover("This decision tree computes the function OR."):
            self.play(Create(g_or), *[Create(t) for t in labels_or])
        or_formula = []
        for i in range(tree_or.n):
            if i not in tree_or.leaves:
                if len(or_formula) > 0:
                    or_formula.append("\\lor")
                or_formula.append("x_{" + str(tree_or.label[i]) + "}")
        or_formula_disp = MathTex(*or_formula).next_to(g_or, direction=DOWN)
        with self.voiceover("""The disjunction of all the variables."""):
            self.play(Write(or_formula_disp))
        depth_indicator = Line(start=[g_or.get_left()[0],
                                      g_or.vertices[0].get_center()[1], 0],
                               end=[g_or.get_left()[0],
                                    g_or.get_bottom()[1], 0])
        depth_text = Tex("depth=$" + str(max(tree_or.height)) + "$").rotate(PI/2).scale(0.5)
        depth_text.move_to(depth_indicator.get_center() + LEFT * 0.3)
        with self.voiceover("""The depth of this tree equals the number of the variables."""):
            self.play(Create(depth_indicator), Write(depth_text))
        statement_text = Tex(""" \\textbf{Fact}. Query complexity of ${\\ OR}_n(x) := \\bigvee_{i=1}^n x_i$ 
                                                 is \emph{exactly} $n$.
                             """).scale(0.5).shift(UP)
        with self.voiceover("""This is not very hard to show, that this depth is required, 
                             no matter what tree we use."""):
            self.play(g_or.animate.set_opacity(0.1), *[FadeOut(t) for t in labels_or], FadeOut(depth_text, depth_indicator),
                       FadeIn(statement_text))
        with self.voiceover("""
                            Whatever bits we query, if we only get zeroes, the value
                            of the function is not fixed until the very end, if it
                            is zero, then the OR is zero, and if it is one, the OR is one.
                            """):
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
            for val in ["0", "1"]:
                cur_text[last] = val
                new_text_disp = Tex(*cur_text).scale_to_fit_width(5)
                self.play(ReplacementTransform(cur_text_disp, new_text_disp))
                cur_text_disp = new_text_disp

    
class QueryComplexityCondensationSlide(VoiceoverScene):
    def construct(self):
        random.seed(123323)
        self.set_speech_service(GTTSService())
        caption = Text("Condensing Query Complexity").move_to(UP * 3) 
        with self.voiceover("""
            Suppose a function f has query complexity k then can one fix all-but big-O of k
            input bits such that query complexity of the restricted function is still k?
                       """):
             condensation_def = Tex("\\textbf{Question}: "
                                    "$f\colon \\{0,1\\}^n \\to \\{0,1\\}$ has query complexity $k$."
                                    " Find $\\rho \\subseteq \\{0,1, *\\}^n$ with $\\rho^{-1}(*) = O(k)$ "
                                    "such that $f|_{\\rho}$ retains query complexity $k$.")\
                                    .scale_to_fit_width(9).next_to(caption, direction=DOWN)
             self.play(FadeIn(caption), Write(condensation_def))
        with self.voiceover("""
                            If query complexity is maximal, then the function is already condensed,
                            so let us look at some example of intermediate query complexity.
                            """):
            pass
        with self.voiceover("""
                            One such function is called sink, the input encodes directions of edges
                            in a complete directed graph (known as tournament graph). 
                            """):
            sink_def = Tex("\\textbf{Example}: $\\textsc{Sink}\\colon \\{0,1\\}^{\\binom{n}{2}} \\to \\{0,1\\}$ is defined"
                           " such that $\\textsc{Sink}(G) = 1$ iff $G$ has a sink.")\
                            .scale_to_fit_width(9).next_to(condensation_def, direction=DOWN)
            edges = [(i, j) if (random.randint(1,2) == 1 or j == 3) and (not i == 3) else (j, i) 
                                       for i,j in combinations(range(7), 2)]
            sink_graph = DiGraph(list(range(7)), 
                                 edges,
                                 layout="circular", layout_scale=1.5, 
                                 edge_config={"tip_config": {"tip_length": 0.35, "tip_width": 0.15}})\
                                 .next_to(sink_def, direction=DOWN)
            self.play(Write(sink_def), Create(sink_graph))
            
        
        with self.voiceover("""
                            The function should return 1 if and only if the graph has a sink, i.e. 
                            there is a node with all edges directed towards it.
                            """):
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
        with self.voiceover("""
                            Query complexity of this function is n, which is about the square root
                            of the number of input bits. 
                            """):
            sink_qc = Tex("Query complexity of $\\textsc{Sink}$ is $\Theta(n)$.")\
                .next_to(sink_graph, direction=DOWN)
            
            self.play(Write(sink_qc))
        with self.voiceover("""
                            Then if we restrict all edges that are not incident to one of the nodes
                            such that there is a cycle among them, then the restricted function is 
                            1 if and only if all the edges are directed towards the fixed node.
                            """):
            ham_path = None
            for pp in permutations([i for i in range(7) if i != 3]):
                p = list(pp)
                if all(e in edges for e in zip([p[-1]] + p[:-1], p)):
                       ham_path = list(zip([p[-1]] + p[:-1], p))
                       break
            animations = []
            for e in ham_path:
                edge_cp = sink_graph.edges[e].copy()
                edge_cp.stroke_color = GREEN_C
                edge_cp.color = GREEN_C
                animations.append(ReplacementTransform(sink_graph.edges[e], edge_cp))
            for i in range(7):
                if i == 3:
                    continue
                edge_cp = Line(sink_graph[i].get_center(), sink_graph[3].get_center(), stroke_color=YELLOW_C)
                animations.append(FadeOut(sink_graph.edges[(i,3)]))
                animations.append(FadeIn(edge_cp))
            self.play(*animations, FadeOut(lbrace, rbrace))
            self.add(Text("1").scale(0.7).next_to(sink_graph[3], direction=UP))
        with self.voiceover("""
                            The query complexity of the restricted function is exactly n minus one
                            since it is equivalent to the OR function up to flipping the signs of some bits.
                            """):
            answer = Tex("Hence for \\textsc{Sink} the condensing restriction $\\rho$ is any restriction"
                         " of the edges $\\binom{\{2,\dots,n\}}{2}$"
                         " that fixes a cycle over all nodes.").scale_to_fit_width(9).next_to(sink_qc, direction=DOWN)
            self.play(Write(answer))
        with self.voiceover("""
                            Therefore for Sink the answer to our initial question is positive, query complexity can be 
                            condensed by restriction. So is this true in general?
                            """):
            condensation_def.add_background_rectangle(color=DARK_BLUE, opacity=0.95)
            target = condensation_def.copy()
            target.z_index = 10000
            target.scale(1.3)
            target.center()
            self.play(ReplacementTransform(condensation_def, target))
        with self.voiceover("""
                            Our main result here is that it is not, there is a function of query
                            complexity k such that for all restrictions rho that leave big-O of k 
                            variables free the query complexity of the restricted version of the
                            function falls polynomially.
                            """):
            answer = Tex("\\textbf{Theorem}: \\\\"
                         "There exists $f\\colon \\{0,1\\}^n \\to \\{0,1\\}$ with query complexty $k$ "
                         "such that for all $\\rho$ with $|\\rho^{-1}(*)| = O(k)$ "
                         "the query complexity of $f|_{\\rho}$ is $\\tilde{O}(k^{3/4})$.")\
                            .scale_to_fit_width(condensation_def.width).next_to(condensation_def, direction=DOWN)
            answer.add_background_rectangle(DARK_BROWN, opacity=0.95)
            self.play(Write(answer))
            