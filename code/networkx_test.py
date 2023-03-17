import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_nodes_from(range(25))
interframe_edges = [
    (0,12),
    (0,16),
    (0,1),
    (1,20),
    (2,3),
    (2,20),
    (4,20),
    (4,5),
    (5,6),
    (6,7),
    (7,21),
    (7,22),
    (8,9),
    (9,10),
    (10,11),
    (11,23),
    (11,24),
    (12,13),
    (13,14),
    (14,15),
    (15,16),
    (15,17),
    (17,18),
    (18,19)
]

edges = [
    (0,1),(0,12),(0,16),
    (1,20),
    (2,3), (2,20),
    (4,20),(4,5),
    (5,6),
    (6,7),(6,22),
    (7, 21),
    (8,20), (8,9),
    (9,10),
    (10,11),(10,24),
    (11,23),
    (12, 13),
    (13,14),
    (14,15),
    (16, 17),
    (17, 18),
    (18, 19)

]

joint_names = [
"SPINBASE",
"SPINMID",
"NECK",
"HEAD",
"SHOULDERLEFT",
"ELBOWLEFT",
"WRISTLEFT",
"HANDLEFT",
"SOULDERRIGHT",
"ELBOWRIGHT",
"WRISTRIGHT",
"HANDRIGHT",
"HIPLEFT",
"KNEELEFT",
"ANKLELEFT",
"FOOTLEFT",
"HIPRIGHT",
"KNEERIGHTIGHT"
]
G.add_edges_from(edges)
nx.draw(G, with_labels=True, font_weight='bold')

plt.show()