class Noeud:
    def __init__(self, nom, x, y, z):
        self.nom = nom
        self.x = x
        self.y = y
        self.z = z


interframe_edges = [(0,12), (0,16),(0,1),(1,20),(2,3),(2,20),(4,20), (4,5),(5,6),(6,7),(7,21),(7,22),(8,9),(9,10),(10,11),(11,23),(11,24),(12,13),(13,14),(14,15),(15,16),(15,17),(17,18),(18,19)]

class Edge:
    def __init__(self, source, target, weight = 1):
        self.source = source
        self.target = target
        self.weight = weight


class Graphe :
    def __init__(self):
        self.noeuds = {}
        self.arcs = {}
        self.root = None
