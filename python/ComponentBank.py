# Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
# Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
# If you use this code, cite:
#   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
#   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
# all components within a bank share one graph
import networkx as nx

class ComponentBank:
    def __init__(self, graphType, edgeTypeFilter, n_nodes, node_metadata, imgsz):
        if imgsz is None:
            imgsz = []
        self.imgsz = imgsz
        self.graph_type = graphType
        self.edge_type_filter = edgeTypeFilter
        self.g = nx.DiGraph() # keeps track of the edges within this bank
        self.edge_states = []
        self.cmp_metadata = {}
        self.g.node_metadata = {}
        self.g.edge_metadata = {}
        self.g.edge_metadata['is_right'] = []
        self.g.edge_metadata['is_down'] = []
        self.g = self.g.add_nodes_from(range(1, n_nodes+1))
        if 'name' not in node_metadata or 'chanidx' not in node_metadata:
            raise ValueError("node_metadata must contain 'name' and 'chanidx' fields")
        for key in node_metadata:
            if isinstance(node_metadata[key], list):
                self.g.node_metadata[key] = []
            else:
                self.g.node_metadata[key] = None
        self.g.node_metadata['name'] = node_metadata['name']
        if self.graph_type == 'GRID2DMULTICHAN':
            self.g.node_metadata['chanidx'] = node_metadata['chanidx']
        self.g.edge_metadata['is_right'] = [False] * self.g.number_of_edges()
        self.g.edge_metadata['is_down'] = [False] * self.g.number_of_edges()
        self.edge_states = [[]] * self.g.number_of_edges()

    def InsertComponents(self, n_new):
        if self.g.number_of_edges() == 0:
            self.edge_states = [[]] * n_new
        else:
            self.edge_states += [[]] * n_new

    def SubsetComponents(self, keep):
        if isinstance(keep, bool) or all(isinstance(x, int) for x in keep):
            m = self.cmp_metadata
            for key in m:
                m[key] = [m[key][i] for i in range(len(m[key])) if keep[i]]
            self.edge_states = [self.edge_states[i] for i in range(len(self.edge_states)) if keep[i]]
            self.cmp_metadata = m
        else:
            raise ValueError("keep must be a boolean mask or an index")

    def InsertNodes(self, nodeIDs, nodeName):
        n_new_nodes = len(nodeIDs)
        n_orig_nodes = self.g.number_of_nodes() # BEFORE any changes to obj
        edges = []
        isEdgeRight = []
        isEdgeDown = []
        nodeChan = []
        if self.graph_type == 'GRID2D' or self.graph_type == 'GRID2DMULTICHAN':
            for i in range(n_orig_nodes + n_new_nodes):
                for j in range(i+1, n_orig_nodes + n_new_nodes):
                    edges.append((i+1, j+1))
                    isEdgeRight.append(False)
                    isEdgeDown.append(False)
                    nodeChan.append(None)
        else:
            raise ValueError("Invalid graph type")
        origNodes = list(self.g.nodes)
        self.g.add_nodes_from(nodeIDs)
        if nodeName is not None and len(nodeName) > 0:
            self.g.node_metadata['name'][-n_new_nodes:] = nodeName
        if self.graph_type == 'GRID2DMULTICHAN':
            self.g.node_metadata['chanidx'] = nodeChan
        if len(origNodes) > 0:
            drop = [(edges[i][0] in origNodes and edges[i][1] in origNodes) for i in range(len(edges))]
            edges = [edges[i] for i in range(len(edges)) if not drop[i]]
            isEdgeRight = [isEdgeRight[i] for i in range(len(isEdgeRight)) if not drop[i]]
            isEdgeDown = [isEdgeDown[i] for i in range(len(isEdgeDown)) if not drop[i]]
        self.g.add_edges_from(edges)
        n_new_edges = len(edges)
        if self.n_cmp == 0:
            self.edge_states = [[]] * n_new_edges
        else:
            self.edge_states += [[]] * n_new_edges
        self.g.edge_metadata['is_right'][-n_new_edges:] = isEdgeRight
        self.g.edge_metadata['is_down'][-n_new_edges:] = isEdgeDown

    def RemoveNodes(self, nodeIDs2Remove):
        self.g.remove_nodes_from(nodeIDs2Remove)
        mask = [not (self.g.edge_endnode_src[i] in nodeIDs2Remove or self.g.edge_endnode_dst[i] in nodeIDs2Remove) for i in range(self.g.number_of_edges())]
        self.edge_states = [self.edge_states[i] for i in range(len(self.edge_states)) if mask[i]]

    def ToMatlabDigraph(self):
        return self.g

    @property
    def n_cmp(self):
        return len(self.edge_states)

    @property
    def cmp_name(self):
        return [str(i) for i in range(self.n_cmp)]

    @property
    def edge_endnodes(self):
        return [(self.g.edge_endnode_src[i], self.g.edge_endnode_dst[i]) for i in range(self.g.number_of_edges())]

    @property
    def edge_endnode_idx(self):
        idx = list(range(1, self.g.number_of_nodes()+1))
        idx = [idx[i] for i in range(len(idx)) if i+1 in self.g.nodes]
        return [(idx[self.g.edge_endnode_src[i]], idx[self.g.edge_endnode_dst[i]]) for i in range(self.g.number_of_edges())]


