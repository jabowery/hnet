# Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
# Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
# If you use this code, cite:
#   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
#   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
# INPUTS
#   model
#   dat
#   alg               - scalar (string) "ica" | ... more pending ... | "spectralkmeans" | "kmeans" | "gmm" | "hierarchical*"
#   k                 - scalar (int-valued numeric)
#   max_edges_per_cmp - scalar (int-valued numeric)
#   bank2Cluster      - (char)
#   bank2FormCmp      - (char)
#   mode              - scalar (string) "unsup" | "sup1" | "sup2" | "unsupsplit" | "sup1split" | "sup2split"
# RETURNS
#   model
def FactorEdgesToExtractComponents(model, dat, alg, k, max_edges_per_cmp, bank2Cluster, bank2FormCmp, mode):
    k_per_class = round(k / dat.n_classes)
    n_edges = model.compbanks[bank2FormCmp].g.n_edges
    t = tic()
    compCode = Encode(model, dat.pixels)
    nodeActivations = compCode[bank2Cluster] # n_nodes x n_pts (logical or numeric)
    n_pts = nodeActivations.shape[1]
    if mode.startswith("sup1") or mode.startswith("sup2"):
        assert dat.n_classes == 2
        nodeActivations = np.vstack((nodeActivations, dat.label_idx-1))
        didx = NeighborPairs(model.compbanks[bank2FormCmp].graph_type, model.compbanks[bank2FormCmp].g.n_nodes+1)
    else:
        didx = model.compbanks[bank2FormCmp].edge_endnode_idx
    edgeRelations = GetEdgeStates(nodeActivations, didx, model.compbanks[bank2FormCmp].edge_type_filter)
    if mode.endswith("split"):
        edgeStates = EDG([])
        for c in range(1, dat.n_classes+1):
            currEdgeRelations = edgeRelations[:,dat.label_idx == c]
            usedEdgeMsk = (np.sum(currEdgeRelations, axis=1) > n_pts * 0.05) # ignore edges that are almost always n/a
            curr = EDG(np.zeros((n_edges, k_per_class), dtype=np.uint8))
            curr[usedEdgeMsk,:] = Factor(alg, currEdgeRelations[usedEdgeMsk,:], k_per_class, model.compbanks[bank2FormCmp].edge_type_filter)
            if mode.startswith("sup1"): # if sup2, leave them in for the below func
                curr[(didx[:,0] == np.max(didx)) | (didx[:,1] == np.max(didx)),:] = EDG.NULL # remove edges to the dv (last node)
            curr[usedEdgeMsk,:] = CropLeastCoOccurringEdges(curr[usedEdgeMsk,:], currEdgeRelations[usedEdgeMsk,:], max_edges_per_cmp)
            if mode.startswith("sup2"):
                curr[(didx[:,0] == np.max(didx)) | (didx[:,1] == np.max(didx)),:] = EDG.NULL # remove edges to the dv (last node)
            edgeStates = np.concatenate((edgeStates, curr), axis=1)
    else:
        usedEdgeMsk = (np.sum(edgeRelations, axis=1) > n_pts * 0.05) # ignore edges that are almost always n/a
        edgeStates = EDG(np.zeros((n_edges, k), dtype=np.uint8))
        edgeStates[usedEdgeMsk,:] = Factor(alg, edgeRelations[usedEdgeMsk,:], k, model.compbanks[bank2FormCmp].edge_type_filter)
        if mode.startswith("sup1"): # if sup2, leave them in for the below func
            edgeStates[(didx[:,0] == np.max(didx)) | (didx[:,1] == np.max(didx)),:] = EDG.NULL # remove edges to the dv (last node)
        edgeStates[usedEdgeMsk,:] = CropLeastCoOccurringEdges(edgeStates[usedEdgeMsk,:], edgeRelations[usedEdgeMsk,:], max_edges_per_cmp)
        if mode.startswith("sup2"):
            edgeStates[(didx[:,0] == np.max(didx)) | (didx[:,1] == np.max(didx)),:] = EDG.NULL # remove edges to the dv (last node)
    if mode.startswith("sup1") or mode.startswith("sup2"):
        edgeStates = np.delete(edgeStates, np.where((didx[:,0] == np.max(didx)) | (didx[:,1] == np.max(didx))), axis=0)
        didx = np.delete(didx, np.where((didx[:,0] == np.max(didx)) | (didx[:,1] == np.max(didx))), axis=0)
        assert np.all(didx == model.compbanks[bank2FormCmp].edge_endnode_idx)
    # handle too few edges
    nEdgesPerCmp = np.sum(edgeStates != EDG.NULL, axis=0)
    mask = (nEdgesPerCmp < Config.MIN_EDGES_PER_CMP)
    print("removing " + str(np.sum(mask)) + " components for having too few edges")
    edgeStates[:,mask] = []
    model = model.ClearComponents(bank2FormCmp)
    model = model.InsertComponents(bank2FormCmp, edgeStates.shape[1])
    model.compbanks[bank2FormCmp].edge_states[:] = edgeStates
    Toc(t, toc(t) > 1)


