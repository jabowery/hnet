# Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
# Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
# If you use this code, cite:
#   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
#   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
# INPUTS
#   modelName    - (str) name of a model (see Layout.m)
#   frontendSpec - (str) dataset and frontend name and parameters (see dataset.m)
#   trnSpec      - (str) training specification string
# USAGE
#   below calls are to be executed from within the "matlab" directory
#   Main("metacred",  "ucicreditgerman", "tier1.memorize-->tier1.extractcorr.icacropsome.100.50.unsupsplit-->meta.extractcorr.kmeans.10.50.unsupsplit");
#   Main("groupedimg", "mnistpy.128", "connectedpart.memorize-->connectedpart.extractconnec.25-->connectedpart.transl.2");
#   Main("clevrpos1", "clevrpossimple", "tier1.memorize");
from Config import Config
from GRF import GRF
from Layout import Layout
from Dataset import Dataset
from LoadCredit import LoadCredit

def Main(modelName, frontendSpec, trnSpec):
    assert Config.MyDir().endswith("/python") or Config.MyDir().endswith("\\python"), "Main() expects to be run from the hnet/python/ directory" 
    import os
    import json
    from pathlib import Path
    import numpy as np
    from scipy.io import savemat
    from scipy.sparse import csr_matrix
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS
    from matplotlib import pyplot as plt

    def endsWith(string, suffix):
        return string.endswith(suffix)

    def addpath(path):
        os.environ["PATH"] += os.pathsep + path

    def fullfile(*args):
        return os.path.join(*args)

    def isfolder(path):
        return os.path.isdir(path)

    def mkdir(path):
        os.makedirs(path)

    def disp(string):
        print(string)

    def writecell(filename, data):
        with open(filename, 'w') as f:
            for row in data:
                f.write(','.join(row) + '\n')

    def writetable(filename, data):
        data.to_csv(filename, index=False)

    def round(data, decimals):
        return np.round(data, decimals)

    def cat(dim, *args):
        return np.concatenate(args, axis=dim)

    def num2cell(data):
        return data.tolist()

    def table(data):
        return pd.DataFrame(data)

    def fieldnames(data):
        return data.columns.tolist()

    def struct2cell(data):
        return data.to_dict(orient='list').items()

    def addvars(data, *args, **kwargs):
        for key, value in kwargs.items():
            data[key] = value
        return data

    def RenderDatapointNoEdges(outDir, trndat, idx, name):
        pass

    def SetRNG(seed):
        np.random.seed(seed)

    def RenderCLEVRPosImg(outDir, trndat, model, i):
        pass

    def Export2JSON(trndat, tstdat, model, frontend_spec, name):
        pass

    def Encode(model, pixels):
        pass

    def ClassHistogram(labels, codes):
        pass

    def PrintPerformance(model, trndat_pixels, trndat_label_idx, tstdat_pixels, tstdat_label_idx, trnCode_comp_code, tstCode_comp_code, name, frontend_spec):
        pass

    def RenderAll(outDir, model, tstdat, tstCode, bank, arg1, arg2):
        pass

    def PrintEdgeRelations(outDir, filename, edge_states, edge_endnode_idx, node_info, bank, tier1_compbank_names):
        pass

    def RenderTstToExplain(outDir, model, tstdat, tstCode, output_bank_name, arg):
        pass

    def RenderComponentBestMatches(outDir, model, tstdat, tstCode, tier1_compbank_names):
        pass

    def RenderHist(outDir, uniq_classes, code, compbank_names, mode):
        pass

    def RenderDiscrimVsSharednessVsFrequency(outDir, model, dat, code, mode):
        pass

    def RenderGroup(outDir, model, tstdat, tstCode, i):
        pass

    def RenderMDS(outDir, tstdat, model, tstCode, arg1, arg2, arg3, arg4, arg5, arg6):
        pass

    def RenderFoveatedGroupImages(outDir, SubsetDatapoints, model, uniq_classes):
        pass

    def GetEdgeStates(pixels, edge_endnode_idx, edge_type_filter):
        pass

    def Edge2Logical(edge_states):
        pass

    def ml_Classify(trnCode_comp_code, trndat_label_idx, tstCode_comp_code, tstdat_label_idx, method, params, arg):
        pass

    def RenderDatapoint(outDir, model, tstdat, tstCode, idx, name):
        pass

    assert endsWith(Config.MyDir(), "/python") or endsWith(Config.MyDir(), "\\python"), "Main() expects to be run from the hnet/python/ directory" 
    addpath(os.path.join(Config.MyDir(), "matlab"))
    cfg = {"model_name": modelName, "frontend_spec": frontendSpec, "trn_spec": trnSpec}
    outDir = fullfile(Config.OUT_DIR, cfg["model_name"] + "_" + cfg["frontend_spec"] + "_" + cfg["trn_spec"].replace("-->", "-"))
    if not isfolder(outDir):
        mkdir(outDir)
    disp("=== " + outDir + " ===")
    layout = Layout(modelName)
    ## load dataset
    SetRNG(1000)
    trndat = Dataset(cfg["frontend_spec"], "trn")
    tstdat = Dataset(cfg["frontend_spec"], "tst")
    # print dataset info to text files
    temp = cat(2, trndat.pixel_metadata.name, num2cell(np.sum(trndat.pixels, axis=1)))
    writecell(fullfile(Config.OUT_DIR, "node_name_" + cfg["frontend_spec"] + ".csv"), cat(1, ['node_name', 'num nonzero pixels'], temp))
    if "category_info" in trndat.other_metadata:
        temp = cat(2, fieldnames(trndat.other_metadata.category_info), struct2cell(trndat.other_metadata.category_info))
        writecell(fullfile(Config.OUT_DIR, "category_info_" + cfg["frontend_spec"] + ".csv"), cat(1, ['field', 'value'], temp))
    if "t" in trndat.other_metadata:
        writetable(fullfile(Config.OUT_DIR, cfg["frontend_spec"] + "_trn.csv"), trndat.other_metadata.t)
        writetable(fullfile(Config.OUT_DIR, cfg["frontend_spec"] + "_tst.csv"), tstdat.other_metadata.t)
    if "t_bin" in trndat.other_metadata:
        writetable(fullfile(Config.OUT_DIR, cfg["frontend_spec"] + "_bin_trn.csv"), trndat.other_metadata.t_bin)
        writetable(fullfile(Config.OUT_DIR, cfg["frontend_spec"] + "_bin_tst.csv"), tstdat.other_metadata.t_bin)
    t = table(trndat.pixel_metadata.name)
    for i in range(trndat.n_classes):
        t = addvars(t, round(np.sum(trndat.pixels[:, trndat.label_idx == i], axis=1) / trndat.n_pts, 4), **{'NewVariableNames': ['frac_occurs_in_pts_of_class_' + trndat.uniq_classes[i]]})
    writetable(fullfile(Config.OUT_DIR, cfg["frontend_spec"] + "_var_cls_cooccur_frequency.csv"), t)
    # render example datapoints
    for c in range(trndat.n_classes):
        idx = np.where(trndat.label_idx == c)[0]
        for i in range(min(len(idx), 5)): # for each image in this class
            RenderDatapointNoEdges(fullfile(Config.OUT_DIR, "samplestim"), trndat, idx[i], trndat.uniq_classes[c] + "_" + str(i))
    ## train
    if Config.DO_CACHE:
        model = CachedCompute(Train, cfg, layout, trndat) # with caching
    else:
        model = Train(cfg, layout, trndat) # without caching
    ## export trained model
    try:
        Export2JSON(trndat, tstdat, model, cfg["frontend_spec"], cfg["model_name"] + "_" + cfg["frontend_spec"] + "_" + cfg["trn_spec"].replace("-->", "-"))
    except:
        disp("issue exporting to json... moving on")
    SetRNG(cfg) # reset RNG after training
    ## encode
    trnCode = {}
    tstCode = {}
    trnCode["comp_code"], trnCode["premerge_idx"] = Encode(model, trndat.pixels)
    tstCode["comp_code"], tstCode["premerge_idx"] = Encode(model, tstdat.pixels)
    if modelName == "clevrpos1" or modelName == "clevrpos2":
        for i in range(100):
            RenderCLEVRPosImg(outDir, trndat, model, i)
        return
    elif modelName == "clevr":
        return
    trnCode["hist"] = {}
    tstCode["hist"] = {}
    trnCode["comp_best_img"] = {}
    tstCode["comp_best_img"] = {}
    for i in range(model.n_compbanks):
        name = model.compbank_names[i]
        trnCode["hist"][name] = ClassHistogram(encode.OneHot(trndat.label_idx, trndat.n_classes), trnCode["comp_code"][name])
        tstCode["hist"][name] = ClassHistogram(encode.OneHot(tstdat.label_idx, tstdat.n_classes), tstCode["comp_code"][name])
        _, trnCode["comp_best_img"][name] = np.max(trnCode["comp_code"][name], axis=1)
        _, tstCode["comp_best_img"][name] = np.max(tstCode["comp_code"][name], axis=1)
    ## print accuracy and related stats
    PrintPerformance(model, trndat.pixels, trndat.label_idx, tstdat.pixels, tstdat.label_idx, trnCode["comp_code"][model.output_bank_name], tstCode["comp_code"][model.output_bank_name], cfg["model_name"] + "_" + cfg["trn_spec"], cfg["frontend_spec"])
    ## render the edges involved in each component
    for i in range(model.n_compbanks):
        bank = model.compbank_names[i]
        RenderAll(outDir, model, tstdat, tstCode, bank, True, True)
        RenderAll(outDir, model, tstdat, tstCode, bank, True, False)
        RenderAll(outDir, model, tstdat, tstCode, bank, False, True)
        # get metadata and pass to PrintEdgeRelations
        node_info = model.compbanks[bank].g.node_metadata.name
        if bank == model.tier1_compbank_names[0] and "category_info" in tstdat.other_metadata:
            for j in range(tstdat.n_nodes):
                if node_info[j] in tstdat.other_metadata.category_info:
                    node_info[j] = tstdat.other_metadata.category_info[node_info[j]]
        for j in range(min(100, model.compbanks[bank].n_cmp)): # for each component in this bank
            PrintEdgeRelations(outDir, bank + "_cmp" + str(j) + ".txt", model.compbanks[bank].edge_states[:, j], model.compbanks[bank].edge_endnode_idx, node_info, bank, model.tier1_compbank_names[0])
    ## explain each tst datapoint
    if modelName == "metacred":
        RenderTstToExplain(outDir, model, tstdat, tstCode, model.output_bank_name, True)
        RenderTstToExplain(outDir, model, tstdat, tstCode, model.output_bank_name, False)
    ## render component best matches
    for i in range(len(model.tier1_compbank_names)):
        # can only call for tier 1 and tier 2 component banks
        RenderComponentBestMatches(outDir, model, tstdat, tstCode, model.tier1_compbank_names[i])
        _, downstreamBankNames = outedges(model.g, model.tier1_compbank_names[i])
        for j in range(len(downstreamBankNames)):
            if downstreamBankNames[j] != "out":
                RenderComponentBestMatches(outDir, model, tstdat, tstCode, downstreamBankNames[j])
    ## render class histograms
    for i in range(model.n_compbanks):
        RenderHist(outDir, trndat.uniq_classes, trnCode, model.compbank_names[i], "trn")
        RenderHist(outDir, trndat.uniq_classes, tstCode, model.compbank_names[i], "tst")
    ## render discriminability vs sharedness of each component's response to the dataset
    if "meta" not in model.compbanks and "group" not in model.compbanks and trndat.n_classes > 2: # if we only have one tier
        RenderDiscrimVsSharednessVsFrequency(outDir, model, trndat, trnCode, "trn")
        RenderDiscrimVsSharednessVsFrequency(outDir, model, tstdat, tstCode, "tst")
    ## render the groups
    if "group" in model.compbanks:
        for i in range(min(10, model.compbanks.group.n_cmp)): # for each group
            RenderGroup(outDir, model, tstdat, tstCode, i)
    ## render PCA and multidimensional scaling plots
    RenderMDS(outDir, tstdat, model, tstCode, None, None, None, "correlation", "scatter", "scatter_corr")
    RenderMDS(outDir, tstdat, model, tstCode, None, None, None, "euclidean", "scatter", "scatter_eucl")
    ## render foveated group images
    if trndat.img_sz and "group" in model.compbank_names: # if the data is images and we have groups
        try:
            for c in range(trndat.n_classes):
                RenderFoveatedGroupImages(outDir, trndat.SubsetDatapoints(trndat.label_idx == c), model, trndat.uniq_classes[c])
        except:
            pass
    ## precompute some stats for below figures
    if isinstance(trnCode["comp_code"][model.output_bank_name], bool) or isinstance(trnCode["comp_code"][model.output_bank_name], np.uint8):
        knnParams = {"k": 1, "distance": "dot"}
        nbParams = {"distribution": "bern"}
    else: # feat codes are probably energies; ~poisson distributed
        knnParams = {"k": 1, "distance": "cosine"}
        nbParams = {"distribution": "gauss"}
    trnSenseEdges = Edge2Logical(GetEdgeStates(trndat.pixels, model.compbanks[model.tier1_compbank_names[0]].edge_endnode_idx, model.compbanks[model.tier1_compbank_names[0]].edge_type_filter))
    tstSenseEdges = Edge2Logical(GetEdgeStates(tstdat.pixels, model.compbanks[model.tier1_compbank_names[0]].edge_endnode_idx, model.compbanks[model.tier1_compbank_names[0]].edge_type_filter))
    try:
        predName = ['onenn', 'nb', 'net', 'svm']
        pred = {}
        _, pred["onenn"] = ml_Classify(trnCode["comp_code"][model.output_bank_name].T, trndat.label_idx, tstCode["comp_code"][model.output_bank_name].T, tstdat.label_idx, 'knn', knnParams, True)
        _, pred["nb"] = ml_Classify(trnCode["comp_code"][model.output_bank_name].T, trndat.label_idx, tstCode["comp_code"][model.output_bank_name].T, tstdat.label_idx, 'nbfast', nbParams, True)
        _, pred["net"] = ml_Classify(trnCode["comp_code"][model.output_bank_name].T, trndat.label_idx, tstCode["comp_code"][model.output_bank_name].T, tstdat.label_idx, 'patternnet', None, True)
        _, pred["svm"] = ml_Classify(trnCode["comp_code"][model.output_bank_name].T, trndat.label_idx, tstCode["comp_code"][model.output_bank_name].T, tstdat.label_idx, 'svmliblinear', None, True)
    except Exception as ex:
        print(ex)
        disp("terminating figure code early due to above issue")
        return
    ## render TP / FP examples
    for c in range(trndat.n_classes):
        className = trndat.uniq_classes[c]
        for i in range(len(predName)):
            idx = np.where((tstdat.label_idx != c) & (pred[predName[i]] == c))[0]
            idx = np.random.choice(idx, size=min(5, len(idx)), replace=False) # shuffle so we don't wind up with all the zeros and ones (limit to 5 renders per call)
            for j in range(len(idx)): # for each datapoint
                RenderDatapoint(outDir, model, tstdat, tstCode, idx[j], predName[i] + "_tst_fp" + className + "-" + str(j))
            idx = np.where((tstdat.label_idx == c) & (pred[predName[i]] == c))[0]
            idx = np.random.choice(idx, size=min(5, len(idx)), replace=False) # shuffle so we don't wind up with all the zeros and ones (limit to 5 renders per call)
            for j in range(len(idx)): # for each datapoint
                RenderDatapoint(outDir, model, tstdat, tstCode, idx[j], predName[i] + "_tst_tp" + className + "-" + str(j))


