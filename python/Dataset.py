# Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
# Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
# If you use this code, cite:
#   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
#   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
from LoadCredit import LoadCredit

class Dataset:
    def __init__(self, frontendSpec, trnOrTst):
        self.frontend_spec = str(frontendSpec)
        self.img_sz = []  # [n_rows,n_cols,n_chan] or empty if not an image dataset
        self.uniq_classes = []
        self.pixels = []  # n_nodes x n
        self.label_idx = []  # n x 1 or empty
        self.pixel_metadata = {}  # each entry n_nodes x 1
        self.label_metadata = {}
        self.other_metadata = {}

        spec = frontendSpec
        if "." in frontendSpec:
            temp = frontendSpec.split(".")
            spec = temp[0]
            n_per_class = float(temp[1])  # only used if trnOrTst == "trn"

        is_trn = (trnOrTst == "trn")

        if (spec == "mnistpy") or (spec == "mnistmat") or (spec == "fashion") or (spec == "emnistletters"):
            self.img_sz = [28, 28, 1]
            self.pixels, self.label_idx, self.uniq_classes, self.pixel_metadata, self.label_metadata, self.other_metadata = LoadMNIST(spec, is_trn, n_per_class)
        elif (spec == "ucicredit") or (spec == "ucicreditaustralian") or (spec == "ucicreditgerman"):
            self.pixels, self.label_idx, self.uniq_classes, self.pixel_metadata, self.label_metadata, self.other_metadata = LoadCredit(spec, is_trn)
        elif spec == "clevr":
            self.img_sz = [80, 120, 24]
            self.pixels, self.pixel_metadata, self.label_metadata, self.other_metadata = LoadCLEVR(spec, is_trn)
            if is_trn:
                RenderCLEVRDatasetSummary(self)  # must be before the foveation
            row, col = PixelRowCol(self.img_sz)
            n_nodes = self.img_sz[0] * self.img_sz[1] * self.img_sz[2]
            # to help with foveation, find connected parts in each image
            compbank = ComponentBank(GRF.GRID2DMULTICHAN, [EDG.NCONV, EDG.NIMPL], n_nodes, self.pixel_metadata, self.img_sz)
            compbank.cmp_metadata = {"src_img_idx": [], "src_chan": []}
            for i in range(1, len(self.other_metadata["chan_color"]) + 1):  # for each channel
                compbank = compbank.InsertComponents(self.other_metadata["n"])
                pixels = self.other_metadata["img"]  # n_rows x n_cols x n_chan x n
                pixels[:,:,[ind for ind in range(pixels.shape[2]) if ind != (i-1)]] = 0
#                pixels[:, :, [i - 1, i + 1:], :] = 0  # mask all channels but one
#                pixels(:,:,[1:i-1,i+1:end],:) = 0; % mask all channels but one

                pixels = pixels.reshape(-1, self.other_metadata["n"])  # n_nodes x n
                compbank.edge_states[:, -self.other_metadata["n"]:] = GetEdgeStates(pixels, compbank.edge_endnode_idx, compbank.edge_type_filter)  # convert from pixels to edges
                compbank.cmp_metadata["src_chan"].extend([i] * self.other_metadata["n"])
            compbank.cmp_metadata["src_img_idx"] = list(range(1, self.other_metadata["n"] + 1))
            max_length = 25  # max length of a connected component (e.g. Inf, 20)
            newRelations, metadata = ExtractConnectedPartComponents(compbank, self.img_sz, max_length, 1.5)
            compbank = compbank.SubsetComponents([False] * compbank.n_cmp)
            compbank = compbank.InsertComponents(newRelations.shape[1])
            compbank.edge_states[:] = newRelations
            partimgidx = metadata["src_img_idx"]
            partpxcoords = np.zeros((compbank.n_cmp, 2))  # (:,1) = row, (:,2) = col
            for i in range(compbank.n_cmp):  # borrowed from Model.PixelCoords()
                edgeMsk = (compbank.edge_states[:, i] != EDG.NULL)
                pixelIdx = compbank.edge_endnode_idx[edgeMsk, :]  # get the pixel indices associated with component i
                partpxcoords[i, 0] = np.mean(row[pixelIdx.flatten()])
                partpxcoords[i, 1] = np.mean(col[pixelIdx.flatten()])
            partpxcoords = np.round(partpxcoords)
            shiftr = self.img_sz[0] / 2 - partpxcoords[:, 0]
            shiftc = self.img_sz[1] / 2 - partpxcoords[:, 1]
            # foveate
            self.other_metadata["n"] = len(partimgidx)
            self.label_metadata["image_idx"] = [self.label_metadata["image_idx"][i] for i in partimgidx]
            self.other_metadata["objects"] = [self.other_metadata["objects"][i] for i in partimgidx]
            self.other_metadata["img"] = self.other_metadata["img"][:, :, :, partimgidx]
            temp = np.zeros(self.other_metadata["img"].shape, dtype=self.other_metadata["img"].dtype)
            for i in range(self.other_metadata["n"]):
                if shiftr[i] > 0:
                    temp[int(shiftr[i]):, :, :, i] = self.other_metadata["img"][:int(-shiftr[i]) + 1, :, :, i]
                elif shiftr[i] < 0:
                    temp[:int(-shiftr[i]) + 1, :, :, i] = self.other_metadata["img"][int(shiftr[i]):, :, :, i]
                if shiftc[i] > 0:
                    temp[:, int(shiftc[i]):, :, i] = self.other_metadata["img"][:, :int(-shiftc[i]) + 1, :, i]
                elif shiftc[i] < 0:
                    temp[:, :int(-shiftc[i]) + 1, :, i] = self.other_metadata["img"][:, int(shiftc[i]):, :, i]
                for j in range(len(self.other_metadata["objects"][i])):  # for each object
                    self.other_metadata["objects"][i][j]["pixel_coords"][:2] = [self.other_metadata["objects"][i][j]["pixel_coords"][k] - partpxcoords[i, k] for k in range(2)]
            self.other_metadata["img"] = temp
            self.other_metadata["foveated_chan"] = metadata["src_chan"]
            # re-set self.pixels now that other_metadata.img has changed
            self.pixels = self.other_metadata["img"].reshape(-1, self.other_metadata["n"])  # n_nodes*n_chan x n
        elif spec == "clevrpossimple":  # clevr positions, simple version
            # images are 320x240, but clevrpos isn't a standard image-based dataset so we don't set self.img_sz
            self.pixels, self.pixel_metadata, self.label_metadata, self.other_metadata = LoadCLEVRPos(spec, is_trn)
        else:
            raise ValueError("unexpected frontend spec")

    def SubsetDatapoints(self, keep):
        assert isinstance(keep, bool) or all(IsIdx(keep))  # it's either a mask or an index
        x = copy.deepcopy(self)  # make a copy
        fn = x.label_metadata.keys()
        for i in range(len(fn)):
            if len(x.label_metadata[fn[i]]) == self.n_pts:  # a vector of correct length
                x.label_metadata[fn[i]] = [x.label_metadata[fn[i]][j] for j in range(len(x.label_metadata[fn[i]])) if keep[j]]
            if x.label_metadata[fn[i]].shape[1] == self.n_pts:  # a matrix of correct length
                x.label_metadata[fn[i]] = x.label_metadata[fn[i]][:, keep]
        x.pixels = x.pixels[:, keep]
        x.label_idx = x.label_idx[keep]
        return x

    @property
    def n_pts(self):
        return self.pixels.shape[1]

    @property
    def n_nodes(self):
        return self.pixels.shape[0]

    @property
    def n_classes(self):
        return len(self.uniq_classes)


