import io
import numpy as np
from numpy.random import RandomState
import pandas as pd
from sklearn.preprocessing import normalize
from ioLoadCredit import ioLoadCredit
from Config import Config
from EqualizeN import EqualizeN
from RandSubsetDataset import RandSubsetDataset
from encodeTransformScalar2SpatialScalar import encodeTransformScalar2SpatialScalar
from encodeTransformScalar2SpikeViaKWTA import encodeTransformScalar2SpikeViaKWTA

def LoadCredit(spec, is_trn):
    pixel_metadata = {}
    label_metadata = {}
    other_metadata = {}
    
    # load
    if spec == "ucicredit":
        other_metadata = ioLoadCredit('uci_credit_screening', Config.DATASET_DIR + "/credit/uci_credit_screening")
    elif spec == "ucicreditaustralian":
        other_metadata = ioLoadCredit('uci_statlog_australian_credit', Config.DATASET_DIR + "/credit/uci_statlog_australian_credit")
    elif spec == "ucicreditgerman":
        other_metadata = ioLoadCredit('uci_statlog_german_credit', Config.DATASET_DIR + "/credit/uci_statlog_german_credit")
        # continuous vars: a2_duration, a5_creditscore, a8_percent (uniques=1,2,3,4), a11_presentresidencesince (uniques=1,2,3,4), a13_age, 16_ncredits (uniques={1,2,3,4}), a18_ndependents (uniques=1,2)
    else:
        raise ValueError("unexpected spec")
    
    # equalize n
    idx = EqualizeN(np.array(other_metadata['t']['dv']) + 1)
    other_metadata['t'] = other_metadata['t'].iloc[idx]
    other_metadata['t_bin'] = other_metadata['t_bin'].iloc[idx]
    
    # split trn from tst (keeping dv balanced)
    rng = RandomState(77)
    idx = RandSubsetDataset(np.array(other_metadata['t']['dv']) + 1, 0.5, rng)
    if is_trn:
        other_metadata['t'] = other_metadata['t'].iloc[idx]
        other_metadata['t_bin'] = other_metadata['t_bin'].iloc[idx]
    else:
        other_metadata['t'] = np.delete(other_metadata['t'], idx, axis=0)
        other_metadata['t_bin'] = np.delete(other_metadata['t_bin'], idx, axis=0)
    
    # separate the dv
    label_idx = np.array(other_metadata['t']['dv']) + 1
    other_metadata['t_bin'] = other_metadata['t_bin'].drop(columns=['dv'])
    
    if spec == "ucicredit":
        uniq_classes = ['-','+'] # according to documentation, "1 = good, 2 = bad"
    elif spec == "ucicreditaustralian":
        uniq_classes = ['-','+'] # according to documentation, "1 = good, 2 = bad"
    elif spec == "ucicreditgerman":
        uniq_classes = ['good','bad'] # according to documentation, "1 = good, 2 = bad"
    else:
        raise ValueError("unexpected spec")
    
    logicalMsk = other_metadata['t_bin'].apply(lambda x: x.dtype == bool)
    
    # extract node info
    pixel_metadata['name'] = list(other_metadata['t_bin'].columns[logicalMsk])
    pixels = other_metadata['t_bin'].loc[:,logicalMsk].T.to_numpy()
#    pixels = other_metadata['t_bin'][logicalMsk].T.to_numpy()
    
    # add binned versions of the non-logical fields
    n_spatial_stops = 5
    nonlogicalIdx = np.where(~logicalMsk)[0]
    for j in range(len(nonlogicalIdx)):
        varName = other_metadata['t_bin'].columns[nonlogicalIdx[j]]
        if (spec == "ucicredit") or (spec == "ucicreditaustralian"):
            data = encodeTransformScalar2SpatialScalar(normalize(other_metadata['t_bin'][varName].to_numpy().reshape(-1, 1), axis=0, norm='max'), n_spatial_stops)
            data = encodeTransformScalar2SpikeViaKWTA(data, 1, 2, [])
            pixels = np.concatenate((pixels, data.T))
            for k in range(n_spatial_stops):
                pixel_metadata['name'].append(varName + '_' + str(k))
        elif spec == "ucicreditgerman":
            if varName == "a8_percent" or varName == "a11_presentresidencesince" or varName == "16_ncredits":
                # all three vars hold integer values 1,2,3,4 - treat as categorical
                for k in range(1, 5):
                    pixels = np.concatenate((pixels, (other_metadata['t_bin'][varName] == k).to_numpy().reshape(1, -1)))
                    pixel_metadata['name'].append(varName + '_' + str(k))
            elif varName == "a18_ndependents": # takes integer values 1,2 - treat as binary
                other_metadata['t_bin']['a18_ndependents'] = (other_metadata['t_bin']['a18_ndependents'] == 2)
                pixels = np.concatenate((pixels, other_metadata['t_bin']['a18_ndependents'].to_numpy().reshape(1, -1)))
                pixel_metadata['name'].append('a18_ndependents')
            else:
                data = encodeTransformScalar2SpatialScalar(normalize(other_metadata['t_bin'][varName].to_numpy().reshape(-1, 1), axis=0, norm='max'), n_spatial_stops)
                data = encodeTransformScalar2SpikeViaKWTA(data, 1, 2, [])
                pixels = np.concatenate((pixels, data.T))
                for k in range(n_spatial_stops):
                    pixel_metadata['name'].append(varName + '_' + str(k))
        else:
            raise ValueError("unexpected spec")
    
    pixel_metadata['chanidx'] = np.ones((pixels.shape[0], 1))
    
    return pixels, label_idx, uniq_classes, pixel_metadata, label_metadata, other_metadata


