# Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
# Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
# If you use this code, cite:
#   Rodriguez A, Bowen EFW, Granger R (2022) https://github.com/DartmouthGrangerLab/hnet
#   Bowen, EFW, Granger, R, Rodriguez, A (2023). A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture. Presented at AAAI EDGeS 2023.
# INPUTS
#   name - scalar (string) name of layout to load
# RETURNS
#   layout - (cell array of structs)
from GRF import GRF
from EDG import EDG,EDGlist

def Layout(name):
    # load
    layout = {}
    if name == "basicimg":
        layout["connectedpart"] = {"graph_type": GRF.GRID2D, "edge_type_filter": [EDG.NCONV, EDG.NIMPL], "encode_spec": "energy"}
        layout["connec"] = 'sense-->connectedpart,connectedpart-->out'
    elif name == "basiccred":
        layout["tier1"] = {"graph_type": GRF.FULL, "edge_type_filter": [EDG.NCONV, EDG.NIMPL, EDG.AND], "encode_spec": "energy"}
        layout["connec"] = 'sense-->tier1,tier1-->out'
    elif name == "basiccredand":
        layout = {"name": 'tier1', "graph_type": GRF.FULL, "edge_type_filter": EDG.AND, "encode_spec": "energy"}
        layout["connec"] = 'sense-->tier1,tier1-->out'
    elif name == "groupedimg":
        layout["connectedpart"] = {"graph_type": GRF.GRID2D, "edge_type_filter": [EDG.NCONV, EDG.NIMPL], "encode_spec": "energy"}
        layout["group"] = {"graph_type": GRF.SELF, "edge_type_filter": [], "encode_spec": "max"}
        layout["connec"] = 'sense-->connectedpart,connectedpart-->group,group-->out'
    elif name == "groupedcred":
        layout["tier1"] = {"graph_type": GRF.FULL, "edge_type_filter": [EDG.NCONV, EDG.NIMPL, EDG.AND], "encode_spec": "energy"}
        layout["group"] = {"graph_type": GRF.SELF, "edge_type_filter": [], "encode_spec": "max"}
        layout["connec"] = 'sense-->tier1,tier1-->group,group-->out'
    elif name == "groupedwta20img":
        layout["connectedpart"] = {"graph_type": GRF.GRID2D, "edge_type_filter": [EDG.NCONV, EDG.NIMPL], "encode_spec": "energy"}
        layout["group"] = {"graph_type": GRF.SELF, "edge_type_filter": [], "encode_spec": "max-->wta.20"}
        layout["connec"] = 'sense-->connectedpart,connectedpart-->group,group-->out'
    elif name == "groupedwta20cred":
        layout["tier1"] = {"graph_type": GRF.FULL, "edge_type_filter": [EDG.NCONV, EDG.NIMPL, EDG.AND], "encode_spec": "energy"}
        layout["group"] = {"graph_type": GRF.SELF, "edge_type_filter": [], "encode_spec": "max-->wta.20"}
        layout["connec"] = 'sense-->tier1,tier1-->group,group-->out'
    elif name == "groupedabsimg":
        layout["connectedpart"] = {"graph_type": GRF.GRID2D, "edge_type_filter": [EDG.NCONV, EDG.NIMPL], "encode_spec": "energy"}
        layout["group"] = {"graph_type": GRF.SELF, "edge_type_filter": [], "encode_spec": "maxabs"}
        layout["connec"] = 'sense-->connectedpart,connectedpart-->group,group-->out'
    elif name == "groupedabscred":
        layout["tier1"] = {"graph_type": GRF.FULL, "edge_type_filter": [EDG.NCONV, EDG.NIMPL, EDG.AND], "encode_spec": "energy"}
        layout["group"] = {"graph_type": GRF.SELF, "edge_type_filter": [], "encode_spec": "maxabs"}
        layout["connec"] = 'sense-->tier1,tier1-->group,group-->out'
    elif name == "groupedabswta20img":
        layout["connectedpart"] = {"graph_type": GRF.GRID2D, "edge_type_filter": [EDG.NCONV, EDG.NIMPL], "encode_spec": "energy"}
        layout["group"] = {"graph_type": GRF.SELF, "edge_type_filter": [], "encode_spec": "maxabs-->wta.20"}
        layout["connec"] = 'sense-->connectedpart,connectedpart-->group,group-->out'
    elif name == "groupedabswta20cred":
        layout["tier1"] = {"graph_type": GRF.FULL, "edge_type_filter": [EDG.NCONV, EDG.NIMPL, EDG.AND], "encode_spec": "energy"}
        layout["group"] = {"graph_type": GRF.SELF, "edge_type_filter": [], "encode_spec": "maxabs-->wta.20"}
        layout["connec"] = 'sense-->tier1,tier1-->group,group-->out'
    elif name == "grouptransl2":
        layout["connectedpart"] = {"graph_type": GRF.GRID2D, "edge_type_filter": [EDG.NCONV, EDG.NIMPL], "encode_spec": "energytransl.2"}
        layout["group"] = {"graph_type": GRF.SELF, "edge_type_filter": [], "encode_spec": "max"}
        layout["connec"] = 'sense-->connectedpart,connectedpart-->group,group-->out'
    elif name == "metaimg": # meta hierarchy for image datasets
        layout["connectedpart"] = {"graph_type": GRF.GRID2D, "edge_type_filter": [EDG.NCONV, EDG.NIMPL], "encode_spec": "energy-->wta.20"}
        layout["meta"] = {"graph_type": GRF.FULL, "edge_type_filter": [], "encode_spec": "energy"}
        layout["connec"] = 'sense-->connectedpart,connectedpart-->meta,meta-->out'
    elif name == "metacred": # meta hierarchy for credit datasets
        layout["tier1"] = {"graph_type": GRF.FULL, "edge_type_filter": [EDG.NCONV, EDG.NIMPL, EDG.AND], "encode_spec": "energy-->wta.20"}
        layout["meta"] = {"graph_type": GRF.FULL, "edge_type_filter": [EDG.NCONV, EDG.NIMPL, EDG.AND], "encode_spec": "energy"}
        layout["connec"] = 'sense-->tier1,tier1-->meta,meta-->out'
    elif name == "metacredand": # meta hierarchy for credit datasets
        layout["tier1"] = {"graph_type": GRF.FULL, "edge_type_filter": EDG.AND, "encode_spec": "energy-->wta.20"}
        layout["meta"] = {"graph_type": GRF.FULL, "edge_type_filter": [], "encode_spec": "energy"}
        layout["connec"] = 'sense-->tier1,tier1-->meta,meta-->out'
    elif name == "metagrpimg": # meta hierarchy for image datasets
        layout["connectedpart"] = {"graph_type": GRF.GRID2D, "edge_type_filter": [EDG.NCONV, EDG.NIMPL], "encode_spec": "energy"}
        layout["group"] = {"graph_type": GRF.SELF, "edge_type_filter": [], "encode_spec": "max"}
        layout["meta"] = {"graph_type": GRF.NULL, "edge_type_filter": [], "encode_spec": "energy"}
        layout["metagroup"] = {"graph_type": GRF.SELF, "edge_type_filter": [], "encode_spec": "max"}
        layout["connec"] = 'sense-->connectedpart,connectedpart-->group,group-->meta,meta-->metagroup,metagroup-->out'
    elif name == "metagrpcred": # meta hierarchy for credit datasets
        layout["tier1"] = {"graph_type": GRF.FULL, "edge_type_filter": [], "encode_spec": "energy"}
        layout["group"] = {"graph_type": GRF.SELF, "edge_type_filter": [], "encode_spec": "max"}
        layout["meta"] = {"graph_type": GRF.FULL, "edge_type_filter": [], "encode_spec": "energy"}
        layout["metagroup"] = {"graph_type": GRF.SELF, "edge_type_filter": [], "encode_spec": "max"}
        layout["connec"] = 'sense-->tier1,tier1-->group,group-->meta,meta-->metagroup,metagroup-->out'
    elif name == "clevr":
        layout["connectedpart"] = {"graph_type": GRF.GRID2DMULTICHAN, "edge_type_filter": [EDG.NCONV, EDG.NIMPL], "encode_spec": "energy"}
        layout["meta"] = {"graph_type": GRF.FULL, "edge_type_filter": [EDG.AND], "encode_spec": "energy"}
        layout["connec"] = 'sense-->connectedpart,connectedpart-->meta,meta-->out'
    elif name == "clevrpos1":
        layout["tier1"] = {"graph_type": GRF.FULL, "edge_type_filter": [EDG.AND], "encode_spec": "energy"}
        layout["connec"] = 'sense-->tier1,tier1-->out'
    elif name == "clevrpos2":
        layout["tier1"] = {"graph_type": GRF.FULL, "edge_type_filter": [EDG.NCONV, EDG.NIMPL], "encode_spec": "energy"}
        layout["meta"] = {"graph_type": GRF.FULL, "edge_type_filter": [EDG.AND], "encode_spec": "energy"}
        layout["connec"] = 'sense-->tier1,tier1-->meta,meta-->out'
    else:
        raise ValueError("unexpected name")
    
    # parse fields
    if "comment" in layout:
        del layout["comment"] # remove a comment if any
    fn = layout.keys()
    for i in fn:
        if "graph_type" in layout[i]:
            layout[i]["graph_type"] = GRF(layout[i]["graph_type"]) # convert to enum
        if "edge_type_filter" in layout[i]:
            layout[i]["edge_type_filter"] = EDGlist(layout[i]["edge_type_filter"]) # convert to enums
    
    # validate
    # layout.field is a string or a struct with fields:
    #   .graph_type       - scalar (GRF enum) type of graph used by this component bank
    #   .edge_type_filter - vector (EDG enum) empty = no filtering
    #   .encode_spec      - (char) type of encoding employed by this component bank; see Encode() for options
    assert isinstance(layout, dict) and layout, "layout must be a non-empty scalar struct"
    fn = layout.keys()
    for id in fn:
        assert i.lower() != "sense" and i.lower() != "label" # reserved
        assert isinstance(layout[i], dict) or isinstance(layout[i], str), "layout field must be a non-empty struct or string"
        if i.lower() == "connec":
            assert isinstance(layout[i], str) and layout[i], "connec field must be a non-empty string"
        else:
            assert isinstance(layout[i]["graph_type"], GRF) and layout[i]["graph_type"] > 0 and isinstance(layout[i]["graph_type"], int), "graph_type field must be a non-empty positive integer"
            assert isinstance(layout[i]["edge_type_filter"], list), "edge_type_filter field must be a list"
            assert isinstance(layout[i]["encode_spec"], str) and layout[i]["encode_spec"], "encode_spec field must be a non-empty string"
    
    return layout


