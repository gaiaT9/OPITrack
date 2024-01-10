from models.BranchedERFNet import *
from models.pointtrack import *
from models.pointtrackpp import *
import models.pointtrack_pred as ppred
import models.pointtrack_strip as pstrip
import models.pointtrack_scsn as pscsn
import models.pointtrack_deep as pdeep
import models.pointtrack_transformer as ptrans
import models.pointtrack_pred_dist as ppred_dist
import models.pointtrack_pred_weight as ppred_weight
import models.pointtrackpp_pred as pp_pred
import models.pointtrackpp_pred_weight as pp_pred_weight


def get_model(name, model_opts):
    if name == "branched_erfnet":
        model = BranchedERFNet(**model_opts)
        return model
    if name == "tracker_offset_emb":
        model = TrackerOffsetEmb(**model_opts)
        return model
    if name == "tracker_offset_emb_pp":
        model = TrackerOffsetEmbPP(**model_opts)
        return model
    if name == "tracker_offset_emb_pp_pred":
        model = pp_pred.TrackerOffsetEmbPP(**model_opts)
        return model
    if name == "tracker_offset_emb_pp_pred_weight":
        model = pp_pred_weight.TrackerOffsetEmbPP(**model_opts)
        return model
    if name == "tracker_offset_emb_pred":
        model = ppred.TrackerOffsetEmb(**model_opts)
        return model
    if name == "tracker_offset_emb_pred_dist":
        model = ppred_dist.TrackerOffsetEmb(**model_opts)
        return model
    if name == "tracker_offset_emb_pred_weight":
        model = ppred_weight.TrackerOffsetEmb(**model_opts)
        return model
    if name == "tracker_offset_emb_strip":
        model = pstrip.TrackerOffsetEmb(**model_opts)
        return model
    if name == "tracker_offset_emb_scsn":
        model = pscsn.TrackerOffsetEmb(**model_opts)
        return model
    if name == "tracker_offset_emb_deep":
        model = pdeep.TrackerOffsetEmb(**model_opts)
        return model
    if name == "tracker_offset_emb_trans":
        model = ptrans.TrackerOffsetEmb(**model_opts)
        return model
    else:
        raise RuntimeError("model \"{}\" not available".format(name))