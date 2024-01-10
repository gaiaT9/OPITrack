from datasets.kitti_dataset import *
import datasets.kitti_dataset.KittiMOTSDataset_with_img as img_ds

dataset_dict = {
    "mots_test": MOTSTest,
    "mots_cars_val": MOTSCarsVal,
    "mots_person_val": MOTSPersonVal,
    "mots_track_val_env_offset": MOTSTrackCarsValOffset,
    "mots_track_val_env_offset_img": img_ds.MOTSTrackCarsValOffset,
    "mots_track_val_env_offset_person": MOTSTrackPersonValOffset,
    "mots_track_cars_train": MOTSTrackCarsTrain,
    "mots_track_person_train": MOTSTrackPersonTrain,
    "mots_cars": MOTSCars,
    "mots_person": MOTSPerson,
    # test part,
    "mots_track_car_train_seq": MOTSTrackCarsTrainSeq,
    "mots_track_car_train_seq_DHN": MOTSTrackCarsTrainSeqDHN,
    "mots_track_car_train_seq_strip": MOTSTrackCarsTrainStrip,
    "mots_track_val_env_offset_strip": MOTSTrackCarsValOffsetStrip,
    'mots_track_car_train_seq_weight': MOTSTrackCarsTrainWeight,
    # person
    "mots_track_person_train_seq": MOTSTrackPersonTrainSeq,
    "mots_track_person_train_seq_weight": MOTSTrackPersonTrainWeight,
    # With TID data
    "mots_track_car_train_seq_tid": MOTSTrackCarsTrainTID,
    # mot challenge data
    "mots_challenge_track_person_train": MOTSChallengeTrackPersonTrain,
    "mots_challenge_track_person_val": MOTSChallengeTrackPersonVal,
    "mots_challenge_track_person_train_seq": MOTSChallengeTrackPersonTrainSeq,
    "mots_challenge_track_person_train_seq_weight": MOTSChallengeTrackPersonTrainWeight,
    # visualization
    "mots_track_val_env_offset_vis": MOTSTrackCarsValOffsetVis,
    "mots_track_val_env_offset_person_vis": MOTSTrackPersonValOffsetVis,
}


def get_dataset(name, dataset_opts):
    # if name == "mots_test":
    #     return MOTSTest(**dataset_opts)
    # elif name == "mots_cars_val":
    #     return MOTSCarsVal(**dataset_opts)
    # elif name == "mots_person_val":
    #     return MOTSPersonVal(**dataset_opts)
    # elif name == "mots_track_val_env_offset":
    #     return MOTSTrackCarsValOffset(**dataset_opts)
    # elif name == "mots_track_val_env_offset_img":
    #     return img_ds.MOTSTrackCarsValOffset(**dataset_opts)
    # elif name == "mots_track_val_env_offset_person":
    #     return MOTSTrackPersonValOffset(**dataset_opts)
    # elif name == "mots_track_cars_train":
    #     return MOTSTrackCarsTrain(**dataset_opts)
    # elif name == "mots_track_person_train":
    #     return MOTSTrackPersonTrain(**dataset_opts)
    # elif name == "mots_cars":
    #     return MOTSCars(**dataset_opts)
    # elif name == "mots_person":
    #     return MOTSPerson(**dataset_opts)
    # else:
    #     raise RuntimeError("Dataset {} not available".format(name))
    if name in dataset_dict.keys():
        return dataset_dict[name](**dataset_opts)
    else:
        raise RuntimeError("Dataset {} not available".format(name))