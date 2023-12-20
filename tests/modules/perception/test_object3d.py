# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-05-11
# @Last Modified by:   spencer@primus
# @Last Modified date: 2022-09-08
# @Description:
"""

"""

import sys
from copy import deepcopy

import numpy as np

from avstack import GroundTruthInformation
from avstack.datastructs import DataContainer
from avstack.modules import perception


sys.path.append("tests/")
from utilities import get_ego, get_object_global, get_test_sensor_data


(
    obj,
    box_calib,
    lidar_calib,
    pc,
    camera_calib,
    img,
    radar_calib,
    rad,
    box_2d,
    box_3d,
) = get_test_sensor_data()
frame = 0


def test_groundtruth_perception():
    # -- set up ego and objects
    ego_init = get_ego(seed=3)
    obj1 = get_object_global(seed=4)
    obj_local = deepcopy(obj1)
    obj_local.change_reference(ego_init, inplace=True)
    assert ego_init.reference == obj1.reference
    assert obj1.reference != obj_local.reference
    assert not np.allclose(obj1.position.x, obj_local.position.x)

    # GT information
    frame = timestamp = 0
    ground_truth = GroundTruthInformation(
        frame, timestamp, ego_state=ego_init, objects=[obj1]
    )

    # -- test update
    percep = perception.object3d.GroundTruth3DObjectDetector()
    detections = percep(ground_truth, frame=frame)
    assert np.allclose(detections[0].box.t.x, obj_local.position.x)


def test_passthrough_perception_box():
    frame = timestamp = 0
    objs = [get_object_global(seed=i) for i in range(4)]
    data = DataContainer(
        frame=frame, timestamp=timestamp, data=objs, source_identifier="sensor-1"
    )
    percep = perception.object3d.Passthrough3DObjectDetector()
    detections = percep(data, frame=frame)
    assert len(detections) == len(data)


def test_passthrough_perception_centroid():
    frame = timestamp = 0
    objs = [get_object_global(seed=i).position for i in range(4)]
    data = DataContainer(
        frame=frame, timestamp=timestamp, data=objs, source_identifier="sensor-1"
    )
    percep = perception.object3d.Passthrough3DObjectDetector()
    detections = percep(data, frame=frame)
    assert len(detections) == len(data)


class LidarMeasurement:
    """To emulate the carla measurements"""

    def __init__(self, raw_data: memoryview) -> None:
        assert isinstance(raw_data, memoryview)
        self.raw_data = raw_data


def run_mmdet3d(datatype, model, dataset, as_memoryview=False):
    try:
        detector = perception.object3d.MMDetObjectDetector3D(
            model=model, dataset=dataset
        )
    except ModuleNotFoundError:
        print("Cannot run mmdet test without the module")
    except FileNotFoundError:
        print(f"Cannot find ({model}, {dataset}) model file for mmdet3d test")
    else:
        if datatype == "lidar":
            data = pc
            if as_memoryview:
                data.data = LidarMeasurement(memoryview(data.data.x))
        elif datatype == "image":
            data = img
        else:
            raise NotImplementedError(datatype)
        _ = detector(data, frame=frame)


# def test_mmdet_3d_pgd_kitti():
#     run_mmdet3d("image", "pgd", "kitti")


# def test_mmdet_3d_pillars_kitti():
#     run_mmdet3d("lidar", "pointpillars", "kitti")


# def test_mmdet_3d_pillars_kitti_memoryview():
#     run_mmdet3d("lidar", "pointpillars", "kitti", as_memoryview=True)
