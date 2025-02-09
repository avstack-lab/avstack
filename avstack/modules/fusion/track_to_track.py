# -*- coding: utf-8 -*-
# @Author: Spencer H
# @Date:   2022-07-27
# @Last Modified by:   Spencer H
# @Last Modified date: 2022-09-24
# @Description:
"""

"""

from typing import Any, List, Union

import numpy as np

from avstack.config import MODELS
from avstack.datastructs import DataContainer
from avstack.geometry import Box3D, Position, Velocity
from avstack.modules.tracking.tracker3d import BasicBoxTrack3D
from avstack.modules.tracking.tracks import TrackBase
from avstack.utils.decorators import apply_hooks

from ..base import BaseModule
from ..clustering.clusterers import Cluster


def ci_fusion(
    x: List[np.ndarray],
    P: List[np.ndarray],
    weights: Union[str, np.ndarray] = "uniform",
):
    """Covariance intersection fusion between filter states
    useful if the cross-correlation between the two data elements is not known

    Cross-correlation can be due to various factors such as common platform
    motion, common occlusion scenarios, etc.

    NOTE: only allows for full state fusion right now
    """
    if (not isinstance(x, list)) or (not isinstance(P, list)):
        raise TypeError(
            "Input state and covariance must be list of states and list of covariances"
        )

    if len(x) == 1:
        return x[0], P[0]  # no fusion to be done for one

    if weights == "uniform":
        ws = 1 / len(x) * np.ones((len(x)))
    elif isinstance(weights, (list, np.ndarray)):
        ws = np.asarray(weights) / np.sum(weights)
    else:
        raise NotImplementedError(weights)

    P_invs = [np.linalg.inv(P_) for P_ in P]
    P_f = np.linalg.inv(sum([w * P_inv for w, P_inv in zip(ws, P_invs)]))
    x_f = P_f @ (sum([w * P_inv @ x_ for w, P_inv, x_ in zip(ws, P_invs, x)]))

    return x_f, P_f


class _BaseFusion(BaseModule):
    def __init__(self, name="fusion", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)


@MODELS.register_module()
class NoFusion(_BaseFusion):
    """Only returns the first set of tracks"""

    @apply_hooks
    def __call__(self, *args: Any, **kwds: Any) -> list:
        tracks_out = [] if len(args) == 0 else args[0]
        if isinstance(tracks_out, (DataContainer, list)):
            pass
        elif isinstance(tracks_out, dict):
            tracks_out = (
                list(tracks_out.values())[0]
                if len(tracks_out) == 1
                else list(tracks_out.values())
            )
        elif isinstance(tracks_out, TrackBase):
            tracks_out = [tracks_out]
        elif isinstance(tracks_out, Cluster):
            pass
        else:
            raise NotImplementedError(type(tracks_out))
        return tracks_out


@MODELS.register_module()
class AggregatorFusion(_BaseFusion):
    """Simply appends all tracks together not worrying about duplicates"""

    @apply_hooks
    def __call__(self, *args: Any, **kwds: Any) -> list:
        tracks_out = []
        for arg in args:
            if isinstance(arg, list):
                tracks_out += arg
            elif isinstance(arg, dict):
                for v in arg.values():
                    tracks_out += v
            elif isinstance(arg, TrackBase):
                tracks_out += [arg]
            else:
                raise NotImplementedError(type(arg))
        return tracks_out


@MODELS.register_module()
class CovarianceIntersectionFusion(_BaseFusion):
    """Covariance intersection to build a track from a cluster"""

    @apply_hooks
    def __call__(
        self,
        tracks: Union[Cluster, List[TrackBase]],
        weights: Union[np.ndarray, str] = "uniform",
    ):
        x_fuse = None
        P_fuse = None

        if len(tracks) > 0:
            # perform fusion on the array
            xs = [track.x for track in tracks]
            Ps = [track.P for track in tracks]
            x_fuse, P_fuse = ci_fusion(xs, Ps, weights=weights)

        return x_fuse, P_fuse


@MODELS.register_module()
class CovarianceIntersectionFusionToBox(_BaseFusion):
    """Performs CI fusion for box tracks and outputs a track"""

    @apply_hooks
    def __call__(
        self,
        tracks: Union[Cluster, List[BasicBoxTrack3D]],
        weights: Union[np.ndarray, str] = "uniform",
        force_ID: bool = True,
    ) -> BasicBoxTrack3D:
        return self.fuse(tracks=tracks, weights=weights, force_ID=force_ID)

    @staticmethod
    def fuse(
        tracks: Union[Cluster, List[BasicBoxTrack3D]],
        weights: Union[np.ndarray, str] = "uniform",
        force_ID: bool = True,
    ):
        """Assume that inputs are box tracks

        Therefore, the state vector is:
        [x, y, z, h, w, l, vx, vy, vz]

        The attitude is NOT being fused here which is an approximation
        """
        if len(tracks) > 0:
            # perform fusion on the array
            xs = [track.x for track in tracks]
            Ps = [track.P for track in tracks]
            x_fuse, P_fuse = ci_fusion(xs, Ps, weights=weights)

            # get other attributes
            t0 = min([track.t0 for track in tracks])
            t = max([track.t for track in tracks])
            reference = tracks[0].reference
            obj_type = tracks[0].obj_type
            dt_coast = min([track.dt_coast for track in tracks])
            n_updates = max([track.n_updates for track in tracks])
            attitude = tracks[0].attitude  # APPROXIMATION

            # wrap into expected attributes
            position = Position(x_fuse[0:3], reference=reference)
            hwl = x_fuse[3:6]
            box3d = Box3D(position, attitude, hwl)
            v = Velocity(x_fuse[6:9], reference=reference)

            # take the minimum ID of the tracks to maintain consistent ID
            ID = min([track.ID for track in tracks]) if force_ID else None

            return BasicBoxTrack3D(
                t0=t0,
                box3d=box3d,
                reference=reference,
                obj_type=obj_type,
                v=v,
                P=P_fuse,
                t=t,
                ID_force=ID,
                dt_coast=dt_coast,
                n_updates=n_updates,
            )
        else:
            return None
