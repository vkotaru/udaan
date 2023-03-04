import numpy as np
from scipy.linalg import expm
import time
import enum

from ..mujoco import MujocoModel
from ..base import BaseModel
from ... import utils


class QuadrotorCSPayload(base.QuadrotorCSPayload):

    class MODEL_TYPE(enum.Enum):
        TENDON = 0  # Using https://mujoco.readthedocs.io/en/stable/XMLreference.html#default-tendon

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # mujoco model param handling
        self._mjMdl = None
        self._mj_quad_index = None
        self._mj_payload_index = None
        self._mj_cable_index = None
