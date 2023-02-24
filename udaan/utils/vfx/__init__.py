
import numpy as np
import vpython as vp
import signal
import threading
import sys


class VFXHandler(object):
    def __init__(self, title='visuals', rate=200):
        self._title = 'floating_models::'+title
        self.scene = vp.canvas(title=self._title,
                            width=1080, # 1920
                            height=640, # 1080
                            center=vp.vector(1, 2, 0),
                            background=vp.color.white,
                            isnotebook=8080)

        # self.scene.caption = """To rotate "camera", drag with right button or Ctrl-drag.
        #         To zoom, drag with middle button or Alt/Option depressed, or use scroll wheel.
        #           On a two-button mouse, middle is left + right.
        #         To pan left/right and up/down, Shift-drag.
        #         Touch screen: pinch/extend to zoom, swipe or two-finger rotate."""
        self.scene.camera.rotate(angle=np.pi/2, axis=vp.vector(1,0,0))

        self._models = {}

        self.axes = None
        self.create_env()
        self._rate = rate

        self.kill_loop = False

        # signal.signal(signal.SIGINT, self.handler)
        # self.thread = threading.Thread(target=self.run, args=())
        # self.thread.start()
        # self.thread.join()
        pass

    def handler(self, signal, frame):
        print("Ctrl-C.... Exiting")
        self.kill_loop = True
        sys.exit(0)

    def create_env(self):
        self.axes = []
        self.axes.append(
            vp.arrow(pos=vp.vector(0, 0, 0),
                  axis=vp.vector(2, 0, 0),
                  color=vp.vector(1, 0, 0),
                  opacity=0.15,
                  shaftwidth=0.01))
        self.axes.append(
            vp.arrow(pos=vp.vector(0, 0, 0),
                  axis=vp.vector(0, 2, 0),
                  color=vp.vector(0, 1, 0),
                  opacity=0.15,
                  shaftwidth=0.01))
        self.axes.append(
            vp.arrow(pos=vp.vector(0, 0, 0),
                  axis=vp.vector(0, 0, 2),
                  color=vp.vector(0, 0, 1),
                  opacity=0.15,
                  shaftwidth=0.01))
        return

    def delete_env(self):
        self.axes[0].render = False
        self.axes[0].render = False
        self.axes[0].render = False
        # del self.axes
        # del self.nozzle
        return

    def reset(self):
        self.delete_env()
        self.create_env()

    def run(self):
        while not self.kill_loop:
            vp.rate(self._rate)
            signal.pause()

    def add_model(self, name, model):
        self._models[name] = model
        return

    def del_model(self, name):
        self._models[name].render = False
        del self._models[name]
        return



class Model(object):
    def __init__(self, name):
        self.name = name

    def update(self):
        raise NotImplementedError

    def clear(self):
        pass


class BoundingBox(Model):
    def __init__(self, name='bounding_box', orig=np.zeros(3), xmin=-1.5, xmax=1.5, ymin=-2.5, ymax=2.5, zmin=0.0, zmax=4.0, color='purple'):
        super().__init__(name)
        self.orig = orig
        self.bndry =  vp.curve(color=colors[color],
                               radius=0.01)
        self.points = []
        self.points.append(vp.vector(orig[0]+xmin, orig[1]+ymin, orig[2]+zmin)) 
        self.points.append(vp.vector(orig[0]+xmax, orig[1]+ymin, orig[2]+zmin)) 
        self.points.append(vp.vector(orig[0]+xmax, orig[1]+ymax, orig[2]+zmin)) 
        self.points.append(vp.vector(orig[0]+xmin, orig[1]+ymax, orig[2]+zmin)) 
        self.points.append(vp.vector(orig[0]+xmin, orig[1]+ymin, orig[2]+zmin))

        self.points.append(vp.vector(orig[0]+xmin, orig[1]+ymin, orig[2]+zmax)) 
        self.points.append(vp.vector(orig[0]+xmax, orig[1]+ymin, orig[2]+zmax)) 
        self.points.append(vp.vector(orig[0]+xmax, orig[1]+ymax, orig[2]+zmax)) 
        self.points.append(vp.vector(orig[0]+xmin, orig[1]+ymax, orig[2]+zmax)) 
        self.points.append(vp.vector(orig[0]+xmin, orig[1]+ymin, orig[2]+zmax))

        self.points.append(vp.vector(orig[0]+xmax, orig[1]+ymin, orig[2]+zmax)) 
        self.points.append(vp.vector(orig[0]+xmax, orig[1]+ymin, orig[2]+zmin)) 
        self.points.append(vp.vector(orig[0]+xmax, orig[1]+ymax, orig[2]+zmin)) 
        self.points.append(vp.vector(orig[0]+xmax, orig[1]+ymax, orig[2]+zmax)) 
        self.points.append(vp.vector(orig[0]+xmin, orig[1]+ymax, orig[2]+zmax)) 
        self.points.append(vp.vector(orig[0]+xmin, orig[1]+ymax, orig[2]+zmin)) 
        for p in self.points:
            self.bndry.append(p)
        pass

from .quadrotor_vfx import QuadrotorVFX
from .quadrotor_cspayload_vfx import QuadrotorCSPayloadVFX
