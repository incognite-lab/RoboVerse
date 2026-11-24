"""Metric ruler rendered as collision-free rigid geometry in the Nyx scene."""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class RulerSettings:
    enabled: bool = False
    size_m: tuple = (1.,.02,.02)
    position: tuple = (-2.7,0.,1.)  # Centre of endpoint A, in Genesis Z-up metres.
    rotation_deg: tuple = (0.,0.,0.)
    reference_length_m: float | None = None
    max_length_m: float = 10.
    move_speed: float = .25
    turn_speed: float = 45.
    length_step_m: float = .01
    light_intensity: float = 3.
    report_path: str = './config_run/output/nyx_calibration/measurement.json'

    def __post_init__(self):
        if not isinstance(self.enabled,bool):
            raise ValueError('nyx_ruler.enabled must be true or false')
        for field in ('size_m','position','rotation_deg'):
            value = np.asarray(getattr(self,field),dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f'nyx_ruler.{field} must contain three finite numbers')
            setattr(self,field,tuple(value))
        self.reference_length_m = float(self.size_m[0] if self.reference_length_m is None else self.reference_length_m)
        for field in ('max_length_m','move_speed','turn_speed','length_step_m','reference_length_m'):
            value = float(getattr(self,field))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'nyx_ruler.{field} must be finite and positive')
            setattr(self,field,value)
        if min(self.size_m) <= 0 or not .001 <= self.size_m[0] <= self.max_length_m <= 20:
            raise ValueError('Ruler needs positive dimensions and .001 <= length <= max_length_m <= 20')
        for length in (self.size_m[0],self.max_length_m,self.length_step_m):
            if not math.isclose(length*1000,round(length*1000),abs_tol=1e-6):
                raise ValueError('Ruler lengths/steps must be multiples of .001 m')
        if not math.isfinite(self.light_intensity) or self.light_intensity < 0:
            raise ValueError('Ruler light_intensity must be finite and nonnegative')
        if not isinstance(self.report_path,str) or not self.report_path.strip():
            raise ValueError('Ruler report_path must be a nonempty path')


class MeasuringRuler:
    def __init__(self, settings, parts=(), markers=()):
        self.settings = settings
        self.parts,self.markers = list(parts),list(markers)
        self.position = np.array(settings.position,dtype=float)
        self.rotation = Rotation.from_euler('xyz',settings.rotation_deg,degrees=True)
        self.length_mm = round(settings.size_m[0]*1000)
        self.reference_length_m = settings.reference_length_m
        self.visible = True
        self._previous = {}

    @property
    def length_m(self):
        return self.length_mm/1000

    @property
    def endpoint_b(self):
        return self.position+self.rotation.apply([self.length_m,0.,0.])

    def layout(self):
        """Use contiguous decimal-length boxes: resize without reloading Nyx.

        Rigid mesh sizes are static in the renderer. A bounded set of boxes
        forms exactly one bar to 1 mm precision; unused boxes sit below ground.
        """
        remaining,cursor = self.length_mm,0.
        hidden = np.array([0.,0.,-100.])
        poses = []
        for ticks,_ in self.parts:
            if self.visible and remaining >= ticks:
                pos = self.position+self.rotation.apply([cursor+ticks/2000,0.,0.])
                remaining -= ticks
                cursor += ticks/1000
            else:
                pos = hidden
            poses.append(pos)
        if self.parts and self.visible and remaining:
            raise ValueError('Ruler geometry cannot represent the requested length')
        return poses

    def apply(self):
        xyzw = self.rotation.as_quat()
        quat = tuple(xyzw[[3,0,1,2]])
        updates = []
        poses = self.layout()
        for (_,entity),pos in zip(self.parts,poses):
            updates.append((entity,tuple(pos),quat))
        for entity,pos in zip(self.markers,(self.position,self.endpoint_b)):
            updates.append((entity,tuple(pos if self.visible else (0.,0.,-100.)),(1.,0.,0.,0.)))
        changed = [(entity,pos,quat) for entity,pos,quat in updates
                   if self._previous.get(id(entity)) != (pos,quat)]
        for index,(entity,pos,quat) in enumerate(changed):
            old = self._previous.get(id(entity))
            if old is None or old[1] != quat:
                entity.set_quat(quat,relative=False,zero_velocity=False,skip_forward=True)
            entity.set_pos(pos,relative=False,zero_velocity=False,skip_forward=index != len(changed)-1)
            self._previous[id(entity)] = (pos,quat)

    def update(self, held, pressed, dt, camera):
        dt = min(max(float(dt),0.),.1)
        fine = .1 if 'control' in held else 1.
        movement = np.array([('kp_6' in held)-('kp_4' in held),
                             ('kp_8' in held)-('kp_2' in held),
                             ('kp_9' in held)-('kp_3' in held)],dtype=float)
        norm = np.linalg.norm(movement)
        if norm:
            self.position += movement/norm*self.settings.move_speed*dt*fine
        angle = math.radians(self.settings.turn_speed)*dt*fine
        turn = np.array([0.,('kp_divide' in held)-('kp_multiply' in held),
                         ('kp_7' in held)-('kp_1' in held)])*angle
        if np.any(turn):
            self.rotation = Rotation.from_rotvec(turn)*self.rotation
        step = max(1,round(self.settings.length_step_m*1000*fine))
        self.length_mm = int(np.clip(self.length_mm+step*(('kp_add' in pressed)-('kp_subtract' in pressed)),
                                     1,round(self.settings.max_length_m*1000)))
        if 'kp_decimal' in pressed:
            self.visible = not self.visible
        if 'kp_0' in pressed:
            self.position = np.array(self.settings.position,dtype=float)
            self.rotation = Rotation.from_euler('xyz',self.settings.rotation_deg,degrees=True)
            self.length_mm = round(self.settings.size_m[0]*1000)
            self.reference_length_m = self.settings.reference_length_m
            self.visible = True
        if 'kp_enter' in pressed:
            self.reference_length_m = self.length_m
            print(f'Ruler: stored real reference length = {self.reference_length_m:.3f} m',flush=True)
        self.apply()
        if 'kp_5' in pressed:
            self.save(camera)

    def measurement(self, camera):
        factor = self.reference_length_m/self.length_m
        scale = np.array(camera.gaussian_scale,dtype=float)
        # Preserve the splat point under endpoint A when changing uniform scale.
        anchor = self.position if camera.gaussian_z_up else self.position[[0,2,1]]*np.array([1,1,-1])
        translation = anchor+factor*(np.array(camera.gaussian_position)-anchor)
        return dict(length_m=self.length_m,size_m=[self.length_m,*self.settings.size_m[1:]],
                    reference_length_m=self.reference_length_m,endpoint_a=self.position.tolist(),
                    endpoint_b=self.endpoint_b.tolist(),rotation_xyzw=self.rotation.as_quat().tolist(),
                    length_resolution_m=.001,scale_correction=factor,
                    current_gaussian_scale=scale.tolist(),recommended_gaussian_scale=(scale*factor).tolist(),
                    recommended_gaussian_position=translation.tolist(),
                    note='Position recommendation keeps A fixed. Check depth from multiple views; restart after editing splat YAML.')

    def save(self, camera):
        result = self.measurement(camera)
        path = Path(self.settings.report_path)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary = path.with_name(path.name+'.tmp')
        temporary.write_text(json.dumps(result,indent=2),encoding='utf-8')
        temporary.replace(path)
        print('RULER MEASUREMENT:',json.dumps(result,indent=2),flush=True)
        print(f'gaussian_scale: {result["recommended_gaussian_scale"]}\n'
              f'gaussian_position: {result["recommended_gaussian_position"]}\nSaved {path}',flush=True)
        return result

    @property
    def caption(self):
        return (f'Ruler L={self.length_m:.3f}m ref={self.reference_length_m:.3f}m '
                f'k={self.reference_length_m/self.length_m:.5f}'+(' [hidden]' if not self.visible else ''))


def build_ruler(scene, config):
    settings = RulerSettings(**config)
    if not settings.enabled:
        return None
    import genesis as gs
    parts = []
    for ticks,count in [(1000,math.ceil(settings.max_length_m)),(100,9),(10,9),(1,9)]:
        for _ in range(count):
            entity = scene.add_entity(
                gs.morphs.Box(size=(ticks/1000,*settings.size_m[1:]),pos=(0,0,-100),fixed=True,collision=False),
                material=gs.materials.Rigid(gravity_compensation=1.),
                surface=gs.surfaces.Default(color=(1.,.75,.03)))
            parts.append((ticks,entity))
    markers = [scene.add_entity(
        gs.morphs.Sphere(radius=.018,pos=(0,0,-100),fixed=True,collision=False),
        material=gs.materials.Rigid(gravity_compensation=1.),
        surface=gs.surfaces.Default(color=color))
        for color in ((1.,.05,.05),(.05,.25,1.))]
    return MeasuringRuler(settings,parts,markers)
