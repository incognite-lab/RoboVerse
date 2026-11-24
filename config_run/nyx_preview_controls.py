"""Free-flight camera and a focused GLFW window for Nyx-rendered RGB."""
import math

import numpy as np
from scipy.spatial.transform import Rotation


class PreviewCamera:
    def __init__(self, handler, camera, robot_camera, *, speed=1., turn_speed=75.):
        self.camera = handler.camera_inst_dict[camera.name]
        self.free_pos = np.array(camera.pos,dtype=float)
        self.free_direction = np.array(camera.look_at,dtype=float)-self.free_pos
        if np.linalg.norm(self.free_direction) < 1e-8:
            raise ValueError('Preview camera pos and look_at must differ')
        self.free_direction /= np.linalg.norm(self.free_direction)
        self.initial = (self.free_pos.copy(),self.free_direction.copy())
        self.speed,self.turn_speed = float(speed),float(turn_speed)
        if not all(math.isfinite(x) and x > 0 for x in (self.speed,self.turn_speed)):
            raise ValueError('Preview move/turn speeds must be finite and positive')
        # One sensor sees the same scene in both modes: no duplicate splat asset.
        entity_name = robot_camera.mount_to
        entity = handler.object_inst_dict.get(entity_name)
        if entity is None:
            raise ValueError(f'Robot camera entity {entity_name!r} is not in the preview scene')
        self.link = next((link for link in entity.links if link.name == robot_camera.mount_link),None)
        if self.link is None:
            raise ValueError(f'Robot camera link {robot_camera.mount_link!r} does not exist')
        for field in ('horizontal_fov','vertical_fov'):
            if not math.isclose(getattr(camera.render_projection,field),
                                getattr(robot_camera.render_projection,field),abs_tol=1e-5):
                raise ValueError('Preview FOV must match the robot camera FOV')
        self.mount_pos = np.array(robot_camera.mount_pos or (0.05,0.,0.),dtype=float)
        w,x,y,z = robot_camera.mount_quat or (1.,0.,0.,0.)
        self.mount_rotation = Rotation.from_quat([x,y,z,w])
        self.robot_view = False
        self.apply()

    @staticmethod
    def _first(value):
        if hasattr(value,'detach'):
            value = value.detach().cpu().numpy()
        array = np.asarray(value,dtype=float)
        return array[0] if array.ndim > 1 else array

    def pose(self):
        if not self.robot_view:
            return self.free_pos,self.free_pos+self.free_direction,np.array([0.,0.,1.])
        position = self._first(self.link.get_pos(envs_idx=[0]))
        w,x,y,z = self._first(self.link.get_quat(envs_idx=[0]))
        rotation = Rotation.from_quat([x,y,z,w])
        position = position+rotation.apply(self.mount_pos)
        rotation = rotation*self.mount_rotation
        return position,position+rotation.apply([0.,0.,-1.]),rotation.apply([0.,1.,0.])

    def apply(self):
        pos,lookat,up = self.pose()
        self.camera.update_camera_pose(pos=tuple(pos),lookat=tuple(lookat),up=tuple(up))

    def update(self, held, pressed, dt):
        if 'c' in pressed or 'tab' in pressed:
            self.robot_view = not self.robot_view
        if '1' in pressed:
            self.robot_view = False
        if '2' in pressed:
            self.robot_view = True
        if 'h' in pressed:
            self.free_pos,self.free_direction = (x.copy() for x in self.initial)
            self.robot_view = False
        dt = min(max(float(dt),0.),.1)  # Never jump after a slow render/build.
        if not self.robot_view:
            yaw = math.atan2(self.free_direction[1],self.free_direction[0])
            pitch = math.asin(float(np.clip(self.free_direction[2],-1.,1.)))
            yaw += math.radians(self.turn_speed)*dt*(('q' in held or 'left' in held)-('e' in held or 'right' in held))
            pitch += math.radians(self.turn_speed)*dt*(('up' in held)-('down' in held))
            pitch = np.clip(pitch,-math.radians(89),math.radians(89))
            self.free_direction = np.array([math.cos(pitch)*math.cos(yaw),
                                           math.cos(pitch)*math.sin(yaw),math.sin(pitch)])
            up = np.array([0.,0.,1.])
            right = np.cross(self.free_direction,up)
            right /= np.linalg.norm(right)
            move = (self.free_direction*(('w' in held)-('s' in held))
                    +right*(('d' in held)-('a' in held))+up*(('r' in held)-('f' in held)))
            length = np.linalg.norm(move)
            if length > 0:
                self.free_pos += move/length*self.speed*dt*(3 if 'shift' in held else 1)
        self.apply()
        if 'p' in pressed:
            pos,lookat,up = self.pose()
            print('Preview camera:',dict(mode=self.mode,pos=pos.tolist(),look_at=lookat.tolist(),up=up.tolist()),flush=True)

    @property
    def mode(self):
        return 'ROBOT (C: free)' if self.robot_view else 'FREE (C: robot)'


class PreviewWindow:
    """Own the displayed camera image and input instead of Nyx's opaque GUI."""
    def __init__(self, width, height):
        import glfw
        from OpenGL import GL
        self.glfw,self.gl = glfw,GL
        self.window,self.texture = None,None
        if not glfw.init():
            raise RuntimeError('Cannot initialize GLFW; open the preview in a desktop/VNC session')
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR,1)
        self.window = glfw.create_window(width,height,'Nyx calibration',None,None)
        if not self.window:
            raise RuntimeError('Cannot open the Nyx calibration window (OpenGL 2.1 required)')
        try:
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)
            self.texture = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D,self.texture)
            GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MIN_FILTER,GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MAG_FILTER,GL.GL_LINEAR)
            self.texture_shape = None
            self.pressed = set()
            self.keys = {name:getattr(glfw,'KEY_'+name.upper()) for name in
                         ('w','s','a','d','q','e','r','f','c','tab','1','2','h','p','g','v','left','right','up','down',
                          'kp_0','kp_1','kp_2','kp_3','kp_4','kp_5','kp_6','kp_7','kp_8','kp_9',
                          'kp_add','kp_subtract','kp_divide','kp_multiply','kp_enter','kp_decimal')}
            reverse = {key:name for name,key in self.keys.items()}
            def on_key(window,key,scancode,action,mods):
                if action == glfw.PRESS or (action == glfw.REPEAT and key in
                                            (glfw.KEY_KP_ADD,glfw.KEY_KP_SUBTRACT)):
                    if key == glfw.KEY_ESCAPE:
                        glfw.set_window_should_close(window,True)
                    elif key in reverse:
                        self.pressed.add(reverse[key])
            glfw.set_key_callback(self.window,on_key)
        except BaseException:
            self.close()
            raise

    def poll(self):
        glfw = self.glfw
        glfw.poll_events()
        pressed,self.pressed = self.pressed,set()
        held = set()
        if glfw.get_window_attrib(self.window,glfw.FOCUSED):
            held = {name for name,key in self.keys.items() if glfw.get_key(self.window,key) == glfw.PRESS}
            if any(glfw.get_key(self.window,key) == glfw.PRESS for key in
                   (glfw.KEY_LEFT_SHIFT,glfw.KEY_RIGHT_SHIFT)):
                held.add('shift')
            if any(glfw.get_key(self.window,key) == glfw.PRESS for key in
                   (glfw.KEY_LEFT_CONTROL,glfw.KEY_RIGHT_CONTROL)):
                held.add('control')
        return held,pressed

    @property
    def closed(self):
        return self.glfw.window_should_close(self.window)

    def show(self, rgb, mode):
        if hasattr(rgb,'detach'):
            rgb = rgb.detach().cpu().numpy()
        if rgb.ndim == 4:
            rgb = rgb[0]
        rgb = np.ascontiguousarray(rgb[...,:3],dtype=np.uint8)
        h,w = rgb.shape[:2]
        gl = self.gl
        self.glfw.make_context_current(self.window)
        self.glfw.set_window_title(self.window,f'Nyx | {mode}')
        gl.glBindTexture(gl.GL_TEXTURE_2D,self.texture)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT,1)
        if self.texture_shape != (h,w):
            gl.glTexImage2D(gl.GL_TEXTURE_2D,0,gl.GL_RGB,w,h,0,gl.GL_RGB,gl.GL_UNSIGNED_BYTE,rgb)
            self.texture_shape = (h,w)
        else:
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D,0,0,0,w,h,gl.GL_RGB,gl.GL_UNSIGNED_BYTE,rgb)
        fw,fh = self.glfw.get_framebuffer_size(self.window)
        gl.glViewport(0,0,fw,fh)
        gl.glClearColor(.04,.04,.04,1.)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        ratio = min(fw/w,fh/h)
        vw,vh = int(w*ratio),int(h*ratio)
        gl.glViewport((fw-vw)//2,(fh-vh)//2,vw,vh)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBegin(gl.GL_QUADS)
        for uv,xy in [((0,1),(-1,-1)),((1,1),(1,-1)),((1,0),(1,1)),((0,0),(-1,1))]:
            gl.glTexCoord2f(*uv)
            gl.glVertex2f(*xy)
        gl.glEnd()
        gl.glDisable(gl.GL_TEXTURE_2D)
        self.glfw.swap_buffers(self.window)

    def close(self):
        if self.window:
            self.glfw.make_context_current(self.window)
            if self.texture is not None:
                self.gl.glDeleteTextures([self.texture])
                self.texture = None
            self.glfw.destroy_window(self.window)
            self.window = None
        # Do not terminate GLFW globally: Genesis/Nyx may still own windows.
