"""Interactive Nyx calibration with optional policy playback and physics pause."""
import math
import time

from loguru import logger as log

from metasim.cfg.sensors import NyxGaussianSplatCameraCfg

try:
    from .visual_augmentation import VisualAugmentation, DEFAULT_SEED
    from .student_vision import preprocess_student_rgb
    from .nyx_preview_policy import PreviewPolicySettings, PreviewPolicy
except ImportError:
    from visual_augmentation import VisualAugmentation, DEFAULT_SEED
    from student_vision import preprocess_student_rgb
    from nyx_preview_policy import PreviewPolicySettings, PreviewPolicy


def robot_camera_from_config(config):
    # Read the training camera, rather than keeping a second copy of its mount.
    try:
        from .main_multi import load_config_from_yaml, get_cameras_from_config
    except ImportError:
        from main_multi import load_config_from_yaml, get_cameras_from_config
    source = config.get('nyx_preview_robot_camera_config','chairman_separate/train_dagger')
    name = config.get('nyx_preview_robot_camera_name','camera0')
    settings = load_config_from_yaml(source)
    cameras = get_cameras_from_config(settings.get('cameras',{}))
    camera = next((camera for camera in cameras if camera.name == name),None)
    if not isinstance(camera,NyxGaussianSplatCameraCfg) or not camera.mount_to or not camera.mount_link:
        raise ValueError(f'{source} must define an attached Nyx robot camera named {name}')
    return camera


def preview_nyx(config, base, env=None):
    handler = base.env.handler
    cameras = [camera for camera in handler.cameras if isinstance(camera,NyxGaussianSplatCameraCfg)]
    if not cameras:
        raise ValueError('preview_nyx requires a NyxGaussianSplatCameraCfg camera')
    fps = float(config.get('nyx_preview_fps',30))
    frames = int(config.get('nyx_preview_max_frames',0))
    if not math.isfinite(fps) or fps <= 0 or frames < 0:
        raise ValueError('nyx_preview_fps must be positive; nyx_preview_max_frames must be nonnegative')
    controls = bool(config.get('nyx_preview_controls',False))
    if not controls and not any(camera.open_nyx_window or camera.debug_show or camera.debug_calibrate for camera in cameras):
        raise ValueError('Enable open_nyx_window or debug_show for a visible Nyx preview')
    settings = PreviewPolicySettings.from_config(config)
    if settings.enabled and (env is None or env is base):
        raise ValueError('Policy playback requires the wrapped environment from build_env(config)')
    playback = PreviewPolicy(settings,base,env) if settings.enabled else None
    paused = settings.start_paused
    if playback is None:
        base.reset()
    ruler = getattr(handler,'_nyx_ruler',None)
    if ruler is not None:
        ruler.apply()
        log.info('Ruler: Num4/6 X, Num2/8 Y, Num3/9 Z; Num7/1 yaw, Num/ and Num* pitch; '
                 'Num+/- length, Ctrl finer; NumEnter stores real reference length; '
                 'Num5 prints/saves measurement; Num0 resets, NumDecimal hides/shows. '
                 'Endpoint centres A (red) and B (blue) define the measured length.')
    window = controller = None
    augmentation = None
    augmentation_parameters = None
    augmentation_visible = False
    if config.get('nyx_preview_augmentation',False):
        augmentation = VisualAugmentation(config.get('nyx_preview_augmentation_seed',DEFAULT_SEED))
        augmentation_size = tuple(config.get('nyx_preview_augmentation_size',[640,360]))
        if len(augmentation_size)!=2 or any(not isinstance(x,int) or x<32 for x in augmentation_size):
            raise ValueError('nyx_preview_augmentation_size must be [width,height], each >=32')
    if controls:
        try:
            from .nyx_preview_controls import PreviewCamera, PreviewWindow
        except ImportError:
            from nyx_preview_controls import PreviewCamera, PreviewWindow
        camera_name = config.get('nyx_preview_camera_name',cameras[0].name)
        camera = next((camera for camera in cameras if camera.name == camera_name),None)
        if camera is None:
            raise ValueError(f'Preview camera {camera_name!r} does not exist')
        controller = PreviewCamera(handler,camera,robot_camera_from_config(config),
            speed=config.get('nyx_preview_move_speed',1.),turn_speed=config.get('nyx_preview_turn_speed',75.))
        if playback is not None and playback.adapter is not None and playback.adapter.camera_name != camera_name:
            raise ValueError('Student checkpoint camera_name must match nyx_preview_camera_name')
        window = PreviewWindow(camera.width,camera.height)
        log.info('Controls: WASD move, QE yaw, RF height, arrows pitch/yaw, Shift faster; '
                 'C/Tab toggle free/robot, 1 free, 2 robot, H reset free, P print pose, Esc quit. '
                 'Click the calibration window to focus keyboard input.')
        if playback is not None:
            log.info('Space toggles physics + policy pause; camera controls stay active. '
                     'One running preview frame advances one control step; slow rendering slows playback.')
        if augmentation is not None:
            log.info('G samples a new augmented view; V toggles clean/augmented. '
                     'Preview samples the augmented half of the training mixture; sensor noise varies per frame.')
    log.info('Nyx preview: {}; Ctrl+C stops the preview. '
             'Edit gaussian_position / gaussian_rotation_xyzw / gaussian_scale in YAML and restart '
             'to update splat placement. The image is rendered by Nyx.',
             'policy playback with Space pause' if playback is not None else 'geometry is stationary')
    count = 0
    previous = time.monotonic()
    try:
        while frames == 0 or count < frames:
            started = time.monotonic()
            if window is not None:
                held,pressed = window.poll()
                if window.closed:
                    break
                if playback is not None and 'space' in pressed:
                    paused = not paused
                    log.info('Nyx policy playback {}', 'PAUSED' if paused else 'RUNNING')
                controller.update(held,pressed,started-previous)
                if ruler is not None:
                    ruler.update(held,pressed,started-previous,camera)
                if augmentation is not None:
                    if 'g' in pressed or ('v' in pressed and augmentation_parameters is None):
                        augmentation_parameters = augmentation.sample(1,force=True)
                        log.info('Preview augmentation: {}',augmentation_parameters.describe())
                    if 'g' in pressed:
                        augmentation_visible = True
                    elif 'v' in pressed:
                        augmentation_visible = not augmentation_visible
            previous = started
            if playback is not None and not paused:
                playback.step(controller)
                # Follow the new head pose after physics (including episode reset).
                controller.apply()
            # During pause/static preview, only re-render: no step, resets, or RNN updates.
            handler._camera_states_cache = None
            states = handler.get_camera_states()
            if window is not None:
                rgb = states[camera_name].rgb
                caption = controller.mode
                if playback is not None:
                    caption += ' | '+playback.caption(paused)
                if augmentation is not None:
                    caption += ' | Aug ON (G/V)' if augmentation_visible else ' | Aug OFF (G/V)'
                if augmentation_visible:
                    images = preprocess_student_rgb(rgb,device=rgb.device,
                        output_size=(augmentation_size[1],augmentation_size[0])).float().div_(255).contiguous()
                    images = augmentation.apply(images,augmentation_parameters,inplace=True)
                    rgb = images.mul_(255).round_().to(dtype=rgb.dtype).permute(0,2,3,1).contiguous()
                    caption += f' {augmentation_size[0]}x{augmentation_size[1]}'
                caption += ' | '+ruler.caption if ruler is not None else ''
                window.show(rgb,caption)
            count += 1
            delay = 1/fps-(time.monotonic()-started)
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        log.info('Nyx preview stopped')
    finally:
        if window is not None:
            window.close()
    return count
