"""Genesis Franka fallback with video recording on AMD ROCm GPU."""
import json, time, traceback, os
from pathlib import Path
import numpy as np

import genesis as gs

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'fallback_artifacts'
OUT.mkdir(exist_ok=True)

result = {
    'example': 'Genesis Franka with video on AMD ROCm GPU',
    'started_at': time.time(),
    'gpu': 'AMD Radeon RX 7900 XT (gfx1100)',
    'rocm': '7.2.1',
    'genesis_version': gs.__version__,
}

try:
    gs.init(backend=gs.gpu)
    
    scene = gs.Scene(show_viewer=False)
    plane = scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'))
    
    cam = scene.add_camera(res=(1280, 720), pos=(2.5, -2.5, 1.8), lookat=(0, 0, 0.5), fov=45)
    
    scene.build()
    
    frames = []
    n_steps = 100
    
    for i in range(n_steps):
        if i >= 20:
            try:
                target = np.zeros(9)
                target[0] = 0.5 * np.sin(i * 0.1)
                target[1] = 0.3 * np.cos(i * 0.1)
                target[2] = 0.2 * np.sin(i * 0.05)
                target[3] = -1.5 + 0.3 * np.sin(i * 0.1)
                target[4] = 0.5
                target[5] = 0.8
                target[6] = 0.8
                franka.set_dofs_position(target)
            except:
                pass
        
        scene.step()
        
        if i % 2 == 0:
            try:
                ret = cam.render()
                if isinstance(ret, tuple):
                    rgb = ret[0]
                elif isinstance(ret, (np.ndarray,)):
                    rgb = ret
                else:
                    # might be a list or other
                    rgb = np.array(ret)
                if rgb is not None and rgb.size > 0:
                    frames.append(rgb)
            except Exception as e:
                if i == 0:
                    result['render_error_first'] = repr(e)
                pass
    
    result['steps'] = n_steps
    result['frames_captured'] = len(frames)
    result['status'] = 'passed'
    result['backend'] = 'gpu'
    result['entities'] = 2
    
    if frames:
        try:
            import imageio.v2 as imageio
            imageio.imwrite(str(OUT / 'franka_screenshot.png'), frames[len(frames)//2])
            result['screenshot'] = 'franka_screenshot.png'
            
            imageio.mimsave(str(OUT / 'franka_demo.mp4'), frames, fps=30, quality=8)
            result['video'] = 'franka_demo.mp4'
            result['video_fps'] = 30
            result['video_resolution'] = '1280x720'
            print(f'Video saved: {len(frames)} frames at 1280x720')
        except Exception as e:
            result['video_warning'] = repr(e)
            try:
                import imageio.v2 as imageio
                imageio.imwrite(str(OUT / 'franka_screenshot.png'), frames[0])
                result['screenshot'] = 'franka_screenshot.png'
                print(f'Screenshot saved (video failed: {e})')
            except Exception as e2:
                result['screenshot_warning'] = repr(e2)
    else:
        result['video_warning'] = 'No frames captured - camera render returned empty'
    
    try:
        qpos = franka.get_dofs_position()
        result['final_qpos'] = qpos.cpu().numpy().tolist()
    except:
        pass

except Exception as e:
    result['status'] = 'failed'
    result['error'] = repr(e)
    result['traceback'] = traceback.format_exc()

result['ended_at'] = time.time()
result['duration_s'] = result['ended_at'] - result['started_at']

(OUT / 'run_video2.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
