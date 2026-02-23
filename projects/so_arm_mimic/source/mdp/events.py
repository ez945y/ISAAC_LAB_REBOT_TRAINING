import random
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from pxr import UsdGeom, UsdShade, Gf
import colorsys

def randomize_material_color(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    color_ranges: dict = {"r": [0.0, 1.0], "g": [0.0, 1.0], "b": [0.0, 1.0]},
):
    asset = env.scene[asset_cfg.name]
    
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    
    stage = env.scene.stage
    
    prim_path_regex = asset.cfg.prim_path
    
    for env_id in env_ids.tolist():
        env_specific_ns = f"/World/envs/env_{env_id}"
        root_path_str = prim_path_regex.replace("{ENV_REGEX_NS}", env_specific_ns)
        root_path_str = root_path_str.replace("env_.*", f"env_{env_id}")
        
        shader_path = f"{root_path_str}/Looks/Diffuse/PreviewSurface"
        
        shader_prim = stage.GetPrimAtPath(shader_path)
        if not shader_prim.IsValid():
            continue
        
        shader = UsdShade.Shader(shader_prim)
        
        # HSV 生成顏色（修正版）
        if random.random() < 0.5:
            if random.random() < 0.5:
                hue = random.uniform(0.0, 0.15)     # 紅→橙→黃
            else:
                hue = random.uniform(0.85, 1.0)     # 粉紅→紫紅
        else:
            # 其他色（藍綠紫）
            hue = random.uniform(0.15, 0.85)

        # 正常鮮豔（85%）
        if random.random() < 0.85:
            saturation = random.uniform(0.8, 1.0)  # 更強飽和
            value = random.uniform(0.5, 0.85)       # 明度稍低，避免過白
        else:
            # 15% 白/淺
            saturation = random.uniform(0.0, 0.4)
            value = random.uniform(0.85, 1.0)

        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        new_color = Gf.Vec3f(r, g, b)
        
        # 設 diffuseColor
        diffuse_input = shader.GetInput("diffuseColor")
        if diffuse_input:
            diffuse_input.Set(new_color)
        
        # 強制粗糙度，避免反光過強
        roughness_input = shader.GetInput("roughness")
        if roughness_input:
            roughness_input.Set(random.uniform(0.55, 0.85))
        
        metallic_input = shader.GetInput("metallic")
        if metallic_input:
            metallic_input.Set(0.0)
    
    env.sim.render()

_COLOR_RANDOMIZERS = {}

def randomize_color_wrapper(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    event_name: str,
    asset_cfg: SceneEntityCfg,
    colors: dict,
    mesh_name: str = ".*",
):
    """
    A wrapper around mdp.randomize_visual_color to bypass the Isaac Lab
    class-initialization bug for 'reset' mode events.
    """
    import isaaclab.envs.mdp as mdp
    global _COLOR_RANDOMIZERS
    
    if event_name not in _COLOR_RANDOMIZERS:
        mock_cfg = EventTermCfg(
            func=mdp.randomize_visual_color,
            mode="reset",
            params={
                "event_name": event_name,
                "asset_cfg": asset_cfg,
                "colors": colors,
                "mesh_name": mesh_name,
            }
        )
        _COLOR_RANDOMIZERS[event_name] = mdp.randomize_visual_color(cfg=mock_cfg, env=env)
    
    # Trigger the randomization
    _COLOR_RANDOMIZERS[event_name](
        env=env,
        env_ids=env_ids,
        event_name=event_name,
        asset_cfg=asset_cfg,
        colors=colors,
        mesh_name=mesh_name,
    )
