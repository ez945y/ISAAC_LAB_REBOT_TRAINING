import random
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from pxr import UsdGeom, UsdShade, Gf

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
            print(f"[Color Random] Shader not found: {shader_path}")
            continue
        
        shader = UsdShade.Shader(shader_prim)
        
        # 隨機顏色
        r = random.uniform(color_ranges["r"][0], color_ranges["r"][1])
        g = random.uniform(color_ranges["g"][0], color_ranges["g"][1])
        b = random.uniform(color_ranges["b"][0], color_ranges["b"][1])
        new_color = Gf.Vec3f(r, g, b)
        
        diffuse_input = shader.GetInput("diffuseColor")
        if diffuse_input:
            diffuse_input.Set(new_color)
        else:
            print(f"[Warning] diffuseColor input not found in {shader_path}")
        
        metallic_input = shader.GetInput("metallic")
        if metallic_input:
            metallic_input.Set(0.0)
        roughness_input = shader.GetInput("roughness")
        if roughness_input:
            roughness_input.Set(0.4)
    
    env.sim.render()  # 更新畫面


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
