"""
將 USD 中的各個 Body 拆成獨立的 USD 檔案，並保留材質。
需透過 Isaac Lab 執行:
    python scripts/split_usd_bodies.py --input tools/exp/test/test3.usd --headless
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="來源 USD 檔案路徑")
parser.add_argument("--output_dir", type=str, default=None, help="輸出目錄")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# pxr 在 AppLauncher 啟動後才可用
from pxr import Usd, UsdGeom, UsdShade, Sdf


def split_bodies(input_path, output_dir):
    stage = Usd.Stage.Open(input_path)
    if not stage:
        print(f"[ERROR] 無法開啟: {input_path}")
        return

    print(f"\n[INFO] USD 結構:")
    for prim in stage.Traverse():
        depth = str(prim.GetPath()).count("/") - 1
        print(f"{'  ' * depth}{prim.GetName()}  (type={prim.GetTypeName()})")

    # 找 default prim 或 root
    dp = stage.GetDefaultPrim()
    parent = dp if (dp and dp.IsValid()) else stage.GetPseudoRoot()

    # 收集 Body prim
    body_prims = []
    for child in parent.GetChildren():
        if "Body" in child.GetName() or "body" in child.GetName():
            body_prims.append(child)
        else:
            for gc in child.GetChildren():
                body_prims.append(gc)

    if not body_prims:
        print("[WARN] 找不到 Body prim")
        return

    # 找材質根路徑
    mat_roots = set()
    for p in stage.Traverse():
        ps = str(p.GetPath())
        if p.GetTypeName() in ("Material", "Shader") or "Look" in ps or "Material" in ps:
            parts = ps.split("/")
            if len(parts) >= 2:
                mat_roots.add("/" + parts[1])

    os.makedirs(output_dir, exist_ok=True)
    root_layer = stage.GetRootLayer()

    print(f"\n[INFO] 材質根: {mat_roots}")
    print(f"[INFO] 拆分 {len(body_prims)} 個 Body...\n")

    for prim in body_prims:
        name = prim.GetName()
        out = os.path.join(output_dir, f"{name}.usd")
        if os.path.exists(out):
            os.remove(out)

        ns = Usd.Stage.CreateNew(out)
        UsdGeom.SetStageUpAxis(ns, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(ns, UsdGeom.GetStageMetersPerUnit(stage))

        # 複製 Body
        Sdf.CopySpec(root_layer, prim.GetPath(), ns.GetRootLayer(), Sdf.Path(f"/{name}"))

        # 找此 Body 綁定的材質
        bound_mats = set()
        for p in Usd.PrimRange(prim):
            mat, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
            if mat:
                bound_mats.add(str(mat.GetPath()))

        # 複製材質根
        copied = set()
        for mp in bound_mats:
            parts = mp.split("/")
            if len(parts) >= 2:
                mr = "/" + parts[1]
                if mr not in copied:
                    Sdf.CopySpec(root_layer, Sdf.Path(mr), ns.GetRootLayer(), Sdf.Path(mr))
                    copied.add(mr)

        # fallback: 沒找到材質就全部複製
        if not bound_mats:
            for mr in mat_roots:
                Sdf.CopySpec(root_layer, Sdf.Path(mr), ns.GetRootLayer(), Sdf.Path(mr))

        ns.SetDefaultPrim(ns.GetPrimAtPath(f"/{name}"))
        ns.GetRootLayer().Save()
        print(f"  ✓ {name} ({len(bound_mats)} materials) → {out}")

    print(f"\n[INFO] 完成！")


if __name__ == "__main__":
    split_bodies(os.path.abspath(args.input),
                 args.output_dir or os.path.join(os.path.dirname(os.path.abspath(args.input)), "bodies"))
    simulation_app.close()
