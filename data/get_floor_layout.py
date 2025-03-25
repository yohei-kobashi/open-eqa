import json
import os
import re
from tqdm import tqdm
import numpy as np
import open3d as o3d
from open3d.visualization import rendering

def render_top_view_aligned(
    mesh_path,
    output_path,
    base_height=1024
):
    """
    ScanNetの完成メッシュ(*_vh_clean_2.ply等)を読み込み、
    1) OrientedBoundingBox(OBB) でメッシュを回転して軸揃え
    2) 縦横比を自動調整
    3) 真上から正射投影でオフスクリーンレンダリング
    する最小限のサンプル。
    """
    # --- 1) メッシュ読み込み ---
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # --- 2) Oriented Bounding Box (OBB) 取得 ---
    #     部屋の大きな面を主成分方向に合わせるような回転が得られる
    obb = mesh.get_oriented_bounding_box()
    # OBBの回転行列 (3x3) を取り出し、中心に対して逆回転を適用
    # すると、OBBがグローバル座標軸と揃う形になる
    R_obb = obb.R  # 3x3 rotation
    center_obb = obb.center
    # メッシュを逆回転
    mesh.rotate(R_obb.T, center_obb)

    # --- 3) 軸揃え後の Axis Aligned Bounding Box (AABB) を取得 ---
    aabb = mesh.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()  # [size_x, size_y, size_z]
    center = aabb.get_center()

    # --- 4) レンダリング解像度をアスペクト比に合わせる ---
    # 縦横比 = (X幅) / (Y幅)
    aspect_ratio = extent[0] / max(extent[1], 1e-6)
    # base_height を基準に width を計算
    height = base_height
    width = int(height * aspect_ratio)

    # --- 5) オフスクリーンレンダラー作成 ---
    renderer = rendering.OffscreenRenderer(width, height)

    # マテリアル設定
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    # メッシュを追加
    renderer.scene.add_geometry("scene_mesh", mesh, mat)

    # --- 6) カメラ設定 (真上から) ---
    #     Z軸が上になっている想定
    #     AABBの高さ方向 (extent[2]) の2倍上から見下ろす
    # eye = center + np.array([0, 0, extent[2] * 2.0])
    # up = np.array([0, 1, 0])  # Yを上とする
    eye = center + np.array([0, 0, extent[2] * 2])
    up = np.array([0, 1, 0])  # Yを上とする
    renderer.scene.camera.look_at(center, eye, up)

    # 正射投影設定
    half_size_x = extent[0] * 0.5
    half_size_y = extent[1] * 0.5
    near_plane = 0.1
    far_plane = extent[2] * 10.0
    renderer.scene.camera.set_projection(
        rendering.Camera.Projection.Ortho,
        -half_size_x, half_size_x,
        -half_size_y, half_size_y,
        near_plane, far_plane
    )

    # --- 7) ライティング (必要に応じて調整) ---
    renderer.scene.scene.set_sun_light(
        direction=[-1.0, -1.0, -1.0],
        color=[1.0, 1.0, 1.0],
        intensity=100000
    )
    renderer.scene.scene.enable_sun_light(True)

    # --- 8) レンダリングして保存 ---
    image = renderer.render_to_image()
    o3d.io.write_image(output_path, image)
    print(f"Saved aligned top-down view to: {output_path}")

def main():
    # 出力先ディレクトリの存在確認・作成
    output_dir = "floor_layouts/scannet-v0"
    os.makedirs(output_dir, exist_ok=True)

    # 指定のディレクトリからシーン一覧を取得（例外処理付き）
    try:
        with open("open-eqa-v0_sceneType.json", encoding="utf-8") as f:
            scenes_data = json.load(f)
        scenes = list(set([row["episode_history"] for row in scenes_data if "sceneType" in row]))
    except Exception as e:
        print(f"JSON 読み込みエラー: {e}")
        scenes = []

    # シーンごとにplyファイルを用いて俯瞰図を生成
    for episode_history in tqdm(scenes):
        m = re.search(r"scannet-v0/(\d+)-scannet-(.*)", episode_history)
        if not m:
            continue
        scene_n = m.group(1)
        scene_id = m.group(2)
        base_dir = None
        if os.path.exists(f"raw/scannet/scans/{scene_id}"):
            base_dir = f"raw/scannet/scans/{scene_id}"
        elif os.path.exists(f"raw/scannet/scans_test/{scene_id}"):
            base_dir = f"raw/scannet/scans_test/{scene_id}"
        
        if base_dir is None:
            continue
        
        ply_file = os.path.join(base_dir, f"{scene_id}_vh_clean_2.ply")
        if not os.path.exists(ply_file):
            ply_file = os.path.join(base_dir, f"{scene_id}_vh_clean.ply")
        render_top_view_aligned(ply_file, f"floor_layouts/scannet-v0/{scene_n}-scannet-{scene_id}.png")
        
if __name__ == "__main__":
    main()
