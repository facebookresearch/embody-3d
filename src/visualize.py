"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
from typing import List, Tuple

import numpy as np
import OpenGL.GL as gl
import pyrender
import torch
import torch.nn as nn
import trimesh
from pyrender.constants import RenderFlags
from load_obj import load_obj

os.environ["PYOPENGL_PLATFORM"] = "egl"
trimesh.util.log.setLevel("ERROR")


class PyrenderRenderer(nn.Module):
    def __init__(
        self,
        topology_path: str,
        rendering_height: int = 512,
        rendering_width: int = 768,
        campos: torch.Tensor = None,
        magnification_factor: float = 1.0,  # 1000.0,
    ):
        super().__init__()
        topology_dict = load_obj(topology_path)
        self.faces = topology_dict["vi"]
        if campos is None:
            self.at = (0.0, -1.0, 0.0)
            self.azimuth = 0
            self.dist = -100.0
            self.elevation = -10.0
            self.light_loc = (0.0, 1.0, 15.0)
        else:
            raise NotImplementedError("configurable campos is not implemented yet")
        self.rendering_height = rendering_height * 2
        self.rendering_width = rendering_width * 2
        self.magnification_factor = magnification_factor
        self._setup_renderer()
        self._setup_floor()

    def _generate_checkerboard_geometry(
        self,
        length: float = 10,
        color0: List[float] = [0.4, 0.4, 0.4],
        color1: List[float] = [0.2, 0.2, 0.2],
        tile_width: float = 2,
        alpha: float = 0.5,
        up: str = "y",
    ):
        """helper function to generate a simple checkerboard geometry as the floor"""
        assert up == "y" or up == "z"
        color0 = np.array(color0 + [alpha])
        color1 = np.array(color1 + [alpha])
        radius = length / 2.0
        num_rows = num_cols = int(length / tile_width)
        vertices = []
        vert_colors = []
        faces = []
        face_colors = []
        for i in range(num_rows):
            for j in range(num_cols):
                u0, v0 = j * tile_width - radius, i * tile_width - radius
                us = np.array([u0, u0, u0 + tile_width, u0 + tile_width])
                vs = np.array([v0, v0 + tile_width, v0 + tile_width, v0])
                zs = np.zeros(4)
                if up == "y":
                    cur_verts = np.stack([us, zs, vs], axis=-1)
                else:
                    cur_verts = np.stack([us, vs, zs], axis=-1)

                cur_faces = np.array([[0, 1, 3], [1, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=int)
                cur_faces += 4 * (i * num_cols + j)
                use_color0 = (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1)
                cur_color = color0 if use_color0 else color1
                cur_colors = np.array([cur_color, cur_color, cur_color, cur_color])

                vertices.append(cur_verts)
                faces.append(cur_faces)
                vert_colors.append(cur_colors)
                face_colors.append(cur_colors)
        vertices = np.concatenate(vertices, axis=0).astype(np.float32)
        vert_colors = np.concatenate(vert_colors, axis=0).astype(np.float32)
        faces = np.concatenate(faces, axis=0).astype(np.float32)
        face_colors = np.concatenate(face_colors, axis=0).astype(np.float32)

        return vertices, faces, vert_colors, face_colors

    def _normalize(self, x):
        """Returns a normalized vector."""
        return x / torch.linalg.norm(x)

    def _viewmatrix(self, center, up, pos):
        """Returns a camera transformation matrix.

        Args:
            center: Point where the camera is looking.
            up: The upward direction of the camera.
            pos: The position of the camera.

        Returns:
            A camera transformation matrix.
        """
        lookat = center - pos
        vec2 = self._normalize(lookat)
        vec1_avg = self._normalize(up)
        vec0 = self._normalize(torch.cross(vec1_avg, vec2))
        vec1 = self._normalize(torch.cross(vec2, vec0))
        m = torch.stack([vec0, -vec1, -vec2, pos], 1)
        return m

    def _setup_renderer(self) -> None:
        """function to set up the scene with camera, lights, and renderer"""
        up_dir = 1
        self.scene = pyrender.Scene(ambient_light=np.array([0.4, 0.4, 0.4]), bg_color=(0.0, 0.0, 0.0))
        camera = pyrender.PerspectiveCamera(yfov=(2 * np.pi / 180), aspectRatio=1.5)
        pos = torch.tensor([self.at[0], self.at[1] + self.elevation, self.at[2] - self.dist], dtype=torch.float)
        center = torch.tensor(self.at, dtype=torch.float)
        up = torch.tensor([0, up_dir, 0], dtype=torch.float)
        camRT = self._viewmatrix(center, up, pos)
        camRT = torch.cat([camRT, torch.tensor([[0, 0, 0, 1]])], dim=0)
        self.camera_node = self.scene.add(camera, pose=camRT, name="camera")

        # Enhanced lighting setup for better detail visibility with reduced harsh shadows
        # Main directional light (good detail visibility)
        main_light = pyrender.DirectionalLight(color=np.array([0.6, 0.6, 0.6]), intensity=4.0)
        main_light_pos = torch.tensor([0, -5e3, 5e3], dtype=torch.float)
        main_lightRT = self._viewmatrix(center, up, main_light_pos)
        main_lightRT = torch.cat([main_lightRT, torch.tensor([[0, 0, 0, 1]])], dim=0)
        self.main_light_node = self.scene.add(main_light, pose=main_lightRT, name="main_light")

        # Add ambient-like fill light to reduce harsh shadows
        fill_light = pyrender.DirectionalLight(color=np.array([0.4, 0.4, 0.4]), intensity=2.0)  # Soften shadows
        fill_light_pos = torch.tensor([3e3, -2e3, -2e3], dtype=torch.float)  # From opposite side
        fill_lightRT = self._viewmatrix(center, up, fill_light_pos)
        fill_lightRT = torch.cat([fill_lightRT, torch.tensor([[0, 0, 0, 1]])], dim=0)
        self.fill_light_node = self.scene.add(fill_light, pose=fill_lightRT, name="fill_light")

        # Top light for overall illumination (offset to avoid numerical issues)
        top_light = pyrender.DirectionalLight(color=np.array([0.3, 0.3, 0.3]), intensity=1.5)  # Gentle top lighting
        top_light_pos = torch.tensor([500, -3e3, 500], dtype=torch.float)  # From above, slightly offset
        top_lightRT = self._viewmatrix(center, up, top_light_pos)
        top_lightRT = torch.cat([top_lightRT, torch.tensor([[0, 0, 0, 1]])], dim=0)
        self.top_light_node = self.scene.add(top_light, pose=top_lightRT, name="top_light")

        # Use softer shadows
        self.flags = RenderFlags.SHADOWS_DIRECTIONAL
        # Keep reference to main light for backwards compatibility
        self.light_node = self.main_light_node

        # set up offscreen renderer
        self.renderer = pyrender.OffscreenRenderer(self.rendering_width, self.rendering_height, point_size=6.0)

    def _setup_floor(self) -> None:
        """function to add the floor to the scene"""
        v, f, color_v, _ = self._generate_checkerboard_geometry()
        v[..., 1] += 0.01  # move the floor a bit down so we can see the feet better
        floor_tri = trimesh.creation.Trimesh(vertices=v, faces=f, face_colors=color_v, process=False)
        self.floor = pyrender.Mesh.from_trimesh(floor_tri, smooth=False)
        self.scene.add(self.floor)

    def _create_mesh_or_point(self, v: torch.Tensor, color: Tuple):
        """function to create a mesh or a point cloud from the vertices
        Args:
            v: vertices of the mesh or condition (V x 3)
            offset: offset to add to the vertices to nudge it away from center
            c: color to render out the vertices
            mesh_mode: bool; create mesh if true, else create point cloud
        Returns:
            mesh: pyrender object
        """
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        v = v * self.magnification_factor
        v[..., 1] *= -1  # convention flips y axis
        # Create vertex colors array (same color for all vertices)
        vertex_colors = np.tile(color, (len(v), 1))
        mesh = trimesh.creation.Trimesh(vertices=v, faces=self.faces, vertex_colors=vertex_colors)
        # Enable smooth shading for rounded edges
        trimesh.repair.fix_normals(mesh)
        # Smooth vertex normals for better shading
        mesh.vertex_normals
        mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)  # Enable smooth shading
        return mesh

    def forward(
        self,
        geometry_list: List[torch.Tensor],
        color: Tuple = (102 / 255, 178 / 255, 255 / 255, 1.0),
    ) -> np.ndarray:
        """
        Function to render mesh given the vertices of the mesh.
        If save name is not none, then save the image to the specified path.
        Args:
            geometry_list: list of vertices of the mesh of shape (B x T x V x 3)
            color: list of colors to render out the meshes
            conditioning_pts: conditioning that will be rendered as 3d point clouds
                (B x T x V x 3)
            conditioning_meshes: conditioning rendered as meshes (B x T x V x 3)
            save_name: path to save the image, will not save if None
        Returns:
            imgs: B x T x 3 x H x W batch of sequences of rendered images in [0, 255]
        """
        assert len(geometry_list) > 0, "found empty geometry list"
        B, T = geometry_list[0].shape[:2]
        imgs = np.zeros((B, T, 3, self.rendering_height, self.rendering_width), dtype=np.uint8)
        for b in range(B):
            for t in range(T):
                mesh_node_list = []
                for geometry in geometry_list:
                    mesh = self._create_mesh_or_point(geometry[b, t], color=color)
                    mesh_node = self.scene.add(mesh)
                    mesh_node_list.append(mesh_node)
                    # render the stuff now
                    gl.glEnable(gl.GL_BLEND)
                    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
                    rgb_image, _ = self.renderer.render(self.scene, flags=self.flags)
                    rgb_image = ((rgb_image / rgb_image.max()) * 255).astype(np.uint8)
                    imgs[b, t] = rgb_image.transpose(2, 0, 1)  # H x W x 3 -> 3 x H x W
                # remove from the frame before you render again
                for mesh_node in mesh_node_list:
                    self.scene.remove_node(mesh_node)
        return imgs
