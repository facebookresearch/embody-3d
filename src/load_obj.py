import numpy as np
from typing import List


def load_obj(path):
    """
    Load wavefront OBJ from file.

    Args:
        path: Path to OBJ file or file-like object.
        return_vn: If True, return vertex normals.
        return_vc: If True, return vertex colors.
    """

    if isinstance(path, str):
        with open(path, "r") as f:
            lines: List[str] = f.readlines()
    else:
        lines: List[str] = path.readlines()

    v = []
    vt = []
    vindices = []
    vtindices = []
    vn = []
    vc = []

    for line in lines:
        if line == "":
            break

        if line[:2] == "v ":
            v_info = line.split()[1:]
            v.append([float(x) for x in v_info])
        elif line[:2] == "vt":
            vt.append([float(x) for x in line.split()[1:]])
        elif line[:2] == "vn":
            vn.append([float(x) for x in line.split()[1:]])
        elif line[:2] == "f ":
            vindices.append(
                [int(entry.split("/")[0]) - 1 for entry in line.split()[1:]]
            )
            if line.find("/") != -1:
                vtindices.append(
                    [int(entry.split("/")[1]) - 1 for entry in line.split()[1:]]
                )

    if len(vt) == 0:
        assert (
            len(vtindices) == 0
        ), "Tried to load an OBJ with texcoord indices but no texcoords!"
        vt = [[0.5, 0.5]]
        vtindices = [[0, 0, 0]] * len(vindices)

    # If we have mixed face types (tris/quads/etc...), we can't create a
    # non-ragged array for vi / vti.
    mixed_faces = False
    for vi in vindices:
        if len(vi) != len(vindices[0]):
            mixed_faces = True
            break

    if mixed_faces:
        vi = [np.array(vi, dtype=np.int32) for vi in vindices]
        vti = [np.array(vti, dtype=np.int32) for vti in vtindices]
    else:
        vi = np.array(vindices, dtype=np.int32)
        vti = np.array(vtindices, dtype=np.int32)

    v = np.array(v, dtype=np.float32)
    out = {
        "v": v,
        "vt": np.array(vt, dtype=np.float32),
        "vi": vi,
        "vti": vti,
    }
    return out
