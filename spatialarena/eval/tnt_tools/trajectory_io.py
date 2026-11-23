"""Reading and writing Tanks and Temples camera trajectory (.log) files."""

import numpy as np
import open3d as o3d


class CameraPose:
    """One trajectory entry: its metadata integers and 4x4 pose matrix."""

    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return ("Metadata : " + " ".join(map(str, self.metadata)) + "\n" +
                "Pose : " + "\n" + np.array_str(self.pose))


def convert_trajectory_to_pointcloud(traj):
    """Returns a point cloud of the camera centers of `traj`."""
    pcd = o3d.geometry.PointCloud()
    for t in traj:
        pcd.points.append(t.pose[:3, 3])
    return pcd


def read_trajectory(filename):
    """Reads a .log trajectory into a list of `CameraPose`."""
    traj = []
    with open(filename, "r") as f:
        metastr = f.readline()
        while metastr:
            metadata = map(int, metastr.split())
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=" \t")
            traj.append(CameraPose(metadata, mat))
            metastr = f.readline()
    return traj


def write_trajectory(traj, filename):
    """Writes a list of `CameraPose` to a .log trajectory file."""
    with open(filename, "w") as f:
        for x in traj:
            p = x.pose.tolist()
            f.write(" ".join(map(str, x.metadata)) + "\n")
            f.write("\n".join(
                " ".join(map("{0:.12f}".format, p[i])) for i in range(4)))
            f.write("\n")
