import torch


uav_dir_pairs = (
    (1, 2),
    (2, 4),
    (1, 3),
    (3, 5),
    (1, 6),
    (6, 8),
    (8, 10),
    (1, 7),
    (7, 9),
    (9, 11),
    (6, 12),
    (12, 14),
    (14, 16),
    (7, 13),
    (13, 15),
    (15, 17),
)  # joint id start from 1

uav_pairs = (
    (11, 9),
    (9, 7),
    (10, 8),
    (8, 6),  # arms
    (16, 14),
    (14, 12),
    (17, 15),
    (15, 13),  # legs
    (12, 6),
    (13, 7),
    (12, 13),
    (6, 7),  # torso
    (6, 1),
    (7, 1),
    (2, 1),
    (3, 2),  # upper body
    (4, 3),
    (5, 3),  # nose, eyes and ears
)

# 根据 uav_pairs 的索引生成 uav_sym_pairs 的对称关系
uav_sym_pairs = (
    (0, 2),  # 左肩和右肩的骨骼对称 (11, 9) 和 (10, 8)
    (1, 3),  # 左肘和右肘的骨骼对称 (9, 7) 和 (8, 6)
    (4, 6),  # 左髋和右髋的骨骼对称 (16, 14) 和 (17, 15)
    (5, 7),  # 左膝和右膝的骨骼对称 (14, 12) 和 (15, 13)
    (10, 11),  # 躯干对称 (12, 6) 和 (13, 7)
    (16, 17),  # 左耳和右耳的骨骼对称 (4, 3) 和 (5, 3)
)


def get_pose2vec_matrix(bone_pairs=uav_dir_pairs, num_joints=17):
    r"""get transfer matrix for transfer 3D pose to 3D direction vectors.

    Returns:
        torch.Tensor: transfer matrix, shape like [num_joints - 1, num_joints]
    """
    matrix = torch.zeros(
        (num_joints - 1, num_joints)
    )  # [V - 1, V] * [V, 3] => [V - 1, 3]
    for i, (u, v) in enumerate(bone_pairs):
        matrix[i, u - 1] = -1
        matrix[i, v - 1] = 1
    return matrix


def get_vec2pose_matrix(bone_pairs=uav_dir_pairs, num_joints=17):
    r"""get transfer matrix for transfer 3D direction vectors to 3D pose.

    Returns:
        torch.Tensor: transfer matrix, shape like [num_joints, num_joints - 1]
    """
    matrix = torch.zeros(
        (num_joints, num_joints - 1)
    )  # [V, V - 1] * [V - 1, 3] => [V, 3]
    for i, (u, v) in enumerate(bone_pairs):
        matrix[v - 1, :] = matrix[u - 1, :]
        matrix[v - 1, i] = 1
    return matrix


def get_sym_bone_matrix(sym_pairs=uav_sym_pairs, num_joints=17):
    r"""get transfer matrix for average the left and right bones

    Returns:
        torch.Tensor: transfer matrix, shape like [num_joints - 1, num_joints - 1]
    """
    matrix = torch.zeros(
        (num_joints - 1, num_joints - 1)
    )  # [V - 1, V - 1] * [V - 1, 1] => [V - 1, 1]
    for i in range(num_joints - 1):
        matrix[i, i] = 1
    for i, j in sym_pairs:
        matrix[i, i] = matrix[i, j] = 0.5
        matrix[j, j] = matrix[j, i] = 0.5
    return matrix


def get_vec_by_pose(joints):
    r"""get unit bone vec & bone len from joints

    Args:
        joints (torch.Tensor): relative to the root, shape like [num_joints, 3]
    Returns:
        torch.Tensor: unit bone vec, shape like [num_joints - 1, 3]
        torch.Tensor: bone len, shape like [num_joints - 1, 1]
    """
    bones = torch.matmul(get_pose2vec_matrix().to(joints.device), joints)
    bones_len = torch.norm(bones, dim=-1, keepdim=True)
    bones_dir = bones / (bones_len + 1e-8)
    return bones_len, bones_dir


def get_pose_by_vec(bones):
    r"""get joints from bone vec (not unit)

    Returns:
        torch.Tensor: relative to the root, shape like [num_joints, 3]
    """
    return torch.matmul(get_vec2pose_matrix().to(bones.device), bones)


# Module Testing
if __name__ == "__main__":
    joints_raw = torch.randn((100, 25, 3))
    root = joints_raw[:, :1, :]

    joints = joints_raw - root  # relative to the root joint

    bones_len, bones_dir = get_vec_by_pose(joints)
    joints_after = get_pose_by_vec(bones_dir * bones_len)

    joints_after = joints_after + root

    EPS = 1e-6
    print((torch.abs(joints_raw - joints_after) < EPS).sum() == 100 * 25 * 3)

    bones_len, bones_dir = get_vec_by_pose(joints)
    scale = torch.zeros(25 - 1).uniform_(-0.2, 0.2) + 1
    scale = scale.unsqueeze(0).unsqueeze(-1)
    scale = torch.matmul(get_sym_bone_matrix(), scale)
    for u, v in uav_sym_pairs:
        if abs(scale[0, u, 0] - scale[0, v, 0]) > EPS:
            print("Testing Failed")
