# From zengweishuai/ScaleBFM, revision abd6f17c02fe0baabc14709feb8d9ea4959aa621.
# See assets/g1/policy/scalebfm_m/PROVENANCE.md.
# Pure PyTorch definitions extracted without the IsaacLab application launcher.
import torch
from torch import nn
import numpy as np
import xml.etree.ElementTree as ET

@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)

@torch.jit.script
def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec - quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([w, x, y, z], dim=-1).view(shape)

@torch.jit.script
def quat_mul_inverse_left(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)    
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w1_inv = w1
    x1_inv = -x1
    y1_inv = -y1
    z1_inv = -z1
    ww = (z1_inv + x1_inv) * (x2 + y2)
    yy = (w1_inv - y1_inv) * (w2 + z2)
    zz = (w1_inv + y1_inv) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1_inv - x1_inv) * (x2 - y2))
    w = qq - ww + (z1_inv - y1_inv) * (y2 - z2)
    x = qq - xx + (x1_inv + w1_inv) * (x2 + w2)
    y = qq - yy + (w1_inv - x1_inv) * (y2 + z2)
    z = qq - zz + (z1_inv + y1_inv) * (w2 - x2)
    
    return torch.stack([w, x, y, z], dim=-1).view(shape)

@torch.jit.script
def quat_mul_inverse_right(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w2_inv = w2
    x2_inv = -x2
    y2_inv = -y2
    z2_inv = -z2
    ww = (z1 + x1) * (x2_inv + y2_inv)
    yy = (w1 - y1) * (w2_inv + z2_inv)
    zz = (w1 + y1) * (w2_inv - z2_inv)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2_inv - y2_inv))
    w = qq - ww + (z1 - y1) * (y2_inv - z2_inv)
    x = qq - xx + (x1 + w1) * (x2_inv + w2_inv)
    y = qq - yy + (w1 - x1) * (y2_inv + z2_inv)
    z = qq - zz + (z1 + y1) * (w2_inv - x2_inv)
    
    return torch.stack([w, x, y, z], dim=-1).view(shape)

class HumanoidTransformerPolicyWrapperWithMode(nn.Module):

    def __init__(
        self,
        policy,
        mode_mappings: torch.Tensor,
        mode_vectors: torch.Tensor,
        default_dof_pos: torch.Tensor,
        action_scale: torch.Tensor,
        context_len: int,
        future_len: int, 
        local_translation: torch.Tensor,
        local_rotation: torch.Tensor,
        parent_indices: torch.Tensor,
        joint_axis: torch.Tensor,
        selected_link_indices: torch.Tensor,
        lab_to_xml_joint_indices: torch.Tensor,
    ) -> None:
        super().__init__()

        # Modules and Components
        self.task_embedder = policy.actor_task_embedder
        self.prop_embedder = policy.actor.prop_projection
        self.action_embedder = policy.actor.action_projection
        self.transformer_blocks = policy.actor.transformer_blocks
        self.final_norm = policy.actor.final_norm
        self.projection_head = policy.actor.projection_head
        self.register_buffer("empty_embedding", policy.actor.empty_embedding)
        attn_mask = torch.zeros(2*context_len, 2*context_len, dtype=torch.bool, device=mode_vectors.device)
        row_idx = torch.arange(2*context_len - 1, device=mode_vectors.device)
        col_idx = torch.full((2*context_len - 1,), 2*context_len - 1, device=mode_vectors.device)
        attn_mask[row_idx, col_idx] = True
        self.register_buffer("self_attn_mask", attn_mask)

        # Observation and Action
        self.register_buffer("mode_mappings", mode_mappings)
        self.register_buffer("mode_vectors", mode_vectors)
        self.register_buffer("gravity_vec", torch.tensor([[0,0,-1]], dtype=torch.float, device=mode_vectors.device).unsqueeze(1).expand(-1, context_len, -1))
        self.register_buffer("default_dof_pos", default_dof_pos)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("selected_link_indices", selected_link_indices)
        tan_vec = torch.zeros(1, future_len, selected_link_indices.shape[-1], 3, dtype=torch.float,device=mode_vectors.device)
        norm_vec = torch.zeros(1, future_len, selected_link_indices.shape[-1], 3, dtype=torch.float,device=mode_vectors.device)
        tan_vec[..., 0] = 1
        norm_vec[..., -1] = 1
        self.register_buffer("tan_vec", tan_vec)
        self.register_buffer("norm_vec", norm_vec)

        # FK Modules
        self.register_buffer("local_translation", local_translation) # (num_joints, 3)
        self.register_buffer("local_rotation", local_rotation) # (num_joints, 4)
        self.register_buffer("parent_indices", parent_indices) # (num_joints + 1)
        self.register_buffer("joint_axis", joint_axis) # (num_joints, 3)
        self.register_buffer("lab_to_xml_joint_indices", lab_to_xml_joint_indices)

    def forward(
        self,
        root_quat_buffer: torch.Tensor, # wxyz
        base_ang_vel_buffer: torch.Tensor,
        dof_pos_buffer: torch.Tensor, # in Isaaclab order
        dof_vel_buffer: torch.Tensor, # in Isaaclab order
        last_action_buffer: torch.Tensor,
        target_body_pos_future_to_robot_base: torch.Tensor, # (bs, num_future, num_link, 3)
        target_body_rot_future_to_robot_base: torch.Tensor,
        mode_index: torch.Tensor,
        time_offsets: torch.Tensor,
    ) -> torch.Tensor:
        
        # Build Context
        projected_gravity_buffer = quat_apply_inverse(root_quat_buffer, self.gravity_vec) # (bs, num_context, 3)
        dof_pos_rel_buffer = dof_pos_buffer - self.default_dof_pos
        prop_obs = torch.cat([
            projected_gravity_buffer,
            base_ang_vel_buffer,
            dof_pos_rel_buffer,
            dof_vel_buffer * 0.05
        ], dim=-1)
        prop_token = self.prop_embedder(prop_obs)
        action_token = self.action_embedder(last_action_buffer)

        x = torch.empty(prop_token.shape[0], 2*prop_token.shape[1], prop_token.shape[2], dtype=prop_token.dtype, device=prop_token.device)
        x[:, ::2] = prop_token
        x[:, 1:-1:2] = action_token[:, 1:]
        x[:, 2*prop_token.shape[1]-1] = self.empty_embedding

        # Extract current state from buffer
        dof_pos = dof_pos_buffer[:, -1]

        # FK Module
        half_angles = dof_pos[:, self.lab_to_xml_joint_indices].unsqueeze(-1) / 2
        sin_half = torch.sin(half_angles)
        cos_half = torch.cos(half_angles)
        joint_rot = torch.cat([cos_half, self.joint_axis.unsqueeze(0) * sin_half], dim=-1)
        
        body_pos = torch.zeros(1, len(self.parent_indices), 3, dtype=torch.float, device=joint_rot.device)
        body_quat = torch.zeros(1, len(self.parent_indices), 4, dtype=torch.float, device=joint_rot.device)
        body_quat[..., 0] = 1
        
        for j in range(1, len(self.parent_indices)):
            j_rot = joint_rot[:, j-1]
            local_trans = self.local_translation[j-1:j]
            local_rot = self.local_rotation[j-1:j]
            parent_idx = self.parent_indices[j:j+1]
            
            parent_pos = body_pos[:, parent_idx].squeeze(1) # (1, 1, 3)
            parent_rot = body_quat[:, parent_idx].squeeze(1) # (1, 1, 4)
    
            world_trans = quat_apply(parent_rot, local_trans)
            curr_pos = parent_pos + world_trans
            curr_rot = quat_mul(local_rot, j_rot)
            curr_rot = quat_mul(parent_rot, curr_rot)
            
            body_pos[:, j] = curr_pos
            body_quat[:, j] = curr_rot

        body_pos_to_robot_base = body_pos[:, self.selected_link_indices]
        body_quat_to_robot_base = body_quat[:, self.selected_link_indices]

        # Build Task Observation
        target_body_pos_future_rel_to_robot_base = target_body_pos_future_to_robot_base - body_pos_to_robot_base[:, None, :, :]
        
        target_body_rot_future_to_robot_base_tan_norm = torch.cat([
            quat_apply(target_body_rot_future_to_robot_base, self.tan_vec),
            quat_apply(target_body_rot_future_to_robot_base, self.norm_vec),
        ], dim=-1)
        target_body_rot_future_rel_to_robot_base = quat_mul_inverse_right(
            target_body_rot_future_to_robot_base,
            body_quat_to_robot_base[:, None].expand(-1, target_body_rot_future_to_robot_base.shape[1], -1, -1)
        )
        target_body_rot_future_rel_to_robot_base_tan_norm = torch.cat([
            quat_apply(target_body_rot_future_rel_to_robot_base, self.tan_vec),
            quat_apply(target_body_rot_future_rel_to_robot_base, self.norm_vec),
        ], dim=-1)
        
        task_obs = torch.cat([
            target_body_pos_future_to_robot_base.flatten(2,3),
            target_body_pos_future_rel_to_robot_base.flatten(2,3),
            target_body_rot_future_to_robot_base_tan_norm.flatten(2,3),
            target_body_rot_future_rel_to_robot_base_tan_norm.flatten(2,3),
            time_offsets
        ], dim=-1) # (bs, nf, ndim)

        # Apply control mode
        mapping = self.mode_mappings[mode_index]
        task_obs_masked = task_obs * mapping.unsqueeze(1)
        mode_vec = self.mode_vectors[mode_index]
        task_input = torch.cat([task_obs_masked, mode_vec.unsqueeze(1).expand(-1, task_obs_masked.shape[1], -1)], dim=-1)
        task_tokens = self.task_embedder(task_input)

        # Forward
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, task_tokens, self_attn_mask=self.self_attn_mask)
        x = self.final_norm(x)

        action = self.projection_head(x[:, -1, :])
        
        # Return Direct PD target and action for buffer
        return action * self.action_scale + self.default_dof_pos, action

def build_mode_mappings(
    mode_table: torch.Tensor,
    feature_dims_per_link,
    with_time: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build (num_modes, task_obs_dim) mapping from mode table and feature dims.

    Replicates the logic of mdp.mode_mapping for each row of mode_table, so the
    exported ONNX can apply the correct mask per mode index without env.
    task_obs_dim = sum(feature_dims_per_link) + (1 if with_time else 0).
    """
    device = device or mode_table.device
    mode_table = mode_table.to(device)
    assert mode_table.dim() == 2, "mode_table must be (num_modes, num_links)"
    num_modes, num_links = mode_table.shape[0], mode_table.shape[1]
    mappings = []
    for dim in feature_dims_per_link:
        # (num_modes, num_links, dim) * (num_modes, num_links, 1) -> view (num_modes, -1)
        part = (
            torch.ones(num_modes, mode_table.shape[1], dim, device=device, dtype=torch.float32)
            * mode_table.unsqueeze(-1)
        ).view(num_modes, -1)
        mappings.append(part)
    if with_time:
        mappings.append(torch.ones(num_modes, 1, device=device, dtype=torch.float32))
    return torch.cat(mappings, dim=-1)

def parse_xml(xml_file, device = "cpu"):
    body_names = []
    parent_indices = []
    local_translation = []
    local_rotation = []
    joint_axis = []
    joint_names = []
   
    tree = ET.parse(xml_file)
    xml_doc_root = tree.getroot()
    xml_world_body = xml_doc_root.find("worldbody")
    assert xml_world_body is not None, "worldbody not found"
    
    xml_body_root = xml_world_body.find("body")
    assert xml_body_root is not None, "body not found"
    
    compiler_data = xml_doc_root.find("compiler")
    rot_unit = compiler_data.attrib.get("angle", "degree")
    assert rot_unit in ["degree", "radian"], f"Invalid rotation unit: {rot_unit}"
    
    def _add_xml_body(xml_node, parent_index, body_index):
        body_name = xml_node.attrib.get("name")
        pos_data = xml_node.attrib.get("pos", "0 0 0")
        pos = np.fromstring(pos_data, dtype=float, sep=" ")
        
        rot_data = xml_node.attrib.get("quat", "1 0 0 0")
        rot = np.fromstring(rot_data, dtype=float, sep=" ")
        
        if body_index == 0:
            pass
        else:
            curr_joints = xml_node.findall("joint")
            num_joints = len(curr_joints)
            assert num_joints == 1
            _axis = np.fromstring(curr_joints[0].attrib.get("axis"), dtype=float, sep=" ")
            axis = torch.from_numpy(_axis)
            local_rotation.append(rot)
            local_translation.append(pos)
            joint_axis.append(axis)
            joint_names.append(curr_joints[0].attrib.get("name"))
        
        body_names.append(body_name)
        parent_indices.append(parent_index)
        
        curr_index = body_index
        body_index += 1
        for child in xml_node.findall("body"):
            body_index = _add_xml_body(child, curr_index, body_index)
            
        return body_index
    
    _add_xml_body(xml_body_root, -1, 0)
    
    parent_indices = torch.tensor(parent_indices, dtype=torch.long, device=device)
    local_translation = torch.tensor(np.array(local_translation), dtype=torch.float, device=device)
    local_rotation = torch.tensor(np.array(local_rotation), dtype=torch.float, device=device)
    joint_axis = torch.stack(joint_axis, dim=0).float().to(device) # weishuai: The original variable is float64

    return body_names, joint_names, parent_indices, joint_axis, local_translation, local_rotation

