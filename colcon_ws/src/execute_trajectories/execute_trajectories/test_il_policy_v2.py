import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import ILPolicy, BlockNormalizer24
#from manipulator_mujoco.envs.ur5e_data_collection import UR5eCollectEnv


# -------------------- helpers --------------------

def find_best_checkpoint(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    cands = []
    for pat in ["*best*.pt", "*best*.pth", "*.pt", "*.pth", "*.ckpt"]:
        cands += list(p.glob(pat))
    if not cands:
        raise FileNotFoundError(f"No checkpoint found in {path}")
    best_like = [c for c in cands if "best" in c.name.lower()]
    chosen = sorted(best_like or cands, key=lambda f: f.stat().st_mtime)[-1]
    print(f"[ckpt] Using: {chosen}")
    return chosen


def maybe_load_json(p: Path):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load {p}: {e}")
        return None


def load_normalizer_from_checkpoint(ckpt_obj):
    """Load normalizer from checkpoint object"""
    if isinstance(ckpt_obj, dict) and 'normalizer' in ckpt_obj:
        norm_data = ckpt_obj['normalizer']
        print("[norm] Loaded normalizer from checkpoint")
        return norm_data
    return None


def load_normalizer_from_file(ckpt_path: Path, norm_path: str | None):
    """Load normalizer from separate file"""
    # 1) explicit path
    if norm_path:
        norm_data = maybe_load_json(Path(norm_path))
        if norm_data:
            print(f"[norm] Loaded {norm_path}")
            return norm_data
    
    # 2) sibling normalizer.json
    sib = ckpt_path.parent / "normalizer.json"
    norm_data = maybe_load_json(sib)
    if norm_data:
        print(f"[norm] Loaded {sib}")
        return norm_data
    
    print("[norm] No normalizer found; running without input normalization.")
    return None


def create_normalizer(norm_data):
    """Create normalizer object from loaded data"""
    if norm_data is None:
        return None
    
    # Try to create BlockNormalizer24 if we have the right structure
    if 'ft_indices' in norm_data and 'blocks' in norm_data:
        idx_ft = np.array(norm_data['ft_indices'])
        normalizer = BlockNormalizer24(idx_ft)
        normalizer.load_state_dict(norm_data)
        return normalizer
    
    # Fallback to simple mean/std normalization
    if 'mean' in norm_data and 'std' in norm_data:
        return {
            'mean': np.array(norm_data['mean'], dtype=np.float32),
            'std': np.array(norm_data['std'], dtype=np.float32)
        }
    
    return None


def normalize_features(x: np.ndarray, normalizer):
    """Apply normalization to features"""
    if normalizer is None:
        return x
    
    if hasattr(normalizer, 'transform'):
        # BlockNormalizer object
        return normalizer.transform(x.reshape(1, -1)).flatten()
    elif isinstance(normalizer, dict) and 'mean' in normalizer:
        # Simple mean/std normalization
        return (x - normalizer['mean']) / normalizer['std']
    
    return x


def load_model_pytorch(ckpt_path: Path, device: torch.device,
                       override_hidden: int | None, override_layers: int | None,
                       override_dropout: float | None):
    
    obj = torch.load(str(ckpt_path), map_location=device)
    
    # Extract config
    if isinstance(obj, dict) and 'config' in obj:
        cfg = obj['config']
    else:
        cfg = maybe_load_json(ckpt_path.parent / "config.json") or {}
    
    # Model dimensions and architecture
    input_dim  = int(cfg.get("input_dim", 24))
    output_dim = int(cfg.get("output_dim", 6))
    hidden     = int(override_hidden if override_hidden is not None else cfg.get("hidden", 512))
    layers     = int(override_layers if override_layers is not None else cfg.get("layers", 5))
    dropout    = float(override_dropout if override_dropout is not None else cfg.get("dropout", 0.0))

    print(f"[model] Architecture: input={input_dim}, output={output_dim}, hidden={hidden}, layers={layers}, dropout={dropout}")

    # Create model
    
    model = ILPolicy(input_dim=input_dim, output_dim=output_dim,
                       hidden=hidden, layers=layers, dropout=dropout).to(device)

    # Load state dict
    if isinstance(obj, dict):
        state = obj.get("model_state", obj.get("state_dict", obj))
    else:
        state = obj.state_dict() if hasattr(obj, 'state_dict') else obj
    
    # Remove 'module.' prefix if present (from DataParallel)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys in state dict: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
    
    model.eval()
    print(f"[model] Weights loaded successfully")
    
    return model, input_dim, output_dim, obj


def set_random_payload(env, randomize=True):
    """Set payload - either randomized or fixed"""
    if randomize:
        # Randomize like in training
        m = np.random.uniform(0.1, 0.6)  # kg
        r = np.random.uniform(0.015, 0.035)  # m
        L2 = np.random.uniform(0.03, 0.08)   # half-length [m]
        com_x = np.random.uniform(-0.02, 0.02)
        com_y = np.random.uniform(-0.02, 0.02)
        com_z = np.random.uniform(0.05, 0.15)
        com = np.array([com_x, com_y, com_z], dtype=np.float32)
    else:
        # Fixed payload (same as in your original test)
        m = 0.35
        r = 0.025
        L2 = 0.05
        com = np.array([0.0, 0.0, 0.10], dtype=np.float32)
    
    env.set_tool_payload(mass=m, com=com, cyl_size=(r, L2), axis="z", collide=True)
    env.tare_ft()
    
    return {"mass": m, "com": com.tolist(), "cyl_size": [r, L2]}


# -------------------- rollout --------------------

class DeploymentPolicy():
    def __init__(self):
        # Device selection
        self.device = torch.device("cuda")
    
        #ckpt_path = find_best_checkpoint("dagger_results_no_effort/model_iter_2/best.pt")
        ckpt_path = find_best_checkpoint("models/dagger_results_delta_penalty/model_iter_1/best.pt")
        #ckpt_path = find_best_checkpoint("best_dagger_results_fast/model_iter_10/best.pt")
    
        # Load model first to get checkpoint object
        self.model, input_dim, output_dim, ckpt_obj = load_model_pytorch(
            ckpt_path, self.device, 64, 4, 0
        )

        # Load normalizer (try checkpoint first, then separate file)
        norm_data = load_normalizer_from_checkpoint(ckpt_obj)
        if norm_data is None:
            norm_data = load_normalizer_from_file(ckpt_path, True)
    
        self.normalizer = create_normalizer(norm_data)

        ## Build env
        #env = UR5eCollectEnv(
        #    xml_path=args.xml,
        #    render_mode=("human" if args.render else None),
        #    terminate_on_collision=False,
        #    safety_joint_margin=0.0,
        #    n_substeps=1,
        #)
        self.ctrl_dt = 0.002 #float(env.model.opt.timestep)
        print(f"Control dt: {self.ctrl_dt:.4f}s")

        # Cache FT sensor addresses
        #try:
        #    sid_f = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_force")
        #    sid_t = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SENSOR, "ft_torque")
        #    f_adr = env.model.sensor_adr[sid_f] if sid_f >= 0 else -1
        #    t_adr = env.model.sensor_adr[sid_t] if sid_t >= 0 else -1
        #except:
        #    f_adr = t_adr = -1
        #    print("Warning: Could not find FT sensors")

        #def read_ft(info_dict) -> np.ndarray:
        #    """Read force-torque sensor data"""
        #    if isinstance(info_dict, dict) and "ft_tool" in info_dict:
        #        return np.asarray(info_dict["ft_tool"], dtype=np.float32)
        #    if f_adr >= 0 and t_adr >= 0:
        #        sd = env.data.sensordata
        #        f = sd[f_adr:f_adr+3]
        #        t = sd[t_adr:t_adr+3]
        #        return np.concatenate([f, t]).astype(np.float32)
        #    return np.zeros(6, dtype=np.float32)


        # Control limits
        #if hasattr(env, 'action_space'):
        #    joint_low, joint_high = env.action_space.low, env.action_space.high
        #else:
        #    # Fallback: use MuJoCo joint limits
        #    joint_low = env.model.jnt_range[:env.model.nu, 0]
        #    joint_high = env.model.jnt_range[:env.model.nu, 1]
    
        #print(f"Joint limits: [{joint_low[0]:.2f}, {joint_high[0]:.2f}] (showing first joint)")

        # Get control mode from config
        self.control_mode = "position_delta"  # default
        if isinstance(ckpt_obj, dict) and 'config' in ckpt_obj:
            self.control_mode = ckpt_obj['config'].get('control_mode', 'position')
        print(f"Control mode: {self.control_mode}")

        #horizon_steps = int(np.round(args.seconds / ctrl_dt))
        #print(f"Running {args.episodes} episodes, {horizon_steps} steps each ({args.seconds}s)")

        # Smoothing buffer for actions
        #if args.smoothing > 0:
        #    action_buffer = None
        #    print(f"Using action smoothing with alpha={args.smoothing}")


        #while True:

        self.joint_low = -0.02
        self.joint_high = 0.02

    def run(self, x):
        # print(f"\n=== Episode {ep+1}/{args.episodes} ===")

        #obs, info = env.reset()

        # Set payload (randomized or fixed based on training)
        #payload_info = set_random_payload(env, randomize=args.randomize_payload)
        #print(f"Payload: mass={payload_info['mass']:.3f}kg, com={payload_info['com']}")

        # Reset smoothing buffer
        #if args.smoothing > 0:
        #action_buffer = None

        step_errors = []

        #for k in range(horizon_steps):
            # Build feature vector
        #x = self.build_feature(obs, info)  # (24,)

        # Apply normalization
        x_norm = normalize_features(x, self.normalizer)

        # Predict action
        with torch.no_grad():
            x_tensor = torch.from_numpy(x_norm).to(self.device).unsqueeze(0).float()
            pred = self.model(x_tensor).cpu().numpy()[0]  # (6,)

        # Convert prediction based on control mode
        #q_current = env.data.qpos[:env.model.nu].astype(np.float32)
        q_current = x[:6]

        #if self.control_mode == "position":
        #    action = pred  # direct position command
        #elif self.control_mode == "velocity":
        #    # Integrate velocity to get position
        #    action = q_current + pred * self.ctrl_dt
        #elif self.control_mode == "position_delta":
        ##else:
        ##    action = pred  # fallback

        # Apply smoothing
        #if args.smoothing > 0:
        #    if action_buffer is None:
        #        action_buffer = action.copy()
        #    else:
        #        action_buffer = (1 - args.smoothing) * action_buffer + args.smoothing * action
        #    action = action_buffer

        # Clip to joint limits
        pred = np.clip(pred.astype(np.float32), self.joint_low, self.joint_high)


        #action = q_current + pred  # add delta to current position
        # action = action * 0.5
        # Apply action
        #obs, reward, term, trunc, info = env.step(action)

        # Track error for debugging
        #position_error = np.linalg.norm(action - q_current)
        #step_errors.append(position_error)

        #if args.render:
        #    env.render()

        # Early termination check
        #if term or trunc:
        #    print(f"Episode terminated early at step {k+1}/{horizon_steps}")
        #    break

        ## Print progress occasionally
        #if k % max(1, horizon_steps // 10) == 0:
        #    print(f"  Step {k:3d}/{horizon_steps}: pos_error={position_error:.4f}")

        # Episode summary
        #avg_error = np.mean(step_errors) if step_errors else 0
        #max_error = np.max(step_errors) if step_errors else 0
        # print(f"Episode {ep+1} completed: avg_error={avg_error:.4f}, max_error={max_error:.4f}")
        return pred #action

    #def build_feature(self, obs, info) -> np.ndarray:
    #    """Build 24D feature vector: [q(6), qd(6), tau(6), ft(6)]"""
    #    nu = env.model.nu
    #    q   = env.data.qpos[:nu].astype(np.float32)
    #    qd  = env.data.qvel[:nu].astype(np.float32)
    #    tau = env.data.qfrc_actuator[:nu].astype(np.float32)
    #    ft  = read_ft(info)
    #    return np.concatenate([q, qd, tau, ft]).astype(np.float32)


# -------------------- cli --------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=str, required=False,
                    default="Manipulator-Mujoco/manipulator_mujoco/assets/robots/ur5e/ur5e.xml",
                    help="Path to UR5e XML")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Path to checkpoint file or directory")
    ap.add_argument("--norm", type=str, default=None,
                    help="Path to normalizer.json (optional). If omitted, tries sibling next to ckpt")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--randomize-payload", action="store_true",
                    help="Randomize tool mass/COM per episode and tare FT if available.")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])

    # Model overrides (only if your ckpt/config lacks these)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)

    ap.add_argument("--smoothing", type=float, default=0.0,
                    help="EMA factor on predicted actions (0 disables)")
    
    ap.add_argument("--hitl", action="store_true", default=False, required=False)
    
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_rollout(args)