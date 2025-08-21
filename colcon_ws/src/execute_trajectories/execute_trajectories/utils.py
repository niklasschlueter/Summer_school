import pinocchio as pin
import numpy as np
from collections import deque

def print_out_collision_results(collision_model, collision_data):
    # For debug
    # Print the status of collision for all collision pairs
    for k in range(len(collision_model.collisionPairs)):
        cr = collision_data.collisionResults[k]
        cp = collision_model.collisionPairs[k]
        print(
            "collision pair:",
            cp.first,
            ",",
            cp.second,
            "- collision:",
            "Yes" if cr.isCollision() else "No",
        )


def update_pinocchio(model, data, collision_model, collision_data, visual_model, visual_data, q):
    """Update Pinocchio model and data with given configuration q"""
    # Update joint configuration
    pin.framesForwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    
    # Update geometry placements
    pin.updateGeometryPlacements(model, data, collision_model, collision_data)
    pin.updateGeometryPlacements(model, data, visual_model, visual_data)

    return model, data, collision_model, collision_data, visual_model, visual_data


def get_ee_position_and_rotation(model, data, q, update=True):
    """Get end-effector position and orientation"""
    if update:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
    
    ee_frame_id = model.getFrameId('tool0')
    ee_position = data.oMf[ee_frame_id].translation
    ee_rotation = data.oMf[ee_frame_id].rotation
    
    return ee_position, ee_rotation



class MovingAverageFilter:
    def __init__(self, window_size=5, vector_size=6):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.vector_size = vector_size

    def update(self, new_value: np.ndarray) -> np.ndarray:
        """Add new 6x1 vector and return moving average"""
        if new_value.shape != (self.vector_size,):
            raise ValueError(f"Expected shape ({self.vector_size},), got {new_value.shape}")

        self.buffer.append(new_value)
        #print(f"Buffer size: {len(self.buffer)}")
        return np.mean(self.buffer, axis=0) if len(self.buffer) > 0 else new_value



class StateLogger:
    def __init__(self):
        # Just collect in lists, convert later
        self.time = []
        self.pos = []
        self.vel = []
        self.eff = []
        self.ft = []
        self.action = []

    def reset(self):
        self.__init__()

    def log(self, t, pos, vel, eff, ft, action):
        # Lists are very fast for append
        self.time.append(t)
        self.pos.append(pos)
        self.vel.append(vel)
        self.eff.append(eff)
        self.ft.append(ft)
        self.action.append(action)

    def save(self, filename="experiment_data.npz"):
        np.savez(
            filename,
            time=np.array(self.time),
            pos=np.array(self.pos),
            vel=np.array(self.vel),
            eff=np.array(self.eff),
            ft = np.array(self.ft),
            action = np.array(self.action)
        )
