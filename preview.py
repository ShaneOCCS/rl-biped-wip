import mujoco.viewer
import mujoco

# loads louis
model = mujoco.MjModel.from_xml_path("robot/biped.xml")
data = mujoco.MjData(model)

# set standing pose
data.qpos[2] = 1.0   # torso height
data.qpos[3] = 1.0   # quaternion w (upright)
data.qpos[4] = 0.0   # quaternion x
data.qpos[5] = 0.0   # quaternion y
data.qpos[6] = 0.0   # quaternion z

# opens mujoco viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.sync()
    while viewer.is_running():
        viewer.sync()