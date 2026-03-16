import mujoco
import mujoco.viewer

# loads louis
model = mujoco.MjModel.from_xml_path("robot/biped.xml")
data = mujoco.MjData(model)

# opens mujuco view
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()