import mujoco.viewer
import mujoco

model = mujoco.MjModel.from_xml_path("robot/biped.xml")
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.sync()
    while viewer.is_running():
        viewer.sync()