import mujoco.viewer
import mujoco
import jax

model = mujoco.MjModel.from_xml_path("robot/biped.xml")
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
print(jax.devices())
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.sync()
    while viewer.is_running():
        viewer.sync()