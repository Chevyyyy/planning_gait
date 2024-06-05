import time
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer
import numpy as np
import json
import os
import pinocchio
import pygame
import csv
from scipy.spatial.transform import Rotation
from pinocchio.robot_wrapper import RobotWrapper








robot=RobotWrapper.BuildFromURDF("GR1_collision_box/urdf/GR1T1_32.urdf", [os.path.join("GR1_collision_box/urdf")])
rmodel=robot.model
rdata=rmodel.createData()


# Initialize pygame
pygame.init()


position_data = []
quaternion_data = []
euler_data = []
inverse_quaternion_data = []

path_to_mujoco = "GR1_collision_box/urdf/GR1T1.xml"
m = mujoco.MjModel.from_xml_path(path_to_mujoco)

m.opt.gravity = [0, 0, 0]
d = mujoco.MjData(m)
nstate = 32


    
i = 0
start = False


paused = False

def flat_feet(q_res):
    q_res[:,7+4]=-q_res[:,7+2]-q_res[:,7+3]
    q_res[:,7+4+6]=-q_res[:,7+2+6]-q_res[:,7+3+6]
    return q_res

    


T=1.0
# speed=np.sin(abs(theta_0_2))*robot_knee_length+np.sin(abs(theta_0_2-theta_0_3))*robot_ankle_length
# speed*=2

    
l1 = 0.46999999886713895
l2 = 0.33999999918048496    
z=0.88
hip_pitch_z=z-0.18
double_support_ratio=0.3
    


theta_1=np.arccos((hip_pitch_z)/(l1+l2))





new_q_res=np.zeros((1,39))
new_q_res[:,6]=1
new_q_res[:,7+2]=np.deg2rad(-20)
new_q_res[:,7+3]=np.deg2rad(20)
new_q_res[:,7+6+2]=theta_1

new_q_res=flat_feet(new_q_res)





def key_callback(keycode):
    if chr(keycode) == " ":
        global paused
        paused = not paused


with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
    start_time = time.time()
    while viewer.is_running() and time.time() - start_time < 6000:
        if not paused:
            step_start = time.time()
            mujoco.mj_step(m, d)
            if i >= len(new_q_res):
                i = 0
            # i=0
            # print(i)

            new_q_res[:,3:7]=0
            d.qpos[:] = new_q_res[i][:]
            # d.qpos[:] = new_q_res[594][:]


            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
            viewer.sync()
            time.sleep(1/120)
            i += 1





