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



import yaml
config_234_joint_init_scale=[-0.2,0.6,0,-0.9,0.3,0.0]


robot=RobotWrapper.BuildFromURDF("GR1_collision_box/urdf/GR1T1.urdf", [os.path.join("GR1_collision_box/urdf")])
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


theta_0_2 = config_234_joint_init_scale[0] 
theta_0_3 = config_234_joint_init_scale[1] 
theta_0_4 = config_234_joint_init_scale[2] 

delta_2=config_234_joint_init_scale[3]
delta_3=config_234_joint_init_scale[4]
delta_4=config_234_joint_init_scale[5]
T=1.0
robot_knee_length = 0.46999999886713895
robot_ankle_length = 0.33999999918048496


# speed=np.sin(abs(theta_0_2))*robot_knee_length+np.sin(abs(theta_0_2-theta_0_3))*robot_ankle_length
# speed*=2
scale=np.arange(0.1,2.5,2.4/400)
new_q_res = np.zeros((400,39)) 
for i in range(len(new_q_res)):
    new_q_res[i,7+2]=theta_0_2
    new_q_res[i,7+6+2]=theta_0_2+delta_2
    new_q_res[i,7+3]=theta_0_3
    new_q_res[i,7+6+3]=theta_0_3+delta_3*scale[i]

    
    





speed=[]

for q0 in new_q_res:
    pinocchio.forwardKinematics(rmodel, rdata, q0)
    pinocchio.updateFramePlacements(rmodel, rdata)
    z_left_x=rdata.oMf[15].translation[0]
    z_right_x=rdata.oMf[35].translation[0]
    speed.append(2*abs(z_left_x-z_right_x)/T)



np.save("planning_motion/speed.npy",speed)
np.save("planning_motion/scale.npy",scale)

plt.plot(scale,speed)
plt.show()


