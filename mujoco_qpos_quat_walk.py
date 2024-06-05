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
from mujoco_qpos_quat_walk_scale_speed_sample import config_234_joint_init_scale


# input speed between 0.1 and 1.0m/s or -0.1 to -1.0m/s
##############################################33
speed=0
HZ=120
length_s=10
##############################################33


negativeSpeed=speed<0
speed=abs(speed)



scales=np.load("planning_motion/scale.npy")
speeds=np.load("planning_motion/speed.npy")

index=np.argmin(np.abs(speeds-speed))
scale=scales[index]
scale=0

# scale=1








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






arm_0_1=0.2*speed
arm_0_2=-arm_0_1/5
arm_0_4=-arm_0_1/2
delta_arm_0_1=-0.8*speed
delta_arm_0_2=-delta_arm_0_1/5
delta_arm_0_4=delta_arm_0_1/2

theta_0_2 = config_234_joint_init_scale[0]
theta_0_3 = config_234_joint_init_scale[1]
theta_0_4 = config_234_joint_init_scale[2]*scale
delta_2=config_234_joint_init_scale[3]
delta_3=config_234_joint_init_scale[4]*scale
delta_4=config_234_joint_init_scale[5]*scale

if negativeSpeed:
    arm_0_1=-0.2*speed
    arm_0_2=-arm_0_1/5
    arm_0_4=arm_0_1/2
    delta_arm_0_1=0.8*speed
    delta_arm_0_2=-delta_arm_0_1/5
    delta_arm_0_4=-delta_arm_0_1/2

    theta_0_2 = -0.6*scale
    theta_0_3 = 0.4*scale
    theta_0_4 = 0.0*scale
    delta_2=0.6*scale
    delta_3=-0.2*scale
    delta_4=0.0*scale
    speed*=-1
    


T=1.0
# speed=np.sin(abs(theta_0_2))*robot_knee_length+np.sin(abs(theta_0_2-theta_0_3))*robot_ankle_length
# speed*=2
new_q_res = np.zeros((length_s*HZ,39)) 
for i in range(len(new_q_res)):
    new_q_res[i,6]=1
    new_q_res[i,7+2]=theta_0_2+delta_2*(1-np.cos(2*np.pi/T*i/HZ+negativeSpeed*np.pi))/2
    new_q_res[i,7+6+2]=theta_0_2+delta_2*(1-np.cos(2*np.pi/T*i/HZ+np.pi+negativeSpeed*np.pi))/2
    new_q_res[i,7+3]=theta_0_3+delta_3*(1-np.cos(2*np.pi/T*i/HZ+negativeSpeed*np.pi))/2
    new_q_res[i,7+6+3]=theta_0_3+delta_3*(1-np.cos(2*np.pi/T*i/HZ+np.pi+negativeSpeed*np.pi))/2

    new_q_res[i,7+4]=-new_q_res[i,7+2]-new_q_res[i,7+3]
    new_q_res[i,7+6+4]=-new_q_res[i,7+6+2]-new_q_res[i,7+6+3]

    new_q_res[i,7+18]=arm_0_1+delta_arm_0_1*(1-np.cos(2*np.pi/T*i/HZ+np.pi))/2
    new_q_res[i,7+19]=arm_0_2+delta_arm_0_2*(1-np.cos(2*np.pi/T*i/HZ+np.pi))/2
    new_q_res[i,7+20]=0.2
    new_q_res[i,7+21]=arm_0_4+delta_arm_0_4*(1-np.cos(2*np.pi/T*i/HZ+np.pi))/2

    new_q_res[i,7+18+7]=arm_0_1+delta_arm_0_1*(1-np.cos(2*np.pi/T*i/HZ))/2
    new_q_res[i,7+19+7]=-arm_0_2+-delta_arm_0_2*(1-np.cos(2*np.pi/T*i/HZ))/2
    new_q_res[i,7+20+7]=-0.2
    new_q_res[i,7+21+7]=arm_0_4+delta_arm_0_4*(1-np.cos(2*np.pi/T*i/HZ))/2
    
    
    



    new_q_res[i,0]=speed*i/HZ







np.savetxt(f"planning_motion/cos_slow_walk_{speed}.txt",new_q_res)



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





