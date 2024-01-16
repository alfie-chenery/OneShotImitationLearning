import pybullet as p
import time
import pybullet_data
import pickle
import os
import keyboard
from controller import XboxController
 
p.connect(p.GUI)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
armId = p.loadURDF("franka_panda/panda.urdf", [0,0,0], [0,0,0,1], useFixedBase=True)
numJoints = p.getNumJoints(armId)
controller = XboxController()


def Clamp(val, low, high):
    if val < low:
        return low
    elif val > high:
        return high
    else:
        return val


def get_joint_positions():
    return [p.getJointState(armId, i)[0] for i in range(numJoints)]


#This function is blocking, no input is accepted until to robot is done moving
def set_joint_positions(prev_positions, desired_positions, interpolation_steps=100):

    for i in range(interpolation_steps):
        alpha = (i+1) / interpolation_steps
        interpolated_position = [(1 - alpha) * prev + alpha * desired for prev, desired in zip(prev_positions, desired_positions)]
        forces = [500.0] * len(interpolated_position)

        p.setJointMotorControlArray(armId, 
                                    range(len(interpolated_position)),
                                    p.POSITION_CONTROL,
                                    targetPositions=interpolated_position,
                                    forces=forces)


        p.stepSimulation()
        time.sleep(1./240.)


def main():
    current_joint = 0    #Id of joint we are currently controlling

    intermediate_pose = get_joint_positions() #Pose we are currently working on. Since we manipulate one joint
                                              # at a time, we use intermediate_pose to save previous manipulated
                                              # joints that we havent saved as a keyframe yet

    trace = [get_joint_positions()]    #The trace so far. trace[-1] is the last keyframe added

    while not controller.Start and not keyboard.is_pressed("q"): #use start button to exit and save trace so far

        if controller.LeftBumper: #move to previous joint
            current_joint = Clamp(current_joint - 1, 0, numJoints - 1)

        if controller.RightBumper: #move to next joint
            current_joint = Clamp(current_joint + 1, 0, numJoints - 1)

        if controller.Y: #cancel this joint's movement but keep other joints, revert to intermediate_pose
            set_joint_positions(get_joint_positions(), intermediate_pose)

        if controller.B: #cancel this frame, revert to previous keyframe in trace
            set_joint_positions(get_joint_positions(), trace[-1])
            intermediate_pose = get_joint_positions()

        if controller.X: #save this joint to intermediate_pose
            intermediate_pose = get_joint_positions()

        if controller.A or keyboard.is_pressed("space"): #save this position as a key frame
            trace.append(get_joint_positions())
            intermediate_pose = get_joint_positions()
            print("!!!!!!!!!!!!!!!!!!")

        joints = get_joint_positions()
        joints[current_joint] += controller.LeftJoystickY
        set_joint_positions(get_joint_positions(), joints, interpolation_steps=5)

        print(f"Currently controlling joint {current_joint}")
            

        p.stepSimulation()
        time.sleep(1./240.)


    p.disconnect()

    
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace.pkl")
    with open(path, 'wb') as f:
        pickle.dump(trace, f)

    print("")
    print(trace)


if __name__ == "__main__":
    main()


