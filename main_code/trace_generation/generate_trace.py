import pybullet as p
from main_code import environment
import time
import pickle
import os
from controller import Controller

controller = Controller()
env = environment.FrankaArmEnvironment()

def Clamp(val, low, high):
    if val < low:
        return low
    elif val > high:
        return high
    else:
        return val


def main():
    cameraMoveSpeed = 0.5
    cameraYaw = 50.0
    cameraPitch = -35.0
    cameraDist = 5.0
    joystickSensitivity = 0.1 #scalar to slow down how much the joysticks move angles

    current_joint = 0    #Id of joint we are currently controlling

    intermediate_pose = env.robotGetJointAngles() #Pose we are currently working on. Since we manipulate one joint
                                                  # at a time, we use intermediate_pose to save previous manipulated
                                                  # joints that we havent saved as a keyframe yet

    trace = [env.robotGetJointAngles()]    #The trace so far. trace[-1] is the last keyframe added
    saveTrace = True
    debounce = False #button debounce, prevents holding button affecting multiple times

    while True:

        if controller.LeftBumper and not debounce: #move to previous joint
            p.setDebugObjectColor(env.robotId, current_joint)
            current_joint = Clamp(current_joint - 1, 0, env.numJoints - 1)
            p.setDebugObjectColor(env.robotId, current_joint, objectDebugColorRGB=[255,0,0])
            debounce = True
            
        if controller.RightBumper and not debounce: #move to next joint
            p.setDebugObjectColor(env.robotId, current_joint)
            current_joint = Clamp(current_joint + 1, 0, env.numJoints - 1)
            p.setDebugObjectColor(env.robotId, current_joint, objectDebugColorRGB=[255,0,0])
            debounce = True
            
        if controller.Y and not debounce: #cancel this joint's movement but keep other joints, revert to intermediate_pose
            env.robotSetJointAngles(intermediate_pose)
            debounce = True
            print("reverted to intermediate pose")

        if controller.B and not debounce: #cancel this frame, revert to previous keyframe in trace
            env.robotSetJointAngles(trace[-1])
            intermediate_pose = env.robotGetJointAngles()
            debounce = True
            print("reverted to last keyframe")

        if controller.X and not debounce: #save this joint to intermediate_pose
            intermediate_pose = env.robotGetJointAngles()
            debounce = True
            print("saved current as intermediate pose")

        if controller.A and not debounce: #save this position as a key frame
            trace.append(env.robotGetJointAngles())
            intermediate_pose = env.robotGetJointAngles()
            debounce = True
            print("saved current as keyframe")

        if controller.DPadUp and not debounce:
            p.setDebugObjectColor(env.robotId, current_joint, objectDebugColorRGB=[255,0,0])
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)
            debounce = True

        if controller.DPadDown and not debounce:
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
            debounce = True

        #eft joystick controls joints
        joints = env.robotGetJointAngles()
        joints[current_joint] += joystickSensitivity * controller.LeftJoystickY
        env.robotSetJointAngles(joints, interpolationSteps=5)

        #right joystick controlls camera
        cameraYaw += cameraMoveSpeed * controller.RightJoystickX
        cameraPitch += cameraMoveSpeed * controller.RightJoystickY
        cameraDist += cameraMoveSpeed * controller.LeftTrigger
        cameraDist -= cameraMoveSpeed * controller.RightTrigger
        p.resetDebugVisualizerCamera(cameraDist, cameraYaw, cameraPitch, [0,0,0])

        if controller.nonePressed():
            debounce = False

        if controller.Start: #quit without saving
            saveTrace = False
            break

        if controller.Menu: #save and quit
            saveTrace = True
            break
            

        p.stepSimulation()
        time.sleep(1./240.)


    p.disconnect()

    print("")
    print(trace)
    
    if saveTrace:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace.pkl")
        with open(path, 'wb') as f:
            pickle.dump(trace, f)

    
if __name__ == "__main__":
    main()


