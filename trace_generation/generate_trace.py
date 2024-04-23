import pybullet as p
import environment
import pickle
import os
from controller import Controller

def Clamp(val, low, high):
    if val < low:
        return low
    elif val > high:
        return high
    else:
        return val


def main():
    controller = Controller()
    env = environment.FrankaArmEnvironment()

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
    snapshot = None
    debounce = False #button debounce, prevents holding button affecting multiple times
    wireframe = False
    p.setDebugObjectColor(env.robotId, current_joint, objectDebugColorRGB=[255,0,0]) #force set so the first time activating wireframe doesnt behave weird

    while True:

        if controller.LeftBumper and not debounce: #move to previous joint
            p.setDebugObjectColor(env.robotId, current_joint) #no colour resets to default
            current_joint = Clamp(current_joint - 1, 0, env.numJoints - 1)
            p.setDebugObjectColor(env.robotId, current_joint, objectDebugColorRGB=[255,0,0])
            debounce = True
            
        if controller.RightBumper and not debounce: #move to next joint
            p.setDebugObjectColor(env.robotId, current_joint) #no colour resets to default
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
            wireframe = not wireframe
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, int(wireframe))
            debounce = True
            print("toggled wireframe " + ("on" if wireframe else "off"))

        if controller.DPadDown and not debounce:
            _, _, snapshot, _, _ = env.robotGetCameraSnapshot()
            debounce = True
            print("taken snapshot")

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

        if controller.nonePressed(): #once button released, disable debounce, allowing a new button to be registered
            debounce = False

        if controller.Start: #quit without saving
            saveTrace = False
            break

        if controller.Menu: #save and quit
            saveTrace = True
            break
            

        env.stepEnv()


    print("")
    print(trace)
    
    if saveTrace:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(dir_path, "trace.pkl")
        with open(path, 'wb') as f:
            pickle.dump(trace, f)

        env.robotSaveCameraSnapshot("trace_snapshot", dir_path, snapshot)
        print("\nSuccessfully saved trace")
    else:
        print("\nQuit without saving trace")
    
    env.closeEnv()

    
if __name__ == "__main__":
    main()


