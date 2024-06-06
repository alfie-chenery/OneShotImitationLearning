import pybullet as p
import environment
import numpy as np
import pickle
import os
from controller import Controller

def main():
    controller = Controller()
    env = environment.FrankaArmEnvironment()

    joystickSensitivity = 0.05 #scalar to slow down how much the joysticks move
    triggerSensitivity = 0.1

    gripperClosed = True
    trace = [(*env.robotGetEefPosition(), gripperClosed)]    #The trace so far. trace[-1] is the last keyframe added
    saveTrace = True
    _, _, rgb, depth, _, vm = env.robotGetCameraSnapshot()
    debounce = False #button debounce, prevents holding button affecting multiple times
    wireframe = False
    cameraMode = False #False to control eef pos and orn, False to control camera


    while True:

        if controller.B and not debounce: #cancel this frame, revert to previous keyframe in trace
            pos, orn, gripper = trace[-1]
            gripperClosed = gripper
            env.robotSetEefPosition(pos, orn)
            env.robotCloseGripper() if gripperClosed else env.robotOpenGripper()
            debounce = True
            print("reverted to last keyframe")

        if controller.A and not debounce: #save this position as a key frame
            trace.append((*env.robotGetEefPosition(), gripperClosed))  #unpack (pos, orn) tuple and make new tuple (pos, orn, gripper)
            debounce = True
            print("saved current as keyframe")

        if controller.DPadUp and not debounce:
            wireframe = not wireframe
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, int(wireframe))
            debounce = True
            print("toggled wireframe " + ("on" if wireframe else "off"))

        if controller.DPadDown and not debounce:
            _, _, rgb, depth, _, vm = env.robotGetCameraSnapshot()
            debounce = True
            print("taken snapshot")

        if controller.DPadLeft and not debounce:
            if not cameraMode:
                gripperClosed = not gripperClosed
                env.robotCloseGripper() if gripperClosed else env.robotOpenGripper()
                debounce = True
                print(("closed" if gripperClosed else "openned") + " gripper")

        if controller.DPadRight and not debounce:
            cameraMode = not cameraMode
            debounce = True
            print("controlling " + ("camera" if cameraMode else "eef"))

        
        if cameraMode:
            cameraDist, cameraYaw, cameraPitch, cameraTarget = env.getDebugCameraState()
            cameraYaw += joystickSensitivity * controller.RightJoystickX
            cameraPitch += joystickSensitivity * controller.RightJoystickY
            cameraDist += triggerSensitivity * 0.1 * (controller.LeftBumper - controller.RightBumper)
            cameraTarget = list(cameraTarget)
            cameraTarget[0] += joystickSensitivity * controller.LeftJoystickX
            cameraTarget[1] += joystickSensitivity * controller.LeftJoystickY
            env.setDebugCameraState(cameraDist, cameraYaw, cameraPitch, cameraTarget)

        else:
            dPos = [joystickSensitivity * controller.LeftJoystickX, 
                    joystickSensitivity * controller.LeftJoystickY, 
                    joystickSensitivity * (controller.RightTrigger - controller.LeftTrigger)]
            dOrn = [joystickSensitivity * controller.RightJoystickX, 
                    joystickSensitivity * controller.RightJoystickY, 
                    joystickSensitivity * (controller.LeftBumper - controller.RightBumper)]
            dQuat = p.getQuaternionFromEuler(dOrn)
            rotationMatrix = np.array(p.getMatrixFromQuaternion(dQuat)).reshape((3,3))

            env.robotMoveEefPosition(dPos, rotationMatrix, interpolationSteps=10)


        if controller.nonePressed(): #once button released, disable debounce, allowing a new button to be registered
            debounce = False

        if controller.Start: #save and quit
            saveTrace = True
            break

        if controller.Back: #quit without saving
            saveTrace = False
            break            

        env.stepEnv()


    print("")
    print(trace)
    
    if saveTrace:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(dir_path, "demonstration.pkl")
        with open(path, 'wb') as f:
            pickle.dump(trace, f)

        env.robotSaveCameraSnapshot("demonstration", dir_path, rgb, depth, vm)
        print("\nSuccessfully saved trace")
    else:
        print("\nQuit without saving trace")
    
    env.closeEnv()

    
if __name__ == "__main__":
    main()


