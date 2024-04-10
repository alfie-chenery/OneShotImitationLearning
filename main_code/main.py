import environment
import keyboard

env = environment.FrankaArmEnvironment()

while True:
    if keyboard.is_pressed('space'):
            img = env.robotGetCameraSnapshot()
            #self.save_snapshot(img)
    env.stepEnv()
