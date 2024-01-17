from inputs import get_gamepad
from math import pow
import threading

class Controller():
    def __init__(self):

        self.TRIGGER_MAX_VALUE = pow(2, 8)
        self.JOYSTICK_MAX_VALUE = pow(2, 15)

        self.LeftJoystickX = 0
        self.LeftJoystickY = 0
        self.RightJoystickX = 0
        self.RightJoystickY = 0
        self.LeftJoystickPress = 0
        self.RightJoystickPress = 0

        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0

        self.DPadLeft = 0
        self.DPadRight = 0
        self.DPadUp = 0
        self.DPadDown = 0        

        self.Start = 0 
        self.Menu = 0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()


    def readAll(self): # return the buttons/triggers that you care about in this methode
        return [self.LeftJoystickX,
                self.LeftJoystickY,
                self.RightJoystickX,
                self.RightJoystickY,
                self.LeftJoystickPress,
                self.RightJoystickPress,
                self.LeftTrigger,
                self.RightTrigger,
                self.LeftBumper,
                self.RightBumper,
                self.A,
                self.X,
                self.Y,
                self.B,
                self.DPadLeft,
                self.DPadRight,
                self.DPadUp,
                self.DPadDown,
                self.Start,
                self.Menu]


    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:

                #print(event.code)

                if event.code == 'ABS_X':
                    val = event.state / self.JOYSTICK_MAX_VALUE
                    self.LeftJoystickX = val if abs(val) >= 0.3 else 0 #deadzone to prevent drift
                elif event.code == 'ABS_Y':
                    val = event.state / self.JOYSTICK_MAX_VALUE
                    self.LeftJoystickY = val if abs(val) >= 0.3 else 0 #deadzone to prevent drift
                elif event.code == 'ABS_RX':
                    val = event.state / self.JOYSTICK_MAX_VALUE
                    self.RightJoystickX = val if abs(val) >= 0.3 else 0 #deadzone to prevent drift
                elif event.code == 'ABS_RY':
                    val = event.state / self.JOYSTICK_MAX_VALUE
                    self.RightJoystickY = val if abs(val) >= 0.3 else 0 #deadzone to prevent drift
                elif event.code == 'BTN_THUMBL':
                    self.LeftJoystickPress = event.state
                elif event.code == 'BTN_THUMBR':
                    self.RightJoystickPress = event.state

                elif event.code == 'ABS_Z':
                    self.LeftTrigger = event.state / self.TRIGGER_MAX_VALUE
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = event.state / self.TRIGGER_MAX_VALUE
                elif event.code == 'BTN_TL':
                    self.LeftBumper = event.state
                elif event.code == 'BTN_TR':
                    self.RightBumper = event.state
                    
                elif event.code == 'BTN_SOUTH':
                    self.A = event.state
                elif event.code == 'BTN_NORTH':
                    self.Y = event.state
                elif event.code == 'BTN_WEST':
                    self.X = event.state
                elif event.code == 'BTN_EAST':
                    self.B = event.state

                elif event.code == 'ABS_HAT0X':
                    self.DPadLeft = int(event.state == -1)
                    self.DPadRight = int(event.state == 1)
                elif event.code == 'ABS_HAT0Y':
                    self.DPadUp = int(event.state == -1)
                    self.DPadDown = int(event.state == 1)

                elif event.code == 'BTN_START':
                    self.Start = event.state
                elif event.code == 'BTN_SELECT':
                    self.Menu = event.state


if __name__ == '__main__':
    controller = Controller()
    while True:
        print(controller.readAll())