
# One Shot Imitation Learning

## trace_generation\

This directory is used to record new demonstrations for the robot agent.

### Running the file

To record a new demonstration you will need a video game controller connected to your device.
This controller must be connected before running the desired python file, otherwise 
an exception will raise and exit the code.

You should run `python .\trace_generation\generate_trace_eef.py`
This file controls the robot by the position and orientation of the end effector and gripper state.

You SHOULD NOT run `python .\trace_generation\generate_trace_joints.py`
This file is depreciated as the main code no longer expects demonstrations in the format provided by this file. It remains purely for documentation.

You can test this demonstration by running `python .\trace_generation\mimic_trace.py`. This file will execute the most recently created demonstration. If this robot fails to execute the demonstration then this is a sign that the demonstration will not work in `main.py`. Since the demonstrations are given during a live environment, it is possible objects can move and interact with each other. Particularly if you make heavy use of undoing movements during the demonstration, then this may lead to an inconsistent demonstration which does not complete the desired task. In this case I recommend creating a new demonstration.

You can modify the environment to move, rotate and change the objects used in the demonstration. I recommend modifying `main_code\environment.py` and then running `make environment` or `sh Makefile.sh`. This way the modified environment will also be available to the agent during testing.
Alternatively you may modify `trace_generation\environment.py` but these changes will not be reflected when `main.py` is run, and will be overwritten by `make environment` or `sh Makefile.sh`.

 
After creating a demonstration 4 files will be generated:
-  `demonstration.pkl` is a pickle file containing an encoding of the demonstration as a list of end effector positions, orientations and gripper states
-  `demonstration-rgb.jpg` is an RGB image of the environment the demonstration was provided in
-  `demonstration-depth.pkl` is a pickle file containing the depth data which accompanies the RGB image
-  `demonstration-vm.pkl` is a pickle file containing the view matrix of the camera as the picture was taken

These files can be renamed so long as they all still have a common substring. The "demonstration" part can be renamed as desired, the rest of the file names should be unaltered.
The file name should not contain dashes "-" or dots "." except for the ones already included at the end of the name. The file name may include underscores "_"

Once renamed these files can be copied or moved into `main_code\demonstrations\`. Once this is done they will be available to the agent as demonstrations to follow when encountering new environments.
Once the files are renamed `mimic_trace.py` will no longer recognise them.

The agent may choose not to use this new demonstration if it believes another demonstration is closer to the current environment. If you use the exact same environment as when the demonstration was given, the agent should select the new demonstration.

### Controller button mapping

The button mapping will be given for an Xbox and PlayStation controller. Most controllers will work, but the buttons may have different names.

- **A / &#10005** - Save the current end effector position, orientation and gripper state as a new key frame in the trace
- **B / &#9675** - Revert the robot to the previous saved key frame (or initial position if none saved)
- **Dpad Up** - Toggle wireframe view
- **Dpad Down** - Override demonstration image. An image is automatically saved at the start of the program. However, you may override this with a new one using this button. The camera is attached to the end effector and faces along the end effector Z axis (blue). The object of interest in the demonstration must be clearly visible in the image, otherwise the agent will be unable to use this demonstration effectively.
- **Dpad Left** - Toggle gripper open and closed
- **Dpad Right** - Toggle between camera control mode and end effector control mode. Initially in end effector control mode
- **Left Joystick**
	- **In end effector control mode**: Moves the end effector position in the X and Y world axis using the X and Y axis of the joystick
	- **In camera control mode**: Moves the camera target position in the X and Y world axis using the X and Y axis of the joystick
- **Right Joystick**
	- **In end effector control mode**: Rotates the end effector Roll about the X world axis using the X axis of the joystick. Rotates the end effector Pitch about the Y world axis using the Y axis of the joystick
	- **In camera control mode**: Rotates the camera Yaw about the Z world axis using the X axis of the joystick. Rotates the camera Pitch about the Y world axis using the Y axis of the joystick
- **Left Trigger / L2**
 	- **In end effector control mode**: Moves the end effector position in the Z world axis in the negative direction
	- **In camera control mode**: Has no effect
- **Right Trigger / R2**
 	- **In end effector control mode**: Moves the end effector position in the Z world axis in the positive direction
	- **In camera control mode**: Has no effect
- **Left Bumper/ L1**
 	- **In end effector control mode**: Rotates the end effector Yaw about the Z world axis in the positive direction
	- **In camera control mode**: Increases the distance between the camera and the target position (zooms out)
- **Right Bumper/ R1**
 	- **In end effector control mode**: Rotates the end effector Yaw about the Z world axis in the negative direction
	- **In camera control mode**: Decreases the distance between the camera and the target position (zooms in)
- **Start / Options** - Saves the trace as the 4 files mentioned above. Once saved exits the program. This will overwrite any other demonstrations in the directory which have not been renamed
- **Back / Share** - Quits the program, does not save the demonstration
