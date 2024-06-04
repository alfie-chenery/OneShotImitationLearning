
# One Shot Imitation Learning

## trace_generation\

This directory is used to record new demonstrations for the robot agent.

### Running the file

To record a new demonstration you will need a video game controller connected to your device.
This controller must be connected before running the desired python file, otherwise
an exception will raise and exit the code.

You should run `python .\trace_generation\generate_trace_eef.py`
This file controlls the robot by the position and orientation of the end effector.

You SHOULD NOT run `python .\trace_generation\generate_trace_joints.py`
This file is depreciated as the main code no longer expects demonstrations in the format provided by this file
It remains purely for documentation

After creating a demonstration 4 files will be generated:
- `demonstration.pkl` is a pickle file containing an encoding of the demonstration as a list of end effector positions, orientations and gripper states
- `demonstration-rgb.jpg` is an RGB image of the environment the demonstration was provided in
- `demonstration-depth.pkl` is a pickle file containing the depth data which accompanies the RGB image
- `demonstration-vm.pkl` is a pickle file containing the view matrix of the camera as the picture was taken

These files can be renamed so long as they all still have a common substring. The "demonstration" part can be renamed as desired
but the file extensions and any part succeeding a "-" must not be altered.
The filename should not contain dashes "-" except for the ones already included at the end of the name. The filename may include underscores "_"

### Controller button mapping

