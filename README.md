
# One Shot Imitation Learning

Final Year Project
Project repo: https://github.com/alfie-chenery/OneShotImitationLearning
  
  

## Setting up and running the code

The first step to setting up the code is installing all necessary python package imports. To do this I first recommend making a virtual environment to prevent compatibility issues:
Run `python -m venv venv`

Activate the virtual environment :
On Windows run: `.\venv\Scrips\activate`
On Linux run: `source venv/bin/activate`

Install requirements with:
Run `pip install -r requirements.txt`

At this point the main code will be working. However, other folders will likely not work since the majority of codes all refer to the environment.py file, which encapsulates the Pybullet simulation into a easy to use wrapper for this project. Unfortunately Python does not allow imports from parent or sibling directories without very unsafe hacky solutions. Not content with placing all my files in one monolithic directory, the alternative is to distribute this common file to all directories where it is required. This can be achieved by running either the makefile or shell script:
`make environment` or `sh Makefile.sh`

With all the necessary setup the code can now be executed. I recommend running all files from the root repo directory `OneShotImitationLearning`

To run the main code from here run the command:
`python .\main_code\main.py`

You are encouraged to explore the repo, particularly to modify the environment and see how the system handles these changes. Try changing the object in the scene, adding other noise objects, moving and rotating the object etc.


## Directory structure

The repo is separated into different directories which each serve generally different purposes
- `main_code\` is the main directory. It contains `main.py` which is the principal file in the repo.
- `main_code\experiments` contains modified files which were used to run the experiments and generate the results discussed in the report.
- `trace_generation\` contains files regarding generating new demonstrations. You can record new demonstrations using a video game controller. Refer to the README.md file in this directory for documentation.
- `figures\` contains some of the figures used in the report and the files used to generate them.
- `videos\` contains some video logs of runs of the main code. 
- `report\` contains a back-up of the report accompanying this project. It should be up to date with the version submitted. If this is not the case for whatever reason, ignore the contents of this directory, the separately submitted version is the correct version.