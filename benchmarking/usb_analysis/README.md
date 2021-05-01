# Usb Analysis
This README.md will contain instructions on how to setup, use and interact with this
project/tools.

# Docker and Wireshark
## Dependencies
```
docker
wireshark
```

## Installation
Based on an Arch system.

```
sudo pacman -S docker wireshark wireshark-cli
sudo systemctl enable docker.service
sudo systemctl start docker.service --now
```
### Groups
To enable the use of docker without sudo.
```
sudo usermod -aG docker <user>
```
Same for wireshark. Also gives it access to USB without the need for root priveleges.
```
sudo usermod -aG wireshark $USER
```

### Docker Setup
Commands will not contain sudo, since the user has been added to the docker group.

Pull down the debian image.
```
docker pull debian
```

Create and run the debian docker. The priveleged flag will allow the docker to interact
with the system's USB. 

Recommended name is **debian-docker** as this is used within my
scripts to interact with the docker.  If one wishes a different name, just search (`grep
-r`) for the single instance where 'debian-docker' is defined and change it.
```
docker run --name <docker_name> --privileged -v /dev/bus/usb:/dev/bus/usb -it -d debian
docker start <docker_name>
docker attach <docker_name>
```
Update packages and install/setup the bare *essentials*.
```
apt-get update
apt-get install vim sudo # Vim Long and Prosper
```

Theoretically, one could use the docker as root always, but I had encountered hiccups here
and there when doing so. It could also well be due to my own  mistakes, but using the docker with
a user worked flawlessly.
```
adduser deb # The user is called deb (as in Debian.. lol)
    <create password>

usermod -aG sudo deb

passwd root
    <create password>

login deb
    <insert password>
```
### Coral Setup
Install project-related packages and tools (The Meat and the Potatoes).

The `Gnupg` package may also not be needed, but Ive encountered issued
without it.
```
sudo apt-get install gnupg curl git python3 python3-pip
```
Installing the coral edge tpu compiler and library. Please check [here](https://coral.ai/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime) to confirm if installing these packages for Ubuntu/Debian has changed.
```
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt-get update
sudo apt-get install libedgetpu1-std edgetpu-compiler
```

### Docker Usage
```
1. docker start <docker_name>
2. docker attach <docker_name>
3. docker info
4. docker images
5. docker ps -a
```
1. Starts docker. 
2. Attaches to it, meaning "ssh's" into the docker installation.
3. Get general information of the dockers within the system.
4. Displays docker images installed (Debian, Arch BTW, etc)
5. Looks at the status of dockers within the systems, useful to visualize if docker has
   started/is stopped.

### Coral Usage
#### Coral 'Hello World'
**Setup:**

Look again [here](https://coral.ai/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime) to ensure this is still the workflow for the *Hello World* for the Coral Edge TPU.
```
mkdir coral && cd coral
git clone https://github.com/google-coral/tflite.git
cd tflite/python/examples/classification
bash install_requirements.sh

sudo pip3 install Pillow
sudo pip3 install tensorflow
sudo apt-get install python3-pycoral
```

**Run:**

```
mkdir coral && cd coral
git clone https://github.com/google-coral/tflite.git
cd tflite/python/examples/classification
bash install_requirements.sh

sudo pip3 install Pillow
sudo pip3 install tensorflow
sudo apt-get install python3-pycoral

python3 classify_image.py \
--model models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite \
--labels models/inat_bird_labels.txt \
--input images/parrot.jpg
```
Now wait and see :).

#### Coral Compilation
```
edgetpu_compiler [options] model...
edgetpu_compiler --help

Ex:
    edgetpu_compiler -s path/to/model
```

### Wireshark
Before use of it's GUI application, you have to run `sudo modprobe usbmon` to load the USB
module onto the Kernel.

# Python Virtual Environment
This is an attempt to summarize the python packages needed to interact/use the
tools within this project. There is a `requirements.txt` text file in the folder `requirements`. There all the packages needed are listed with use of `pip list > requirements.txt`, provided you have activated the virtual environment. Theoretically one could install them either one by one or with `pip install -r <path_to_requirements>`. 
<br></br>
That hasn't worked for me much, so I will go through the creation of what I believe is a working environment for this work.
## Dependencies
Meta-packages. Their 'pip names' may be different. If I have forgotten something a
corresponding error will be thrown during runtime, so please just add the missing package
here.
```
pyshark     # Wireshark Python wrapper
pillow
tensorflow
tensorboard
numpy
statistics
scipy
matplotlib
ipython     # Optional for general interactive use
ipdb3       # Opitonal for debugging (Directly in Terminal, Feature-Rich and scriptable)
```

## Setup
1. `git clone <this_repo>`
2. `cd TensorDSE/benchmarking/usb_analysis`
3. `mkdir venv`
4. `python -m venv ./venv` 

    No  need for conda, if not wished for. It will basically contain some IDE's, 
 manages virtual environments for you and pre-installs some packages.

    **Important**:
    
    Depending on the python version of your system it will create a virtual environment
    according to that version of python. This wouldn't be much of a problem, but in case
    you wish to work with `Python 3.8`, which was the case during my work, you would have
    to install it separately.

    On Arch this is done by running `yay -S python38` (or any other AUR Helper). Once
    installed run the same command from 4, but substitute python with `/usr/bin/python3.8`
    instead.

5. `source ./venv/bin/activate` to 'enter' this virtual environment within the open
   terminal. Now everything you do here will be using the python version and packages
   within this virtual environment.

6. `pip install tensorflow tensorboard pillow numpy statistics scipy matplotlib ipython
   ipdb3`

*Voilà*.

# Project structure
```
Located at benchmarking/usb_analysis

├── analyze.py         (Executable)
├── cmpile.py          (Executable)
├── convert.py         (Executable)
├── deploy.py          (Executable)
├── docker.py          (Library)
├── models/            (Directory - Contains sub-folders with different models)
├── parse.py           (Not Implemented Fully - JSON Alternative to Python Helper Methods)
├── plot.py            (Executable)
├── results/           (Directory - Contains sub-folders with results from different stages)
├── schema/            (Directory - Contains schema.fbs)
└── utils.py           (Library)
```
Executable files mean that they can be called onto as a stand-alone script. Well all
python scripts are by definition executable as they are interpreted directly by the python
interpreter, but the idea here is to differentiate files that can be called on their own
to files who serve as libraries of sorts to be imported by other scripts.

Each file/script name should indicate the general purpose of said file/script.
