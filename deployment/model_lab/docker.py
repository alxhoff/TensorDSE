import os
from deployment.utils.utils import RunTerminalCommand
from deployment.model_lab.optimize import SUB_DIR

class Docker:
    def __init__(self, name):
        self.name = name
        self.Start()
        self.CreateSubDir()

    def RunDockerCommand(self, command: str):
        log.info(RunTerminalCommand('sudo', 'docker', 'exec', '-ti', 'debian-docker', 'sh', '-c', command, save_output=True))

    def CreateSubDir(self):
        create_subdir_command = "mkdir /home/sub"
        self.RunDockerCommand(create_subdir_command)

    def Start(self):
        try:
            log.info("Starting Docker Container: {0}".format(self.name))
            log.info(RunTerminalCommand("sudo", "docker", "start", self.name, save_output=True))
            log.info("OK\n")
        except Exception as e:
            log.error("Failed to start Docker Container! Potential Cause: {}".format(str(e)))

    def Copy(self, filename: str, src: str, trg: str):
        if [src, trg] == ["host", "docker"]:
            src_path = os.path.join(SUB_DIR, "tflite", filename)
            trg_path = "{0}:{1}".format(self.name, os.path.join("/home/sub", filename)) 
        else:
            src_path = "{0}:{1}".format(self.name, os.path.join("/home/sub", filename))
            trg_path = os.path.join(SUB_DIR, "tflite", filename)
             
        RunTerminalCommand("sudo", "docker", "cp", src_path, trg_path)
    
    def Compile(self, filename: str):
        compiling_command = "edgetpu_compiler -o /home/sub/ -s /home/sub/{0}".format(filename)
        self.RunDockerCommand(compiling_command)

    def Clean(self):
        command = "find -type f -name '*submodel_tpu*' -delete"
        self.RunDockerCommand(command)

    def __del__(self):
        command = "rm -rf /home/sub"
        self.RunDockerCommand(command)
        
