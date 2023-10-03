class Log:
    def __init__(self, file):
        import os
        self.file = file
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "w") as f:
            f.write("LOG\n")
            f.write("------------------\n")

    def info(self, msg:str):
        with open(self.file, "a") as f:
            f.write(f"[INFO]: {msg}\n")

    def warning(self, msg:str):
        with open(self.file, "a") as f:
            f.write(f"[WARNING]: {msg}\n")

    def error(self, msg:str):
        with open(self.file, "a") as f:
            f.write(f"[ERROR]: {msg}\n")