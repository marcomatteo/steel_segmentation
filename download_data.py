import subprocess
from sys import platform
from pathlib import Path

def run_cmd(cmd):
    """From: https://stackoverflow.com/questions/14894993/running-windows-shell-commands-with-python"""
    result = []
    process = subprocess.Popen(cmd,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    for line in process.stdout:
        result.append(line)
    errcode = process.returncode
    for line in result:
        print(line)
    if errcode is not None:
        raise Exception('cmd %s failed, see above for details', cmd)

if __name__ == "__main__":
    if platform == "linux" or platform == "linux2":
        # Check kaggle.json file
        path = Path()
        kaggle_path = path / "kaggle.json"
        if not kaggle_path.is_file():
            raise FileExistsError(f"File {kaggle_path.name} doesn't exist! Download it from Kaggle.")
        # Authentication with the Kaggle API
        run_cmd("mkdir ~/.kaggle")
        run_cmd("cp ../kaggle.json ~/.kaggle/kaggle.json")
        run_cmd("chmod 600 ~/.kaggle/kaggle.json")
        # Download the data
        run_cmd(f"kaggle competitions download -c severstal-steel-defect-detection -p {path}")
        run_cmd("mkdir data")
        run_cmd(f"unzip -q -n {path}/severstal-steel-defect-detection.zip -d {path}")

