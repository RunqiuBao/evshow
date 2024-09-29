def check_cuda():
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("nvidia-smi is executable. Here's the output:")
            print(result.stdout)
            return True
        else:
            print("nvidia-smi returned an error. Here's the error message:")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("nvidia-smi is not found or not installed.")
        return False