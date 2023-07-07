import subprocess

def check_git_status():
    try:
        output = subprocess.check_output(['git', 'status'])
        print(output.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_git_status()
