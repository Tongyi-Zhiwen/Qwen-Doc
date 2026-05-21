import os
import subprocess
import sys


def main():
    argv = sys.argv[1:]
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(cur_dir, "app.py")
    cmd = [
        "streamlit",
        "run",
        app_path,
        "--server.fileWatcherType",
        "none",
        "--server.headless",
        "true",
    ]
    if argv:
        cmd += ["--"] + argv
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
