import os
import pydevd_pycharm


def check_and_start_debugger():
    debug_port = int(os.environ.get("REMOTE_PYCHARM_DEBUG_PORT", 12034))
    if os.environ.get("REMOTE_PYCHARM_DEBUG_SESSION", False):
        pydevd_pycharm.settrace(
            "localhost",
            port=debug_port,
            stdoutToServer=True,
            stderrToServer=True,
            suspend=False,
        )
