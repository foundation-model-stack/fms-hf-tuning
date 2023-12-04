import os
from aim.hugging_face import AimCallback

def get_aimstack_callback():
    # Initialize a new run
    aim_server = os.environ.get('AIMSTACK_SERVER')
    aim_db = os.environ.get('AIMSTACK_DB')
    aim_experiment = os.environ.get('AIMSTACK_EXPERIMENT')
    if aim_experiment is None:
        aim_experiment = ""

    if aim_server:
        aim_callback = AimCallback(repo='aim://'+aim_server+'/', experiment=aim_experiment)
    if aim_db:
        aim_callback = AimCallback(repo=aim_db, experiment=aim_experiment)
    else:
        aim_callback = AimCallback(experiment=aim_experiment)

    return aim_callback
