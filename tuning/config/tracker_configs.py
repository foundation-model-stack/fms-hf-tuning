# Standard
from dataclasses import dataclass


@dataclass
class AimConfig:
    # Name of the experiment
    experiment: str = None
    # 'aim_repo' can point to a locally accessible directory (e.g., '~/.aim') or a remote repository hosted on a server.
    # When 'aim_remote_server_ip' or 'aim_remote_server_port' is set, it designates a remote aim repo.
    # Otherwise, 'repo' specifies the directory, with a default of None representing '.aim'.
    # See https://aimstack.readthedocs.io/en/latest/using/remote_tracking.html for documentation on Aim remote server tracking.
    aim_repo: str = ".aim"
    aim_remote_server_ip: str = None
    aim_remote_server_port: int = None
    # Location of where run_hash is exported, if unspecified this is output to
    # training_args.output_dir/.aim_run_hash if the output_dir is set else not exported.
    aim_run_hash_export_path: str = None
