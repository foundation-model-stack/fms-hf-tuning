from dataclasses import dataclass

@dataclass
class AimConfig:
    # 'repo' can point to a locally accessible directory (e.g., '~/.aim') or a remote repository hosted on a server.
    # When 'remote_server_ip' or 'remote_server_port' is set, it designates a remote aim repo.
    # Otherwise, 'repo' specifies the directory, with a default of None representing '.aim'.
    repo: str = None
    remote_server_ip: str = None
    remote_server_port: int = None
    # Name of the experiment
    experiment: str = None
    # Location of where run_hash is exported
    run_hash_export_location: str = None
