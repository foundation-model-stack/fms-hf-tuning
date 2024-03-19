from accelerate.commands.launch import launch_command_parser, launch_command
import json
import os
import base64
import pickle


def txt_to_obj(txt):
    base64_bytes = txt.encode("ascii")
    message_bytes = base64.b64decode(base64_bytes)
    try:
        # If the bytes represent JSON string
        return json.loads(message_bytes)
    except UnicodeDecodeError:
        # Otherwise the bytes are a pickled python dictionary
        return pickle.loads(message_bytes)

def main():
    json_configs = {}
    json_path = os.getenv("SFT_TRAINER_CONFIG_JSON_PATH")
    json_env_var = os.getenv("SFT_TRAINER_CONFIG_JSON_ENV_VAR")

    if json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            json_configs = json.load(f)

    elif json_env_var:
        json_configs = txt_to_obj(json_env_var)

    # parse multiGPU args
    print("json_configs[multiGPU]", json_configs["multiGPU"])

    multigpu_args = []
    if json_configs.get("multiGPU"):
        for key, val in json_configs["multiGPU"].items():
            multigpu_args.append(f"--{key}")
            multigpu_args.append(str(val))
    
    # TODO: add training_script // rest of args -- no args 
    multigpu_args.append("--training_script /app/launch_training.py")    
    
    print("multigpu_args", multigpu_args)
    parser = launch_command_parser()
    args = parser.parse_args(args=multigpu_args)
    print(args)
    launch_command(args)

if __name__ == "__main__":
    main()