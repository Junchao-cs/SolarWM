"""Standard configuration selection with the shared conditioning encoder."""
from deploy.solar_wm.shared.prepare_scene import main as prepare
from .runtime.configuration import load_config


def main():
    prepare(config_loader=load_config)


if __name__ == "__main__":
    main()
