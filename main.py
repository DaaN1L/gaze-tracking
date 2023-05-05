import logging as log
from src.utils import resize_image

import hydra
from omegaconf import DictConfig, OmegaConf
from src.app import App


def deploy(cfg: DictConfig):
    app = App(cfg, "Gaze tracker")
    app.mainloop()

@hydra.main(config_path="conf", config_name="config")
def run_model(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    deploy(cfg)


if __name__ == "__main__":
    run_model()
