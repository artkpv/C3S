
from omegaconf import DictConfig
import hydra
import eval
import train


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    train.main(cfg)
    eval.main(cfg)

if __name__ == "__main__":
    main()
