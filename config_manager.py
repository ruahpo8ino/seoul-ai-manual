import yaml
from box import Box

class ConfigLoader:
    """
    구성 파일을 로드하고 설정을 반환하는 클래스입니다.
    """

    def __init__(self, config_path="config.yaml"):
        """
        ConfigLoader를 초기화하고 구성 파일을 로드합니다.

        Args:
            config_path (str): 구성 파일의 경로 (기본값은 "config.yaml")
        """
        self.config_path = config_path
        with open(self.config_path) as f:
            config_yaml = yaml.load(f, Loader=yaml.FullLoader)
            self.config = Box(config_yaml)

    def get_config(self):
        """
        로드된 구성 설정을 반환합니다.

        Returns:
            Box: 로드된 설정을 포함하는 Box 객체
        """
        return self.config
