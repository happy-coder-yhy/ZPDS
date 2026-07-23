"""experience_manifest 管理。"""


class ExperienceManifest:
    """经验标注清单。"""

    def load(self, path: str) -> dict:
        raise NotImplementedError

    def save(self, manifest: dict, path: str) -> None:
        raise NotImplementedError
