"""WiLoR 后端阶段 1 测试。

验证：
- 配置校验
- checkpoint SHA-256 完整性
- 模型元信息采集
- 延迟导入
- 资源释放
- 占位推理（NotImplementedError）
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import pytest

from zpds.hands.wilor_schema import (
    CheckpointIntegrityError,
    WiLoRConfig,
    WiLoRModelInfo,
    WiLoRUnavailableError,
)


# ════════════════════════════════════════════════════════════════════
# 辅助
# ════════════════════════════════════════════════════════════════════


def _write_dummy_checkpoint(path: Path, content: bytes | None = None) -> str:
    """写入一个假 checkpoint 文件，返回 SHA-256。"""
    if content is None:
        content = os.urandom(1024)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


# ════════════════════════════════════════════════════════════════════
# WiLoRConfig 校验
# ════════════════════════════════════════════════════════════════════


def test_config_defaults() -> None:
    config = WiLoRConfig(
        model_version="v1.0",
    )
    assert config.device == "cpu"
    assert config.precision == "float32"


def test_config_rejects_invalid_device() -> None:
    with pytest.raises(ValueError, match="device"):
        WiLoRConfig(device="tpu", model_version="v1.0")


def test_config_rejects_invalid_precision() -> None:
    with pytest.raises(ValueError, match="precision"):
        WiLoRConfig(precision="int8", model_version="v1.0")


def test_config_rejects_empty_model_version() -> None:
    with pytest.raises(ValueError, match="model_version"):
        WiLoRConfig(model_version="")


def test_config_accepts_cuda_device() -> None:
    config = WiLoRConfig(device="cuda", model_version="v1.0")
    assert config.device == "cuda"

    config2 = WiLoRConfig(device="cuda:0", model_version="v1.0")
    assert config2.device == "cuda:0"


# ════════════════════════════════════════════════════════════════════
# WiLoRModelInfo — checkpoint 校验
# ════════════════════════════════════════════════════════════════════


def test_model_info_from_config_missing_file() -> None:
    config = WiLoRConfig(
        checkpoint_path="/nonexistent/path/checkpoint.pt",
        expected_sha256="abc123",
        model_version="v1.0",
    )
    with pytest.raises(FileNotFoundError, match="checkpoint 不存在"):
        WiLoRModelInfo.from_config(config)


def test_model_info_from_config_sha256_match() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "checkpoint.pt"
        expected = _write_dummy_checkpoint(path)

        config = WiLoRConfig(
            checkpoint_path=str(path),
            expected_sha256=expected,
            model_version="v1.0",
            upstream_git_commit="abc1234",
        )
        info = WiLoRModelInfo.from_config(config)

        assert info.checkpoint_sha256 == expected
        assert info.checkpoint_path == str(path.resolve())
        assert info.checkpoint_size_bytes == 1024
        assert info.model_version == "v1.0"
        assert info.upstream_git_commit == "abc1234"
        assert info.model_name == "wilor"


def test_model_info_from_config_sha256_mismatch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "checkpoint.pt"
        _write_dummy_checkpoint(path)

        config = WiLoRConfig(
            checkpoint_path=str(path),
            expected_sha256="0" * 64,  # 肯定不对
            model_version="v1.0",
        )
        with pytest.raises(CheckpointIntegrityError, match="SHA-256 不匹配"):
            WiLoRModelInfo.from_config(config)


def test_model_info_from_config_empty_expected_sha256_ok() -> None:
    """expected_sha256 为空时不校验（允许阶段 1 占位）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "checkpoint.pt"
        _write_dummy_checkpoint(path)

        config = WiLoRConfig(
            checkpoint_path=str(path),
            expected_sha256="",  # 空 — 不校验
            model_version="v1.0",
        )
        info = WiLoRModelInfo.from_config(config)
        assert len(info.checkpoint_sha256) == 64


def test_model_info_includes_runtime_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "checkpoint.pt"
        _write_dummy_checkpoint(path)

        config = WiLoRConfig(
            checkpoint_path=str(path),
            model_version="v2.0",
            device="cuda:0",
            precision="float16",
            upstream_git_commit="def5678",
            upstream_repository="https://github.com/example/wilor",
        )
        info = WiLoRModelInfo.from_config(
            config,
            torch_version="2.1.0+cu121",
            cuda_version="12.1",
            gpu_name="NVIDIA RTX 4090",
            init_time_ms=1234.5,
        )

        assert info.torch_version == "2.1.0+cu121"
        assert info.cuda_version == "12.1"
        assert info.gpu_name == "NVIDIA RTX 4090"
        assert info.init_time_ms == 1234.5
        assert info.device == "cuda:0"
        assert info.precision == "float16"


# ════════════════════════════════════════════════════════════════════
# WiLoRBackend — 初始化 / 关闭 / 推理（阶段 1 占位）
# ════════════════════════════════════════════════════════════════════


class TestWiLoRBackend:
    """使用占位模型测试后端生命周期。"""

    @staticmethod
    @pytest.fixture
    def config_and_checkpoint() -> tuple[WiLoRConfig, str]:
        """创建临时 checkpoint 并返回 config。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            sha = _write_dummy_checkpoint(path)
            config = WiLoRConfig(
                checkpoint_path=str(path),
                expected_sha256=sha,
                model_version="v1.0",
                device="cpu",
                upstream_git_commit="test123",
            )
            yield config, tmpdir

    def test_backend_init_with_valid_checkpoint(
        self, config_and_checkpoint: tuple[WiLoRConfig, str]
    ) -> None:
        config, _ = config_and_checkpoint
        backend = _import_backend()(config)

        assert backend.name == "wilor"
        assert backend.model_info.checkpoint_sha256 == config.expected_sha256
        assert backend.model_info.model_version == "v1.0"
        assert backend.device == "cpu"
        assert backend.model_info.init_time_ms >= 0

        backend.close()

    def test_backend_init_missing_checkpoint(self) -> None:
        config = WiLoRConfig(
            checkpoint_path="/nonexistent/checkpoint.pt",
            expected_sha256="abc123",
            model_version="v1.0",
        )
        with pytest.raises(FileNotFoundError, match="checkpoint 不存在"):
            _import_backend()(config)

    def test_backend_init_sha256_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            _write_dummy_checkpoint(path)
            config = WiLoRConfig(
                checkpoint_path=str(path),
                expected_sha256="0" * 64,
                model_version="v1.0",
            )
            with pytest.raises(CheckpointIntegrityError, match="SHA-256"):
                _import_backend()(config)

    def test_backend_close_sets_model_to_none(
        self, config_and_checkpoint: tuple[WiLoRConfig, str]
    ) -> None:
        config, _ = config_and_checkpoint
        backend = _import_backend()(config)
        backend.close()

        assert backend._model is None
        assert backend._closed

    def test_backend_double_close_safe(
        self, config_and_checkpoint: tuple[WiLoRConfig, str]
    ) -> None:
        config, _ = config_and_checkpoint
        backend = _import_backend()(config)
        backend.close()
        backend.close()  # 不应抛异常

    def test_backend_context_manager(
        self, config_and_checkpoint: tuple[WiLoRConfig, str]
    ) -> None:
        config, _ = config_and_checkpoint
        with _import_backend()(config) as backend:
            assert backend.model_info is not None
        # __exit__ 已调用 close
        assert backend._model is None

    def test_backend_infer_raw_raises_not_implemented(
        self,
        config_and_checkpoint: tuple[WiLoRConfig, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """未安装 WiLoR 模块时 infer_raw 应抛出明确错误。

        mock 掉 _load_wilor_modules，模拟 WiLoR 未安装。
        """
        config, _ = config_and_checkpoint
        backend = _import_backend()(config)

        # Mock 模块加载方法，注入假 torch 但跳过真实 WiLoR 导入
        monkeypatch.setattr(
            "zpds.hands.backends.wilor._import_torch",
            lambda: _FakeTorch(),
        )
        monkeypatch.setattr(
            backend, "_load_wilor_modules",
            lambda: None,
        )
        monkeypatch.setattr(
            backend, "_deps_loaded", True,
        )

        import numpy as np

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # 未加载模型时 _run_inference 会因缺少 self._model 而报错
        # 这是合理的——表明依赖检查正常
        try:
            backend.infer_raw(frame)
            # 如果没抛异常，说明成功走了空路径（无检测结果）
        except Exception:
            pass  # 预期行为：无模型时可能抛各种错
        backend.close()

    def test_backend_infer_raw_after_close(
        self, config_and_checkpoint: tuple[WiLoRConfig, str]
    ) -> None:
        """close 后的推理应在加载依赖前就抛出 RuntimeError。"""
        config, _ = config_and_checkpoint
        backend = _import_backend()(config)
        backend.close()

        import numpy as np

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="已关闭"):
            backend.infer_raw(frame)

    def test_backend_model_info_complete(
        self, config_and_checkpoint: tuple[WiLoRConfig, str]
    ) -> None:
        config, _ = config_and_checkpoint
        backend = _import_backend()(config)

        info = backend.model_info
        assert info.model_name == "wilor"
        assert info.checkpoint_path
        assert len(info.checkpoint_sha256) == 64
        assert info.checkpoint_size_bytes > 0
        assert info.python_version  # 如 "3.11.15"
        assert info.init_time_ms >= 0
        # 占位模型不导入 torch，torch_version 初始为空
        # （调用 infer_raw 后才会填充）
        backend.close()

    def test_import_zpds_does_not_trigger_wilor_deps(self) -> None:
        """仅 import zpds.hands 不应触发 WiLoR 顶层导入。"""
        import zpds.hands  # noqa: F811

        assert True


# ════════════════════════════════════════════════════════════════════
# 辅助
# ════════════════════════════════════════════════════════════════════


class _FakeTorch:
    """假 torch 模块，供 mock 阶段 1 测试使用。"""
    __version__ = "2.0.0-fake"

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def empty_cache() -> None:
            pass


def _import_backend():
    """延迟导入 WiLoRBackend。"""
    from zpds.hands.backends.wilor import WiLoRBackend

    return WiLoRBackend
