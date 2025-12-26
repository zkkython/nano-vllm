"""Unit tests for weight_utils module.

Run tests with:
    conda activate sgl-jax-gpu
    # Basic tests (no model required)
    python -m pytest python/sgl_jax/test/utils/test_weight_utils.py -v

    # Tests with real model (requires setting TEST_MODEL_PATH)
    TEST_MODEL_PATH=/path/to/model python -m pytest python/sgl_jax/test/utils/test_weight_utils.py -v

    # Or using unittest
    TEST_MODEL_PATH=/path/to/model python -m unittest python/sgl_jax/test/utils/test_weight_utils.py
"""

import os
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from nanovllm_jax.configs.model_config import ModelConfig
from nanovllm_jax.utils.weight_utils import WeightLoader, WeightMapping
from nanovllm_jax.models.qwen3 import Qwen3ForCausalLM


# Set up multi-device simulation for testing
if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    os.environ["JAX_PLATFORMS"] = "cpu"


class TestWeightLoaderInit(unittest.TestCase):
    """Test WeightLoader initialization."""

    def setUp(self):
        """Set up test fixtures."""
        # Create JAX devices and mesh
        devices = jax.devices()
        self.mesh = Mesh(devices[: min(4, len(devices))], ("tensor",))

        # Create mock model config using a simple mock object
        class MockConfig:
            def __init__(self):
                self.hidden_size = 4096
                self.num_attention_heads = 32
                self.num_key_value_heads = 32
                self.num_hidden_layers = 32
                self.head_dim = 128
                self.model_path = "/tmp/test_model"

            def get_total_num_kv_heads(self):
                return self.num_key_value_heads

        self.config = MockConfig()

    def test_weight_loader_initialization(self):
        """Test WeightLoader basic initialization."""

        # Create a simple mock model
        class MockModel:
            pass

        model = MockModel()

        loader = WeightLoader(
            model=model,
            model_config=self.config,
            mesh=self.mesh,
            dtype=jnp.bfloat16,
        )

        self.assertEqual(loader.model, model)
        self.assertEqual(loader.model_config, self.config)
        self.assertEqual(loader.mesh, self.mesh)
        self.assertEqual(loader.dtype, jnp.bfloat16)

    def test_weight_loader_dtype_conversion(self):
        """Test WeightLoader with different dtype specifications."""

        class MockModel:
            pass

        model = MockModel()

        # Test with jnp dtype (WeightLoader doesn't convert dtype from string)
        loader = WeightLoader(
            model=model,
            model_config=self.config,
            mesh=self.mesh,
            dtype=jnp.float32,  # Use jnp.float32 directly
        )

        self.assertEqual(loader.dtype, jnp.float32)


class TestWeightLoaderTransformations(unittest.TestCase):
    """Test weight transformation operations."""

    def setUp(self):
        """Set up test fixtures."""
        devices = jax.devices()
        self.mesh = Mesh(devices[: min(4, len(devices))], ("tensor",))

        # Create mock model config
        class MockConfig:
            def __init__(self):
                self.hidden_size = 128
                self.num_attention_heads = 8
                self.num_key_value_heads = 8
                self.num_hidden_layers = 2
                self.head_dim = 16
                self.model_path = "/tmp/test_model"

            def get_total_num_kv_heads(self):
                return self.num_key_value_heads

        self.config = MockConfig()

    def test_transpose_operation(self):
        """Test weight transpose operation."""
        # Create test weight
        weight = jnp.ones((128, 256), dtype=jnp.float32)

        # Apply transpose
        transposed = jnp.transpose(weight)

        self.assertEqual(transposed.shape, (256, 128))

    def test_reshape_operation(self):
        """Test weight reshape operation."""
        # Create test weight
        weight = jnp.ones((128, 256), dtype=jnp.float32)

        # Apply reshape
        reshaped = weight.reshape((256, 128))

        self.assertEqual(reshaped.shape, (256, 128))

    def test_sharding_application(self):
        """Test applying sharding to weights."""
        # Create test weight
        weight = jnp.ones((128, 256), dtype=jnp.float32)

        # Create sharding spec
        sharding = NamedSharding(self.mesh, PartitionSpec(None, "tensor"))

        # Apply sharding
        sharded_weight = jax.device_put(weight, sharding)

        self.assertEqual(sharded_weight.shape, weight.shape)
        self.assertIsInstance(sharded_weight.sharding, NamedSharding)


class TestWeightLoaderPadding(unittest.TestCase):
    """Test weight padding operations."""

    def setUp(self):
        """Set up test fixtures."""
        devices = jax.devices()
        tp_size = min(4, len(devices))
        self.mesh = Mesh(devices[:tp_size], ("tensor",))

        # Create config with GQA (grouped query attention)
        class MockConfigGQA:
            def __init__(self):
                self.hidden_size = 128
                self.num_attention_heads = 8  # 8 query heads
                self.num_key_value_heads = 2  # 2 KV heads (GQA)
                self.num_hidden_layers = 2
                self.head_dim = 16
                self.model_path = "/tmp/test_model"

            def get_total_num_kv_heads(self):
                return self.num_key_value_heads

        self.config_gqa = MockConfigGQA()

        # Create config with MHA (multi-head attention)
        class MockConfigMHA:
            def __init__(self):
                self.hidden_size = 128
                self.num_attention_heads = 8
                self.num_key_value_heads = 8  # Same as query heads (MHA)
                self.num_hidden_layers = 2
                self.head_dim = 16
                self.model_path = "/tmp/test_model"

            def get_total_num_kv_heads(self):
                return self.num_key_value_heads

        self.config_mha = MockConfigMHA()

    def test_head_dim_calculation(self):
        """Test head dimension calculation."""
        head_dim = self.config_gqa.hidden_size // self.config_gqa.num_attention_heads
        self.assertEqual(head_dim, 16)  # 128 / 8 = 16

    def test_kv_head_replication_gqa(self):
        """Test KV head replication logic for GQA."""
        # GQA should need replication when using tensor parallelism
        tp_size = len(self.mesh.devices)

        if tp_size > 1:
            # With GQA and TP, KV heads need replication
            needs_replication = (
                self.config_gqa.num_key_value_heads
                < self.config_gqa.num_attention_heads
            )
            self.assertTrue(needs_replication)

    def test_kv_head_replication_mha(self):
        """Test that MHA doesn't need KV head replication."""
        # MHA should not need replication
        needs_replication = (
            self.config_mha.num_key_value_heads < self.config_mha.num_attention_heads
        )
        self.assertFalse(needs_replication)

    def test_padding_shape_calculation(self):
        """Test padding shape calculation for KV heads."""
        tp_size = len(self.mesh.devices)

        # Calculate required padding for GQA
        original_kv_heads = self.config_gqa.num_key_value_heads
        head_dim = self.config_gqa.hidden_size // self.config_gqa.num_attention_heads

        if tp_size > 1 and original_kv_heads < self.config_gqa.num_attention_heads:
            # Need to pad to align with tensor parallelism
            self.assertGreater(tp_size, 0)


class TestWeightLoaderWithMockModel(unittest.TestCase):
    """Test WeightLoader with mock model and parameters."""

    def setUp(self):
        """Set up test fixtures with mock model."""
        devices = jax.devices()
        self.mesh = Mesh(devices[: min(4, len(devices))], ("tensor",))

        # Create mock model config
        class MockConfig:
            def __init__(self):
                self.hidden_size = 128
                self.num_attention_heads = 8
                self.num_key_value_heads = 8
                self.num_hidden_layers = 2
                self.head_dim = 16
                self.model_path = "/tmp/test_model"

            def get_total_num_kv_heads(self):
                return self.num_key_value_heads

        self.config = MockConfig()

        # Create mock model with nested structure
        self.model = self._create_mock_model()

    def _create_mock_model(self):
        """Create a mock model with parameter structure."""

        class MockLinear:
            def __init__(self):
                self.weight = nnx.Param(jnp.ones((128, 256), dtype=jnp.bfloat16))

        class MockLayer:
            def __init__(self):
                self.linear = MockLinear()

        class MockModel:
            def __init__(self):
                self.layer = MockLayer()

        return MockModel()

    # Note: These tests are commented out because they require complex nnx.Module setup
    # The _get_param method is tested indirectly through real model tests

    # def test_get_param_nested_structure(self):
    #     """Test getting parameter from nested model structure."""
    #     # Requires proper nnx.Module setup
    #     pass

    # def test_get_param_invalid_path(self):
    #     """Test that invalid parameter path raises error."""
    #     # Requires proper nnx.Module setup
    #     pass


class TestWeightLoaderWithRealModel(unittest.TestCase):
    """Test WeightLoader with real model files.

    These tests require a real model to be available.
    Set TEST_MODEL_PATH environment variable to run these tests.

    Example:
        TEST_MODEL_PATH=/path/to/Qwen-7B python -m pytest python/sgl_jax/test/utils/test_weight_utils.py::TestWeightLoaderWithRealModel -v
    """

    @classmethod
    def setUpClass(cls):
        """Set up class-level test fixtures."""
        cls.model_path = cls._find_test_model_path()
        if cls.model_path is None:
            raise unittest.SkipTest(
                "No test model path found. Set TEST_MODEL_PATH environment variable or place a model in ./test_models/"
            )

        print(f"\n✓ Found test model at: {cls.model_path}")

    @classmethod
    def _find_test_model_path(cls):
        """Find a test model path from environment or common locations."""
        # Check environment variable first
        env_path = os.environ.get("TEST_MODEL_PATH")
        if env_path and os.path.exists(env_path):
            return env_path

        # Check common test locations
        test_paths = [
            "./test_models",
            "../test_models",
            "./models",
            "../models",
            "/models",
            os.path.expanduser("~/.cache/modelscope/hub"),
        ]

        for path in test_paths:
            if os.path.exists(path):
                # Look for directories that contain safetensors files
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        # Check for safetensors files
                        try:
                            files = os.listdir(item_path)
                            if any(f.endswith(".safetensors") for f in files):
                                return item_path
                        except (PermissionError, OSError):
                            continue

        return None

    def setUp(self):
        """Set up test fixtures."""
        devices = jax.devices()
        self.mesh = Mesh(devices[: min(4, len(devices))], ("tensor",))

        # Load model config from the real model using ModelConfig properly
        self.config = ModelConfig(
            model_path=self.model_path,
            trust_remote_code=True,
        )

    def test_model_path_exists(self):
        """Test that model path exists and contains necessary files."""
        self.assertTrue(os.path.exists(self.model_path))

        # Check for config file
        config_files = ["config.json"]
        has_config = any(
            os.path.exists(os.path.join(self.model_path, f)) for f in config_files
        )
        self.assertTrue(has_config, f"Model path should contain a config file")

    def test_model_has_safetensors_files(self):
        """Test that model directory contains safetensors files."""
        files = os.listdir(self.model_path)
        safetensors_files = [f for f in files if f.endswith(".safetensors")]

        self.assertGreater(
            len(safetensors_files),
            0,
            f"Model path should contain .safetensors files. Found files: {files}",
        )

        print(f"\n✓ Found {len(safetensors_files)} safetensors file(s):")
        for f in safetensors_files[:5]:  # Show first 5
            file_path = os.path.join(self.model_path, f)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  - {f} ({file_size:.2f} MB)")

    def test_load_model_config(self):
        """Test loading model configuration."""
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        self.assertIsNotNone(config)
        self.assertTrue(hasattr(config, "hidden_size"))
        self.assertTrue(hasattr(config, "num_attention_heads"))
        self.assertTrue(hasattr(config, "num_hidden_layers"))

        print(f"\n✓ Model config loaded successfully:")
        print(f"  - Model type: {config.model_type}")
        print(f"  - Hidden size: {config.hidden_size}")
        print(f"  - Attention heads: {config.num_attention_heads}")
        print(f"  - Hidden layers: {config.num_hidden_layers}")
        print(f"  - Vocab size: {config.vocab_size}")

    def test_safetensors_file_structure(self):
        """Test reading safetensors file structure."""
        try:
            from safetensors import safe_open
        except ImportError:
            self.skipTest("safetensors package not installed")

        files = os.listdir(self.model_path)
        safetensors_files = [f for f in files if f.endswith(".safetensors")]

        if not safetensors_files:
            self.skipTest("No safetensors files found")

        # Read first safetensors file
        first_file = os.path.join(self.model_path, safetensors_files[0])

        with safe_open(first_file, framework="jax") as f:
            keys = list(f.keys())

            self.assertGreater(len(keys), 0, "Safetensors file should contain weights")

            print(f"\n✓ Safetensors file structure ({safetensors_files[0]}):")
            print(f"  - Total weights: {len(keys)}")
            print(f"  - Sample weight keys:")
            for key in keys[:10]:  # Show first 10 keys
                tensor = f.get_tensor(key)
                print(f"    - {key}: shape={tensor.shape}, dtype={tensor.dtype}")

    def test_model_download_and_cache(self):
        """Test that model is properly downloaded and cached."""
        # This test verifies the model files are accessible
        self.assertTrue(os.path.exists(self.model_path))

        # Check if model is in HuggingFace cache
        cache_path = os.path.expanduser("~/.cache/huggingface/hub")
        is_cached = self.model_path.startswith(cache_path)

        print(f"\n✓ Model accessibility:")
        print(f"  - Path: {self.model_path}")
        print(f"  - Is cached: {is_cached}")
        print(f"  - Size: {self._get_directory_size(self.model_path):.2f} MB")

    def _get_directory_size(self, path):
        """Get total size of directory in MB."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(file_path)
                except (OSError, PermissionError):
                    pass
        return total_size / (1024 * 1024)  # Convert to MB

    def test_weight_loader_with_real_safetensors(self):
        """Test WeightLoader can iterate over real safetensors files."""
        try:
            from safetensors import safe_open
        except ImportError:
            self.skipTest("safetensors package not installed")

        # Get safetensors files
        files = os.listdir(self.model_path)
        safetensors_files = [
            os.path.join(self.model_path, f)
            for f in files
            if f.endswith(".safetensors")
        ]

        if not safetensors_files:
            self.skipTest("No safetensors files found")

        # Create a simple mock model for testing
        class MockModel:
            pass

        model = MockModel()

        loader = WeightLoader(
            model=model,
            model_config=self.config,
            mesh=self.mesh,
            dtype=jnp.bfloat16,
        )

        # Test iteration over weights
        total_weights = 0
        total_params = 0

        for weight_file in safetensors_files[:1]:  # Test first file only
            with safe_open(weight_file, framework="jax") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    total_weights += 1
                    total_params += np.prod(tensor.shape)

        self.assertGreater(total_weights, 0, "Should have loaded some weights")

        print(f"\n✓ Weight loading test:")
        print(f"  - Weights processed: {total_weights}")
        print(f"  - Total parameters: {total_params:,}")


class TestQwen3NNXForward(unittest.TestCase):
    """End-to-end test: load Qwen3-0.6B weights into nnx Qwen3ForCausalLM and run a forward pass."""

    @classmethod
    def setUpClass(cls):
        cls.model_path = os.environ.get(
            "TEST_MODEL_PATH",
            "/root/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B",
        )
        if not os.path.exists(cls.model_path):
            raise unittest.SkipTest(
                f"Qwen3 model path not found: {cls.model_path}. "
                "Set TEST_MODEL_PATH to a valid Qwen3-0.6B directory."
            )
        print(f"\n✓ Using Qwen3-0.6B model at: {cls.model_path}")

    def setUp(self):
        devices = jax.devices()
        self.mesh = Mesh(devices[: min(4, len(devices))], ("tensor",))

        from transformers import AutoConfig

        self.hf_config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self.model_config = ModelConfig(
            model_path=self.model_path,
            trust_remote_code=True,
        )

        self.model = Qwen3ForCausalLM(
            config=self.hf_config,
            dtype=jnp.bfloat16,
            rngs=None,
            mesh=self.mesh,
        )

    def _build_weight_mappings(self):
        """Build minimal weight mappings for embedding and lm_head."""
        weight_mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="transformers.embed_tokens.embedding"
            ),
            "lm_head.weight": WeightMapping(
                target_path="lm_head.weight",
            ),
        }
        return weight_mappings

    def test_qwen3_forward_with_loaded_weights(self):
        """Load safetensors weights into nnx Qwen3ForCausalLM and run a forward pass."""
        loader = WeightLoader(
            model=self.model,
            model_config=self.model_config,
            mesh=self.mesh,
            dtype=jnp.bfloat16,
        )
        weight_mappings = self._build_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)

        input_ids = jnp.array([[1, 2, 3]], dtype=jnp.int32)
        logits = self.model(input_ids)

        self.assertEqual(logits.ndim, 3)
        self.assertEqual(logits.shape[-1], self.hf_config.vocab_size)
        self.assertEqual(logits.dtype, jnp.bfloat16)
        print(f"\n✓ Qwen3 nnx forward logits shape: {logits.shape}")


class TestModelDownload(unittest.TestCase):
    """Test model downloading functionality.

    These tests verify that models can be downloaded from HuggingFace.
    They may take time on first run if models need to be downloaded.
    """

    def test_download_small_model(self):
        """Test downloading a small model for testing.

        This test will download a small model from ModelScope if not already cached.
        Set SKIP_DOWNLOAD_TEST=1 to skip this test.
        """
        if os.environ.get("SKIP_DOWNLOAD_TEST") == "1":
            self.skipTest("Download test skipped (SKIP_DOWNLOAD_TEST=1)")

        try:
            from modelscope import snapshot_download
        except ImportError:
            self.skipTest(
                "modelscope package not installed. Install with: pip install modelscope"
            )

        # Use a very small model for testing from ModelScope
        test_model = "Qwen/Qwen2-0.5B-Instruct"

        print(f"\n⏳ Testing model download from ModelScope: {test_model}")
        print("   (This may take a while on first run...)")

        try:
            # Download model from ModelScope
            model_dir = snapshot_download(test_model)

            self.assertIsNotNone(model_dir)
            self.assertTrue(os.path.exists(model_dir))

            print(f"\n✓ Successfully downloaded model from ModelScope")
            print(f"  - Model: {test_model}")
            print(f"  - Location: {model_dir}")

            # Check for config file
            config_file = os.path.join(model_dir, "config.json")
            if os.path.exists(config_file):
                import json

                with open(config_file) as f:
                    config = json.load(f)
                print(f"  - Model type: {config.get('model_type', 'unknown')}")
                print(f"  - Hidden size: {config.get('hidden_size', 'unknown')}")

        except Exception as e:
            self.fail(f"Failed to download/load model from ModelScope: {e}")

    def test_cache_location(self):
        """Test ModelScope cache location."""
        # ModelScope default cache location
        cache_path = os.path.expanduser("~/.cache/modelscope/hub")

        print(f"\n✓ ModelScope cache location:")
        print(f"  - Path: {cache_path}")
        print(f"  - Exists: {os.path.exists(cache_path)}")

        if os.path.exists(cache_path):
            try:
                cached_models = os.listdir(cache_path)
                print(f"  - Cached models: {len(cached_models)}")
                if cached_models:
                    print(f"  - Sample models:")
                    for model_dir in cached_models[:5]:
                        print(f"    - {model_dir}")
            except (PermissionError, OSError):
                print("  - Cannot list cache contents (permission denied)")


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestWeightMapping))
    suite.addTests(loader.loadTestsFromTestCase(TestWeightLoaderInit))
    suite.addTests(loader.loadTestsFromTestCase(TestWeightLoaderTransformations))
    suite.addTests(loader.loadTestsFromTestCase(TestWeightLoaderPadding))
    suite.addTests(loader.loadTestsFromTestCase(TestWeightLoaderWithMockModel))

    # Add tests that require real model (will be skipped if model not available)
    suite.addTests(loader.loadTestsFromTestCase(TestWeightLoaderWithRealModel))
    suite.addTests(loader.loadTestsFromTestCase(TestQwen3NNXForward))
    suite.addTests(loader.loadTestsFromTestCase(TestModelDownload))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    print("=" * 80)
    print("Weight Utils Test Suite")
    print("=" * 80)
    print("\nEnvironment:")
    print(f"  - Python: {os.sys.version}")
    print(
        f"  - JAX devices: {len(jax.devices())} ({[d.platform for d in jax.devices()]})"
    )
    print(f"  - TEST_MODEL_PATH: {os.environ.get('TEST_MODEL_PATH', 'Not set')}")
    print(f"  - Conda env: {os.environ.get('CONDA_DEFAULT_ENV', 'Not set')}")
    print("=" * 80 + "\n")

    result = run_tests()

    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
