import os
import unittest
import warnings
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

# Suppress XLA warnings about missing SoL config
os.environ["JAX_PLATFORMS"] = "cuda"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow/XLA warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure JAX logging to suppress XLA warnings
import logging

# Set up basic logging configuration to show INFO level
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)

# Only suppress JAX-related loggers to hide XLA warnings
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("jax._src").setLevel(logging.ERROR)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
logging.getLogger("jax._src.compiler").setLevel(logging.ERROR)

from nanovllm_jax.configs.model_config import ModelConfig
from nanovllm_jax.utils.weight_utils import WeightLoader, WeightMapping
from nanovllm_jax.models.qwen3 import Qwen3ForCausalLM

# Ensure WeightLoader logger is visible
logging.getLogger("nanovllm_jax.utils.weight_utils").setLevel(logging.INFO)


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


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestQwen3NNXForward))
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    run_tests()
