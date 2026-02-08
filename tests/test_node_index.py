"""Tests for NodeIndex — on-demand node discovery."""

from __future__ import annotations

import pytest

from comfyui_agent.knowledge.node_index import NodeIndex


def _make_fake_object_info() -> dict:
    """Simulate ComfyUI /api/object_info response."""
    return {
        "KSampler": {
            "display_name": "KSampler",
            "category": "sampling",
            "description": "Main sampler node",
            "input": {
                "required": {
                    "model": ["MODEL"],
                    "positive": ["CONDITIONING"],
                    "negative": ["CONDITIONING"],
                    "latent_image": ["LATENT"],
                    "seed": ["INT", {"default": 0, "min": 0, "max": 2**32}],
                    "steps": ["INT", {"default": 20, "min": 1, "max": 200}],
                    "cfg": ["FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}],
                    "sampler_name": [["euler", "euler_ancestral", "dpmpp_2m"]],
                    "scheduler": [["normal", "karras", "sgm_uniform"]],
                    "denoise": ["FLOAT", {"default": 1.0}],
                },
                "optional": {},
            },
            "output": ["LATENT"],
            "output_name": ["LATENT"],
        },
        "CheckpointLoaderSimple": {
            "display_name": "Load Checkpoint",
            "category": "loaders",
            "description": "Load a checkpoint model",
            "input": {
                "required": {
                    "ckpt_name": [["model_a.safetensors", "model_b.safetensors"]],
                },
                "optional": {},
            },
            "output": ["MODEL", "CLIP", "VAE"],
            "output_name": ["MODEL", "CLIP", "VAE"],
        },
        "CLIPTextEncode": {
            "display_name": "CLIP Text Encode",
            "category": "conditioning",
            "description": "Encode text with CLIP",
            "input": {
                "required": {
                    "text": ["STRING", {"multiline": True}],
                    "clip": ["CLIP"],
                },
                "optional": {},
            },
            "output": ["CONDITIONING"],
            "output_name": ["CONDITIONING"],
        },
        "EmptyLatentImage": {
            "display_name": "Empty Latent Image",
            "category": "latent",
            "description": "Create empty latent",
            "input": {
                "required": {
                    "width": ["INT", {"default": 512}],
                    "height": ["INT", {"default": 512}],
                    "batch_size": ["INT", {"default": 1}],
                },
                "optional": {},
            },
            "output": ["LATENT"],
            "output_name": ["LATENT"],
        },
        "VAEDecode": {
            "display_name": "VAE Decode",
            "category": "latent",
            "description": "Decode latent to image",
            "input": {
                "required": {
                    "samples": ["LATENT"],
                    "vae": ["VAE"],
                },
                "optional": {},
            },
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
        },
        "SaveImage": {
            "display_name": "Save Image",
            "category": "image",
            "description": "Save image to disk",
            "input": {
                "required": {
                    "images": ["IMAGE"],
                    "filename_prefix": ["STRING", {"default": "ComfyUI"}],
                },
                "optional": {},
            },
            "output": [],
            "output_name": [],
        },
        "UpscaleModelLoader": {
            "display_name": "Load Upscale Model",
            "category": "loaders",
            "description": "Load an upscale model",
            "input": {
                "required": {
                    "model_name": [["RealESRGAN_x4.pth"]],
                },
                "optional": {},
            },
            "output": ["UPSCALE_MODEL"],
            "output_name": ["UPSCALE_MODEL"],
        },
        "ImageUpscaleWithModel": {
            "display_name": "Upscale Image (using Model)",
            "category": "image/upscaling",
            "description": "Upscale image using a model",
            "input": {
                "required": {
                    "upscale_model": ["UPSCALE_MODEL"],
                    "image": ["IMAGE"],
                },
                "optional": {},
            },
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
        },
    }


@pytest.fixture
def node_index() -> NodeIndex:
    """Create a NodeIndex with fake data (no ComfyUI needed)."""
    idx = NodeIndex()
    # Directly populate internal state instead of calling build()
    fake_info = _make_fake_object_info()
    idx._nodes = fake_info
    idx._by_category.clear()
    idx._search_corpus.clear()
    for class_name, info in fake_info.items():
        category = info.get("category", "uncategorized")
        idx._by_category.setdefault(category, []).append(class_name)
        display = info.get("display_name", class_name)
        desc = info.get("description", "")
        corpus = f"{class_name} {display} {category} {desc}".lower()
        idx._search_corpus[class_name] = corpus
    idx._built = True
    return idx


class TestNodeIndexBasic:
    def test_not_built(self):
        idx = NodeIndex()
        assert not idx.is_built
        assert idx.node_count == 0
        assert idx.categories == []
        assert "not built" in idx.search("test")

    def test_built_state(self, node_index: NodeIndex):
        assert node_index.is_built
        assert node_index.node_count == 8
        assert len(node_index.categories) > 0

    def test_categories(self, node_index: NodeIndex):
        cats = node_index.categories
        assert "sampling" in cats
        assert "loaders" in cats
        assert "conditioning" in cats
        assert "latent" in cats
        assert "image" in cats


class TestNodeIndexSearch:
    def test_search_by_keyword(self, node_index: NodeIndex):
        result = node_index.search("sampler")
        assert "KSampler" in result
        assert "Search results" in result

    def test_search_upscale(self, node_index: NodeIndex):
        result = node_index.search("upscale")
        assert "UpscaleModelLoader" in result
        assert "ImageUpscaleWithModel" in result

    def test_search_no_results(self, node_index: NodeIndex):
        result = node_index.search("nonexistent_xyz")
        assert "No nodes found" in result

    def test_search_limit(self, node_index: NodeIndex):
        result = node_index.search("a", limit=2)
        # Should limit results
        lines = [l for l in result.split("\n") if l.strip().startswith("- ")]
        assert len(lines) <= 2

    def test_list_categories(self, node_index: NodeIndex):
        result = node_index.list_categories()
        assert "Node categories" in result
        assert "sampling" in result
        assert "loaders" in result

    def test_list_category_exact(self, node_index: NodeIndex):
        result = node_index.list_category("loaders")
        assert "CheckpointLoaderSimple" in result
        assert "UpscaleModelLoader" in result

    def test_list_category_partial(self, node_index: NodeIndex):
        result = node_index.list_category("load")
        assert "CheckpointLoaderSimple" in result

    def test_list_category_not_found(self, node_index: NodeIndex):
        result = node_index.list_category("nonexistent")
        assert "not found" in result


class TestNodeIndexDetail:
    def test_get_detail(self, node_index: NodeIndex):
        result = node_index.get_detail("KSampler")
        assert "Node: KSampler" in result
        assert "sampling" in result
        assert "Required inputs:" in result
        assert "model:" in result
        assert "steps:" in result
        assert "Outputs:" in result
        assert "LATENT" in result

    def test_get_detail_case_insensitive(self, node_index: NodeIndex):
        result = node_index.get_detail("ksampler")
        assert "Node: KSampler" in result

    def test_get_detail_not_found(self, node_index: NodeIndex):
        result = node_index.get_detail("NonExistent")
        assert "not found" in result

    def test_detail_shows_enum(self, node_index: NodeIndex):
        result = node_index.get_detail("KSampler")
        assert "euler" in result  # sampler_name enum

    def test_detail_shows_outputs(self, node_index: NodeIndex):
        result = node_index.get_detail("CheckpointLoaderSimple")
        assert "MODEL" in result
        assert "CLIP" in result
        assert "VAE" in result


class TestNodeIndexValidation:
    def test_valid_workflow(self, node_index: NodeIndex):
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "model_a.safetensors"},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "hello", "clip": ["1", 1]},
            },
        }
        result = node_index.validate_workflow(workflow)
        assert "valid" in result.lower()
        assert "2 nodes" in result

    def test_unknown_node(self, node_index: NodeIndex):
        workflow = {
            "1": {"class_type": "FakeNode", "inputs": {}},
        }
        result = node_index.validate_workflow(workflow)
        assert "unknown class_type" in result
        assert "FakeNode" in result

    def test_missing_required_input(self, node_index: NodeIndex):
        workflow = {
            "1": {
                "class_type": "KSampler",
                "inputs": {"seed": 42},  # missing model, positive, negative, etc.
            },
        }
        result = node_index.validate_workflow(workflow)
        assert "missing required input" in result
        assert "model" in result

    def test_missing_class_type(self, node_index: NodeIndex):
        workflow = {
            "1": {"inputs": {"foo": "bar"}},
        }
        result = node_index.validate_workflow(workflow)
        assert "missing class_type" in result

    def test_unknown_input_warning(self, node_index: NodeIndex):
        workflow = {
            "1": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["0", 0],
                    "filename_prefix": "test",
                    "unknown_param": 123,
                },
            },
        }
        result = node_index.validate_workflow(workflow)
        assert "unknown input" in result
        assert "unknown_param" in result
