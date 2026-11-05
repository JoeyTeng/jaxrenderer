from ._meta_utils import add_tracing_name
from ._meta_utils import typed_jit as jit
from .geometry import (
    Camera,
    normalise,
    quaternion,
    quaternion_mul,
    rotation_matrix,
)
from .model import Model, ModelObject, batch_models, merge_objects
from .pipeline import render
from .renderer import CameraParameters, LightParameters, Renderer, ShadowParameters
from .scene import GUID, Scene
from .shader import PassthroughShader, Shader
from .shader_types import (
    Fragment,
    FragmentCollection,
    FragmentInput,
    InterpolationInput,
    MixInput,
    MixerOutput,
    PrimitiveInput,
    VertexInput,
)
from .shapes.capsule import UpAxis, create_capsule
from .shapes.cube import create_cube
from .types import (
    Buffers,
    Colour,
    JaxFloating,
    JaxInteger,
    LightSource,
    PixelState,
    SpecularMap,
    Texture,
    Vec3f,
)
from .utils import (
    build_texture_from_PyTinyrenderer,
    transpose_for_display,
)

__all__ = [
    "add_tracing_name",
    "batch_models",
    "Buffers",
    "build_texture_from_PyTinyrenderer",
    "Camera",
    "CameraParameters",
    "Colour",
    "create_capsule",
    "create_cube",
    "Fragment",
    "FragmentCollection",
    "FragmentInput",
    "GUID",
    "InterpolationInput",
    "JaxFloating",
    "JaxInteger",
    "jit",
    "LightParameters",
    "LightSource",
    "MixInput",
    "MixerOutput",
    "merge_objects",
    "Model",
    "ModelObject",
    "normalise",
    "PassthroughShader",
    "PixelState",
    "PrimitiveInput",
    "quaternion_mul",
    "quaternion",
    "render",
    "Renderer",
    "rotation_matrix",
    "Scene",
    "Shader",
    "ShadowParameters",
    "SpecularMap",
    "Texture",
    "transpose_for_display",
    "UpAxis",
    "VertexInput",
    "Vec3f",
]
