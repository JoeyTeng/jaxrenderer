from functools import partial
from typing import Any, Callable, Iterable, NamedTuple, Optional, TypeVar

from jaxtyping import Array, Bool, Float, Integer
import jax

from .geometry import Interpolation
from .types import Colour, Vec2f, Vec3f, Vec4f

jax.config.update('jax_array', True)

ID = Integer[Array, ""]


class VertexShaderInput(NamedTuple):
    pass


class PerVertex(NamedTuple):
    gl_Position: Vec4f
    # gl_PointSize is meaningful only when rendering point primitives.
    gl_PointSize: Float[Array, ""]


class ToInterpolate(NamedTuple):
    """Used for Shader.interpolate. The 0-th axis is the batch axis"""
    gl_Position: Float[Array, "3 4"]
    # gl_PointSize is meaningful only when rendering point primitives.
    gl_PointSize: Float[Array, "3"]


# Used for Shader.interpolate
_InterpolatedT = TypeVar("_InterpolatedT", bound=PerVertex)


class PerFragment(NamedTuple):
    """Output from Fragment Shader.

    If gl_FragDepth is not set, gl_FragCoord[3] will be used later by default.
    """
    gl_FragDepth: Optional[Float[Array, ""]] = None


class DefaultPerFragment(PerFragment):
    """When render to only one buffer, for simplicity.

    Reference:
      - https://stackoverflow.com/questions/51459596/using-gl-fragcolor-vs-out-vec4-color
    """
    gl_FragDepth: Optional[Float[Array, ""]] = None
    gl_FragColor: Colour


_ShaderT = TypeVar("_ShaderT", bound=NamedTuple)


class Shader(NamedTuple):
    """Base class for customised shader.

    Since JAX is pure functional (stateless), the state will be maintained by
    returning and passing the updated instance of `Shader` using `self.replace`.

    In one rendering process, `vertex` will be called for all primitives
    concurrently and thus they share same state when being called. The state in
    the shader are like `varying` qualifier of GLSL prior to 1.40. They are
    additional input of a fragment shader or the output of a vertex shader.
    """

    def vertex(
        self: _ShaderT,
        *,
        gl_VertexID: ID,
        gl_InstanceID: ID,
        **kwargs,
    ) -> tuple[_ShaderT, PerVertex]:
        """Override this to implement the vertex shader as defined by OpenGL.

        The meaning of the inputs follows the definitions in OpenGL. Additional
        named parameters can be defined and passed in if you need, as defined
        in `**kwargs`.

        If any internal state of this shader needs to be updated (in a
        per-vertex basis), it must be updated using `self.replace` method and
        return the updated instance as the function output. Otherwise the
        update/internal state will not be tracked.

        Relevant information from the original document is copied below.

        Parameters:
          - gl_VertexID
            the index of the vertex currently being processed. When using
            non-indexed rendering, it is the effective index of the current
            vertex (the number of vertices processed + the `first` value). For
            indexed rendering, it is the index used to fetch this vertex from
            the buffer.
          - gl_InstanceID
            the index of the current instance when doing some form of instanced
            rendering. The instance count always starts at 0, even when using
            base instance calls. When not using instanced rendering, this value
            will be 0.

        Return: a tuple of updated shader instance and PerVertex, used for
            internals.
          - gl_Position
            the clip-space output position of the current vertex.
          - gl_PointSize
            the pixel width/height of the point being rasterized. It only has a
            meaning when rendering point primitives. It will be clamped to the
            GL_POINT_SIZE_RANGE.

        Reference:
          - [Vertex Shader/Defined Inputs](https://www.khronos.org/opengl/wiki/Vertex_Shader/Defined_Inputs)
          - [Vertex Shader#Outputs](https://www.khronos.org/opengl/wiki/Vertex_Shader#Outputs)
        """
        raise NotImplementedError()

    def interpolate(
        values: VaryingT,
        barycentric_screen: Vec3f,
        barycentric_clip: Vec3f,
    ) -> VaryingT:
        """Override this to customise the interpolation of user-defined inputs.

        The default implementation is to interpolate all the fields of given
        values as `smooth`, which is perspective interpolation, as defined in
        GLSL.

        Parameters:
          - values: values at the vertices of the triangle, with axis 0 being
            the batch axis. It is expected to be a tuple of multiple batched
            values.
          - barycentric_screen: barycentric coordinates in screen space of the
            point to interpolate
          - barycentric_clip: barycentric coordinates in clip space of the
            point to interpolate

        Return: interpolated values for fragment shader process, with same
            structure (order of members) as `values`
        """
        f = partial(Interpolation.SMOOTH, barycentric_screen, barycentric_clip)

        return self, factory((f(value) for value in values))

    def fragment(
        self: _ShaderT,
        *,
        gl_FragCoord: Vec4f,
        gl_FrontFacing: Bool[Array, ""],
        gl_PointCoord: Vec2f,
        **kwargs,
    ) -> tuple[_ShaderT, PerFragment]:
        """Override this to implement the vertex shader as defined by OpenGL.

        This is optional. The default implementation writes nothing and thus
        `gl_FragDepth` further down the pipeline will use `gl_FragCoord[3]`.

        Parameters:
          - gl_FragCoord: homogeneous coordinates in screen device space.
          - gl_FrontFacing: True if the primitive is NOT back facing.
          - gl_PointCoord: 2d coordinates in screen device space.

        Return: Return PerFragment for further blending process. Return an
            (updated) instance of `self` to keep some states for further draw
            command.
          - gl_FragDepth: if not set (remains None), gl_FragCoord[3] will be
            used later.

        Reference:
          - [Fragment Shader/Defined Inputs](https://www.khronos.org/opengl/wiki/Fragment_Shader/Defined_Inputs)
          - [Fragment Shader#Outputs](https://www.khronos.org/opengl/wiki/Fragment_Shader#Outputs)
        """
        return self, PerFragment()

    def blender(
        self: _ShaderT,
        *,
        source: Colour,
        destination: Colour,
    ) -> tuple[_ShaderT, Colour]:
        """Override this to define the customised behaviour for colour blending
            as defined by OpenGL.

        Blending happens independently for fragment shader output with type
        `Colour`. This means different fragments results will not affect each
        other, nor the different outputs for the same fragment.

        This is optional. The default implementation writes simply returns the
        source colour.

        Parameters:
          - source: the source colour, which is the output of the fragment
            shader.
          - destination: the destination colour, which is the colour already in
            the framebuffer.

        Return: Return Colour to be written to the framebuffer. Return an
            (updated) instance of `self` to keep some states for further draw
            command.

        Reference:
          - [Blending](https://www.khronos.org/opengl/wiki/Blending)
        """
        return self, source + 0 * destination
