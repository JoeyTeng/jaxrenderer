from collections.abc import Sequence
from functools import partial
from typing import NamedTuple, Optional, TypeAlias, Union, cast

import jax
from jax import Array
import jax.experimental.checkify as checkify
import jax.lax as lax
import jax.numpy as jnp
from jax.tree_util import tree_map
from jaxtyping import (
    Bool,
    Float,
    Integer,
    Num,
    Shaped,
)

from ._meta_utils import (
    add_tracing_name,
    is_array_type,
    typed_array,
    typed_asarray,
    typed_full,
    typed_modf,
    typed_ones,
)
from ._meta_utils import typed_jaxtyped as jaxtyped
from ._meta_utils import typed_jit as jit
from .geometry import Camera, transform_matrix_from_rotation
from .types import (
    FALSE_ARRAY,
    BoolV,
    FaceIndices,
    FloatV,
    IntV,
    Normals,
    SpecularMap,
    Texture,
    UVCoordinates,
    Vec3f,
    Vec4f,
    Vertices,
)
from .value_checker import index_in_bound

ModelMatrix: TypeAlias = Float[Array, "4 4"]


class Model(NamedTuple):
    """Model with vertices specification and attached maps.

    NormalMap is not included for now as it is not used in the reference
    implementation
    [erwincoumans/tinyrenderer](https://github.com/erwincoumans/tinyrenderer).
    """

    verts: Vertices
    norms: Normals
    uvs: UVCoordinates
    faces: FaceIndices
    faces_norm: FaceIndices
    faces_uv: FaceIndices

    diffuse_map: Texture
    specular_map: SpecularMap

    @classmethod
    @jaxtyped
    def create(
        cls,
        verts: Vertices,
        norms: Normals,
        uvs: UVCoordinates,
        faces: FaceIndices,
        diffuse_map: Texture,
        specular_map: Optional[SpecularMap] = None,
    ) -> "Model":
        """A convenient method to create a Model assuming faces_norm and
        faces_uv are the same as faces. A default specular_map is used if
        not given, with a constant value of 2.0.
        """
        if specular_map is None:
            # reference: https://github.com/erwincoumans/tinyrenderer/blob/89e8adafb35ecf5134e7b17b71b0f825939dc6d9/model.cpp#L215
            specular_map = lax.full(diffuse_map.shape[:2], 2.0)

        assert is_array_type(verts, Vertices), f"{verts}"
        assert is_array_type(norms, Normals), f"{norms}"
        assert is_array_type(uvs, UVCoordinates), f"{uvs}"
        assert is_array_type(faces, FaceIndices), f"{faces}"
        assert is_array_type(diffuse_map, Texture), f"{diffuse_map}"
        assert is_array_type(specular_map, SpecularMap), f"{specular_map}"

        return cls(
            verts=verts,
            norms=norms,
            uvs=uvs,
            faces=faces,
            faces_norm=faces,
            faces_uv=faces,
            diffuse_map=diffuse_map,
            specular_map=specular_map,
        )

    @jaxtyped
    @jit
    def asserts(self) -> None:
        """Asserts that all fields are of correct shape and type."""
        assert is_array_type(self.verts, Vertices), f"{self.verts}"
        assert is_array_type(self.norms, Normals), f"{self.norms}"
        assert is_array_type(self.uvs, UVCoordinates), f"{self.uvs}"
        assert is_array_type(self.faces, FaceIndices), f"{self.faces}"
        assert is_array_type(self.faces_norm, FaceIndices), f"{self.faces_norm}"
        assert is_array_type(self.faces_uv, FaceIndices), f"{self.faces_uv}"
        assert is_array_type(self.diffuse_map, Texture), f"{self.diffuse_map}"
        assert is_array_type(self.specular_map, SpecularMap), f"{self.specular_map}"

    @jaxtyped
    @checkify.checkify
    @jit
    def value_checks(self) -> None:
        """Check values are of correct value ranges.

        Checks implemented:
          - indices are in bound, for faces, faces_norm, faces_uv (against
            verts, norms, uv).

        UV coordinates are not checked as out-of-bound values are allowed and used as "repeat" mode.

        Usage:
          model = Model(...)
          err, _ = model.value_checks()
          err.throw()  # throw if any error is found.
        """
        # indices are in bound
        checkify.check(  # pyright: ignore[reportUnknownMemberType]
            index_in_bound(self.faces, self.verts.shape[0]),
            # f-string for shape as shape is compile-time constant and
            # is not supported as a format string parameter in checkify.
            f"faces out of bound, expected [0, {self.verts.shape[0]}),"
            # separately specify fmt_kwargs in a non-f-string.
            " got {max_idx}.",
            max_idx=self.faces.max(),
        )
        checkify.check(  # pyright: ignore[reportUnknownMemberType]
            index_in_bound(self.faces_norm, self.norms.shape[0]),
            f"faces_norm out of bound, expected [0, {self.norms.shape[0]}),"
            " got {max_idx}.",
            max_idx=self.faces_norm.max(),
        )
        checkify.check(  # pyright: ignore[reportUnknownMemberType]
            index_in_bound(self.faces_uv, self.uvs.shape[0]),
            f"faces_uv out of bound, expected [0, {self.uvs.shape[0]}),"
            " got {max_idx}.",
            max_idx=self.faces_uv.max(),
        )


VertT: TypeAlias = Num[Array, "_dim ..."]
VertsT: TypeAlias = Sequence[VertT]
FaceIndicesT: TypeAlias = Num[Array, "_faces 3"]
FaceIndicessT: TypeAlias = Sequence[FaceIndicesT]

MapT: TypeAlias = Num[Array, "_width _height ..."]
MapsT: TypeAlias = Sequence[MapT]


class MergedModel(NamedTuple):
    """Merged model with vertices, normals, uv coordinates, and faces."""

    verts: Vertices
    norms: Normals
    uvs: UVCoordinates
    faces: FaceIndices
    faces_norm: FaceIndices
    faces_uv: FaceIndices

    # broadcasted object info into per-vertex
    texture_index: Integer[Array, "vertices"]
    """Texture map index for each vertex."""
    double_sided: Bool[Array, "vertices"]
    """Whether each face is double sided."""

    # Merged maps
    texture_shape: Integer[Array, "objects 2"]
    """Width, height of each texture map."""
    offset: int
    """Width of biggest merged maps, as [0] returned by `merge_maps`."""
    diffuse_map: Texture
    specular_map: SpecularMap

    @staticmethod
    @jaxtyped
    @add_tracing_name
    def generate_object_vert_info(
        counts: Sequence[int],
        values: Sequence[Union[int, Shaped[Array, "..."]]],
    ) -> Shaped[Array, "vertices ..."]:
        """Generate object-wide info for each vertex in merged model as
            vertex-level info.

        Parameters:
          - counts: Number of vertices of each object.
          - values: value to be filled in for each object.

        Returns: Map indices for each face.

        Note: this function cannot be jitted by itself as it uses the value of
            `counts` to create matrices: the shape depends on the value of
            `counts`.
        """

        def fill_vertices(
            count: int,
            value: Union[int, Shaped[Array, "..."]],
        ) -> Array:
            vertices: Array = typed_full(
                (count, *typed_asarray(value).shape),
                value,
            )
            return vertices

        _values: Sequence[Array] = cast(
            Sequence[Array],
            tree_map(fill_vertices, counts, values),
        )
        map_indices = lax.concatenate(_values, dimension=0)

        return map_indices

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def merge_verts(
        vs: VertsT,
        fs: FaceIndicessT,
    ) -> tuple[VertT, FaceIndicesT]:
        """"""
        counts = [v.shape[0] for v in vs[:-1]]
        cumsum = [0]
        for count in counts:
            cumsum.append(cumsum[-1] + count)

        dtype = jax.dtypes.result_type(*vs)

        # merge vertices
        verts: VertT = lax.concatenate(
            [v.astype(dtype) for v in vs],
            dimension=0,
        )
        # merge faces
        faces: FaceIndicesT = lax.concatenate(
            [f + cumsum[i] for i, f in enumerate(fs)],
            dimension=0,
        )

        return verts, faces

    @staticmethod
    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def merge_maps(maps: MapsT) -> tuple[MapT, tuple[int, int]]:
        """Merge maps by concatenating them along the first axis.

        All maps must have the same number of dimensions, i.e., `len(m.shape)`
        must be the same for all maps `m`.

        If the other axises is not the same, they are padded with undefined
        values, keeping [0:shape[i]] as given values, and [shape[i]:] being
        padded values at dimension i. If they are of different dtypes, they are
        promoted together.

        Parameters:
          - maps: a list of maps to merge.

        Returns:
          - A merged map.
          - shape of first and second axis of each map.
        """
        # TODO: find a better way to merge maps
        with jax.ensure_compile_time_eval():
            dims: int = len(maps[0].shape)

            def map_shape(m: MapT) -> tuple[int, ...]:
                return m.shape

            shapes: Sequence[tuple[int, ...]] = tree_map(map_shape, maps)
            # pick the largest shape for each dimension
            single_shape: tuple[int, ...] = tuple(
                max(shape[i] for shape in shapes) for i in range(dims)
            )

            dtype = jax.dtypes.result_type(*maps)

        def pad_map(m: MapT) -> MapT:
            def padding(
                capacity: int,
                content: int,
            ) -> tuple[int, int, int]:
                return (0, capacity - content, 0)

            padding_config = tree_map(padding, single_shape, m.shape)
            return lax.pad(m, lax.full((), 0, dtype=m.dtype), padding_config).astype(
                dtype
            )

        padded_maps: Sequence[MapT] = tree_map(pad_map, maps)
        new_map: MapT = lax.concatenate(padded_maps, dimension=0)

        return new_map, (single_shape[0], single_shape[1])

    @staticmethod
    @jaxtyped
    @add_tracing_name
    def uv_repeat(
        uv: Float[Array, "2"],
        shape: Integer[Array, "2"],
        map_index: IntV,
        offset: IntV,
    ) -> Float[Array, "2"]:
        """Compute final UV coordinates as if it is repeatedly tiled (in case
            of out-of-bound uv coordinate).

        Parameters:
          - uv: raw uv coordinates, in floating numbers. Only fractional part
            is used, as if the uv coordinates are in [0, 1].
          - shape: of the map being used, according to `map_index`.
          - offset: of each map in the merged maps, as [0] returned by
            `merge_maps`. Only first axis is required, thus here we just need a
            scalar.
          - map_index: index of the map to use.
        """
        assert is_array_type(uv, Float[Array, "2"]), f"{uv}"
        assert is_array_type(shape, Integer[Array, "2"]), f"{shape}"
        assert is_array_type(map_index, IntV), f"{map_index}"
        assert is_array_type(offset, IntV), f"{offset}"
        # since given uv are in [0, 1] (and may be scaled, if is cube),
        # we need to multiply it by (w, h) of the texture map first.
        # This is equivalent to just obtain the fractional part of uv.
        fractional_uv: FloatV = typed_modf(uv)[0]
        fractional_uv = jnp.where(
            fractional_uv < 0,
            fractional_uv + 1,
            fractional_uv,
        )
        assert is_array_type(fractional_uv, Float[Array, "2"])

        return (fractional_uv * shape).at[0].add(map_index * offset)


class ModelObject(NamedTuple):
    """Model object with model and transform."""

    model: Model
    """Reference to the model, with mesh, attached maps, etc."""
    local_scaling: Vec3f = typed_ones(3)
    """Local scaling factors in x, y, z. Zero factors flatten axes.

    Normal directions use a scale-only limit where it is unique.
    This concerns unit directions with source normals held fixed; raw batched
    normal magnitudes need not be continuous. Other singular cases retain
    finite fallbacks without a general differentiability guarantee.
    """
    transform: ModelMatrix = jnp.identity(4)
    """Transform matrix (model matrix) of the model."""
    # TODO: Support double_sided
    double_sided: BoolV = FALSE_ARRAY
    """Whether the object is double-sided."""

    @jaxtyped
    def replace_with_position(self, position: Vec3f) -> "ModelObject":
        """Return a new ModelObject with given position.

        !!This does not change the original object.

        Parameters:
          - position: the new position of the object.
        """
        return self._replace(transform=self.transform.at[:3, 3].set(position))

    @jaxtyped
    def replace_with_orientation(
        self,
        orientation: Optional[Vec4f] = None,
        rotation_matrix: Optional[Float[Array, "3 3"]] = None,
    ) -> "ModelObject":
        """Return a new ModelObject with given orientation or rotation_matrix.

        If rotation_matrix is specified, it takes precedence over orientation.
        If none is specified, the object's orientation is set to identity.

        !!This does not change the original object.

        Parameters:
          - orientation: the new orientation of the object, optional.
          - rotation_matrix: the new rotation matrix of the object, optional
        """
        if rotation_matrix is None:
            if orientation is None:
                orientation = typed_array((0.0, 0.0, 0.0, 1.0))

            assert is_array_type(orientation, Vec4f), f"{orientation}"
            rotation_matrix = transform_matrix_from_rotation(orientation)

        assert is_array_type(rotation_matrix, Float[Array, "3 3"]), f"{rotation_matrix}"

        return self._replace(transform=self.transform.at[:3, :3].set(rotation_matrix))

    @jaxtyped
    def replace_with_local_scaling(self, local_scaling: Vec3f) -> "ModelObject":
        """Return a new ModelObject with given local_scaling.

        !!This does not change the original object.

        Parameters:
          - local_scaling: the new local scaling of the object.
        """
        return self._replace(local_scaling=local_scaling)

    @jaxtyped
    def replace_with_double_sided(self, double_sided: BoolV) -> "ModelObject":
        """Return a new ModelObject with given double_sided.

        !!This does not change the original object.

        Parameters:
          - double_sided: whether the object is double-sided.
        """
        return self._replace(double_sided=double_sided)


@add_tracing_name
def batch_models(models: Sequence[MergedModel]) -> MergedModel:
    """Merge multiple MergedModel into one, with each field being a batch, with
    batch axis at 0. This is intended to facilitate `jax.vmap`.
    """
    merged_model = MergedModel._make(
        (
            lax.concatenate(
                [typed_asarray(model[i])[None, ...] for model in models],
                dimension=0,
            )
            for i in range(len(models[0]))
        )
    )

    return merged_model


@jaxtyped
@add_tracing_name
def merge_objects(objects: Sequence[ModelObject]) -> MergedModel:
    """Merge objects into a single model.

    Parameters:
      - objects: a list of objects to merge.

    Returns: A model containing the merged objects into one single mesh.
    """
    with jax.ensure_compile_time_eval():
        models = [obj.model for obj in objects]

        # broadcasted per vertex info
        counts: list[int] = [len(m.verts) for m in models]

        map_indices: Integer[Array, "vertices"]
        map_indices = MergedModel.generate_object_vert_info(
            counts,
            list(range(len(models))),
        )
        assert is_array_type(map_indices, Integer[Array, "vertices"])

        map_wh_per_object = typed_asarray([m.diffuse_map.shape[:2] for m in models])
        assert is_array_type(map_wh_per_object, Integer[Array, "objects 2"])

        double_sided: Bool[Array, "vertices"]
        double_sided = MergedModel.generate_object_vert_info(
            counts,
            [obj.double_sided for obj in objects],
        )
        assert is_array_type(double_sided, Bool[Array, "vertices"])

    # merge maps
    diffuse_map, single_map_shape = MergedModel.merge_maps(
        [m.diffuse_map for m in models]
    )
    specular_map = MergedModel.merge_maps([m.specular_map for m in models])[0]

    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def transform_vert(
        verts: Float[Array, "N 3"],
        local_scaling: Vec3f,
        transform: ModelMatrix,
    ) -> Vertices:
        """Apply transforms defined in `ModelObject` to vertices."""
        world: Float[Array, "N 3"] = Camera.apply_pos(verts * local_scaling, transform)
        assert is_array_type(world, Float[Array, "N 3"])

        return world

    # merge verts
    verts, faces = MergedModel.merge_verts(
        [
            transform_vert(
                verts=obj.model.verts,
                local_scaling=obj.local_scaling,
                transform=obj.transform,
            )
            for obj in objects
        ],
        [m.faces for m in models],
    )

    @jaxtyped
    @partial(jit, inline=True)
    @add_tracing_name
    def transform_normals(
        normals: Float[Array, "N 3"],
        local_scaling: Vec3f,
        transform: ModelMatrix,
    ) -> Vertices:
        """Apply the local cofactor and the object's inverse-transpose transform."""
        # The diagonal cofactor is proportional to inverse local scaling for
        # every nonsingular scale, but stays defined when an axis is flattened.
        scale_magnitude = jnp.max(jnp.abs(local_scaling))
        safe_scale_magnitude = jnp.where(scale_magnitude > 0.0, scale_magnitude, 1.0)
        scaled: Vec3f = local_scaling / safe_scale_magnitude
        local_cofactor: Vec3f = jnp.stack(
            (
                scaled[1] * scaled[2],
                scaled[0] * scaled[2],
                scaled[0] * scaled[1],
            )
        )
        # Cofactors include the determinant sign. Restore the inverse-transpose
        # direction for negative scales; treat zero components as positive.
        orientation_sign = jnp.prod(jnp.where(scaled < 0.0, -1.0, 1.0))
        local_normals = normals * local_cofactor * orientation_sign
        has_transformed_normal = jnp.any(local_normals != 0.0, axis=-1, keepdims=True)

        # If a normal has no component on a flattened axis, its scale-only
        # limit is unique. Replacing zero scale entries with one lets the
        # cofactor compute that direction without dividing by zero.
        zero_scale = local_scaling == 0.0
        zero_axis_support = jnp.any(
            (normals != 0.0) & zero_scale,
            axis=-1,
            keepdims=True,
        )
        extension_scale = jnp.where(zero_scale, 1.0, scaled)
        extension_cofactor: Vec3f = jnp.stack(
            (
                extension_scale[1] * extension_scale[2],
                extension_scale[0] * extension_scale[2],
                extension_scale[0] * extension_scale[1],
            )
        )
        extension_normals = normals * extension_cofactor * orientation_sign
        extension_magnitude = jnp.max(
            jnp.abs(extension_normals),
            axis=-1,
            keepdims=True,
        )
        has_extension_normal = extension_magnitude > 0.0
        use_scale_limit = (
            ~has_transformed_normal
            & jnp.any(zero_scale)
            & ~zero_axis_support
            & has_extension_normal
        )
        finite_fallback = jnp.where(has_transformed_normal, local_normals, normals)
        # Keep the cofactor or finite fallback where no unique scale-only
        # extension exists.
        local_normals = jnp.where(
            use_scale_limit,
            extension_normals,
            finite_fallback,
        )

        # Normal magnitude is not part of the shading contract. Rescale each
        # row before batch normalisation so tiny cofactor directions stay finite.
        local_magnitude = jnp.max(jnp.abs(local_normals), axis=-1, keepdims=True)
        safe_local_magnitude = jnp.where(local_magnitude > 0.0, local_magnitude, 1.0)
        # The positive factor cancels under per-vertex direction normalisation;
        # freezing it avoids reciprocal-square overflow in AD.
        safe_local_magnitude = lax.stop_gradient(safe_local_magnitude)
        local_normals = local_normals / safe_local_magnitude

        world: Float[Array, "N 3"] = Camera.apply_vec(
            local_normals,
            # transform by inverse transpose
            jnp.linalg.inv(transform).T,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        )
        assert is_array_type(world, Float[Array, "N 3"])

        return world

    norms, faces_norm = MergedModel.merge_verts(
        [
            transform_normals(obj.model.norms, obj.local_scaling, obj.transform)
            for obj in objects
        ],
        [m.faces_norm for m in models],
    )
    uvs, faces_uv = MergedModel.merge_verts(
        [m.uvs for m in models],
        [m.faces_uv for m in models],
    )

    return MergedModel(
        verts=verts,
        norms=norms,
        uvs=uvs,
        faces=faces,
        faces_norm=faces_norm,
        faces_uv=faces_uv,
        texture_shape=map_wh_per_object,
        texture_index=map_indices,
        double_sided=double_sided,
        offset=single_map_shape[0],
        diffuse_map=diffuse_map,
        specular_map=specular_map,
    )
