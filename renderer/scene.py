from __future__ import annotations  # tolerate "subscriptable 'type' for < 3.9

from typing import NamedTuple, NewType, Optional, Union

import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, Float
from jaxtyping import jaxtyped  # pyright: ignore[reportUnknownVariableType]

from ._backport import DictT, Tuple, replace_dict
from .model import Model, ModelObject
from .shapes.capsule import UpAxis, create_capsule
from .shapes.cube import create_cube
from .types import BoolV, SpecularMap, Texture, Vec3f, Vec4f

GUID = NewType("GUID", int)


class Scene(NamedTuple):
    """Scene with models and objects. Noticed that with each update to the
    scene, the scene instance is replaced with a new one.
    """

    guid: GUID = GUID(0)
    """Max unique identifier among all objects in the scene. It equals to the
        numbers of models and objects ever created in the scene.
    """
    models: DictT[GUID, Model] = {}
    """Models in the scene, indexed by their unique identifier."""
    objects: DictT[GUID, ModelObject] = {}
    """Objects in the scene, indexed by their unique identifier."""

    @jaxtyped
    def add_model(self, model: Model) -> Tuple["Scene", GUID]:
        """Add a model to the scene.

        Parameters:
          - model: a model to add to the scene.

        Returns:
          A tuple of the updated scene and the unique identifier of the model.
        """
        guid = self.guid
        new_scene = self._replace(
            guid=GUID(guid + 1),
            models={**self.models, guid: model},
        )

        return new_scene, guid

    @jaxtyped
    def add_cube(
        self,
        half_extents: Union[Float[Array, "3"], Tuple[float, float, float]],
        diffuse_map: Texture,
        texture_scaling: Union[Float[Array, "2"], Tuple[float, float], float],
    ) -> Tuple["Scene", GUID]:
        """Add a cube to the scene.

        Parameters:
          - half_extents: the half-size of the cube. The final cube would have
            x-y-z dimension of 2 * half_extents.
          - diffuse_map: the diffuse map of the cube.
          - texture_scaling: the scaling factor of the texture, in x and y. If
            only one number is given, it is used for both x and y.
        """
        # reference: https://github.com/erwincoumans/tinyrenderer/blob/89e8adafb35ecf5134e7b17b71b0f825939dc6d9/model.cpp#L215
        specular_map: SpecularMap = (
            lax.full(  # pyright: ignore[reportUnknownMemberType]
                diffuse_map.shape[:2], 2.0
            )
        )

        _half_extents = jnp.asarray(  # pyright: ignore[reportUnknownMemberType]
            half_extents
        )
        assert isinstance(
            _half_extents, Float[Array, "3"]
        ), f"Expected 2 floats in half_extends, got {half_extents}"

        _texture_scaling = jnp.asarray(  # pyright: ignore[reportUnknownMemberType]
            texture_scaling
        )
        if _texture_scaling.size == 1:
            _texture_scaling = lax.full(  # pyright: ignore[reportUnknownMemberType]
                (2,), _texture_scaling
            )
        assert isinstance(
            _texture_scaling, Float[Array, "2"]
        ), f"Expected 2 floats in texture_scaling, got {texture_scaling}"

        model: Model = create_cube(
            half_extents=_half_extents,
            texture_scaling=_texture_scaling,
            diffuse_map=diffuse_map,
            specular_map=specular_map,
        )

        return self.add_model(model)

    @jaxtyped
    def add_capsule(
        self,
        radius: float,
        half_height: float,
        up_axis: UpAxis,
        diffuse_map: Texture,
    ) -> Tuple["Scene", GUID]:
        """Add a capsule to the scene.

        Parameters:
          - radius: the radius of the capsule.
          - half_height: the half height of the capsule.
          - up_axis: the up axis of the capsule.
          - diffuse_map: the diffuse map of the capsule.
        """
        # reference: https://github.com/erwincoumans/tinyrenderer/blob/89e8adafb35ecf5134e7b17b71b0f825939dc6d9/model.cpp#L215
        specular_map: SpecularMap = (
            lax.full(  # pyright: ignore[reportUnknownMemberType]
                diffuse_map.shape[:2],
                2.0,
            )
        )
        model: Model = create_capsule(
            radius=jnp.asarray(radius),  # pyright: ignore[reportUnknownMemberType]
            half_height=jnp.asarray(  # pyright: ignore[reportUnknownMemberType]
                half_height
            ),
            up_axis=up_axis,
            diffuse_map=diffuse_map,
            specular_map=specular_map,
        )

        return self.add_model(model)

    @jaxtyped
    def add_object_instance(self, model_id: GUID) -> Tuple["Scene", GUID]:
        """Add an object instance to the scene.

        Parameters:
          - model_id: the unique identifier of the model to add.

        Returns:
          A tuple of the updated scene and the unique identifier of the object.
        """
        guid = self.guid
        new_scene = self._replace(
            guid=GUID(guid + 1),
            objects={**self.objects, guid: ModelObject(model=self.models[model_id])},
        )

        return new_scene, guid

    @jaxtyped
    def delete_model(self, model_id: GUID, check: bool) -> "Scene":
        """Delete a model from the scene.

        Parameters:
          - model_id: the unique identifier of the model to delete.
          - check: whether to check if the model exists and/or being used.
        """
        if model_id not in self.models:
            if check:
                raise RuntimeError(f"model {model_id} does not exist")

            return self

        model = self.models[model_id]
        if check:
            for object_id, object in self.objects.items():
                if object.model == model:
                    raise RuntimeError(
                        f"model {model_id} is being used by object" f" {object_id}"
                    )

        models = {k: v for k, v in self.models.items() if k != model_id}
        del model

        return self._replace(models=models)

    @jaxtyped
    def delete_object(self, object_id: GUID, check: bool) -> "Scene":
        """Delete an object from the scene.

        Parameters:
          - object_id: the unique identifier of the object to delete.
          - check: whether to check if the object exists.
        """
        if object_id not in self.objects:
            if check:
                raise RuntimeError(f"object {object_id} does not exist")

            return self

        _object = self.objects[object_id]
        objects = {k: v for k, v in self.objects.items() if k != object_id}
        del _object

        return self._replace(objects=objects)

    @jaxtyped
    def _replace_obj(self, object_id: GUID, new_obj: ModelObject) -> "Scene":
        """Replace an object in the scene.

        Parameters:
          - object_id: the unique identifier of the object to replace.
          - new_obj: the new object.
        """
        return self._replace(
            objects=replace_dict(
                self.objects,
                {object_id: new_obj},
            )
        )

    @jaxtyped
    def set_object_position(
        self,
        object_id: GUID,
        position: Union[Vec3f, Tuple[float, float, float]],
    ) -> "Scene":
        """Set the position of an object in the scene.

        Parameters:
          - object_id: the unique identifier of the object.
          - position: the new position of the object.
        """
        position = jnp.asarray(  # pyright: ignore[reportUnknownMemberType]
            position,
            dtype=float,
        )
        assert isinstance(position, Vec3f), f"{position}"

        new_obj = self.objects[object_id].replace_with_position(position)

        return self._replace_obj(object_id, new_obj)

    @jaxtyped
    def set_object_orientation(
        self,
        object_id: GUID,
        orientation: Optional[Union[Vec4f, Tuple[float, float, float, float]]] = None,
        rotation_matrix: Optional[Float[Array, "3 3"]] = None,
    ) -> "Scene":
        """Set the orientation of an object in the scene.

        If rotation_matrix is specified, it takes precedence over orientation.
        If none is specified, the object's orientation is set to identity.

        Parameters:
          - object_id: the unique identifier of the object.
          - orientation: the new orientation of the object, optional.
          - rotation_matrix: the new rotation matrix of the object, optional
        """
        if orientation is not None:
            orientation = jnp.asarray(  # pyright: ignore[reportUnknownMemberType]
                orientation
            )
        if rotation_matrix is not None:
            rotation_matrix = jnp.asarray(  # pyright: ignore[reportUnknownMemberType]
                rotation_matrix
            )

        new_obj = self.objects[object_id].replace_with_orientation(
            orientation=orientation,
            rotation_matrix=rotation_matrix,
        )

        return self._replace_obj(object_id, new_obj)

    @jaxtyped
    def set_object_local_scaling(
        self,
        object_id: GUID,
        local_scaling: Union[Vec3f, Tuple[float, float, float]],
    ) -> "Scene":
        """Set the local scaling of an object in the scene.

        Parameters:
          - object_id: the unique identifier of the object.
          - local_scaling: the new local scaling of the object.
        """
        scaling = jnp.asarray(  # pyright: ignore[reportUnknownMemberType]
            local_scaling,
            dtype=float,
        )
        assert isinstance(scaling, Vec3f), f"{scaling}"

        new_obj = self.objects[object_id].replace_with_local_scaling(scaling)

        return self._replace_obj(object_id, new_obj)

    @jaxtyped
    def set_object_double_sided(
        self,
        object_id: GUID,
        double_sided: Union[bool, BoolV],
    ) -> "Scene":
        """Set whether an object in the scene is double-sided.

        Parameters:
          - object_id: the unique identifier of the object.
          - double_sided: whether the object is double-sided.
        """
        new_obj = self.objects[object_id].replace_with_double_sided(
            jnp.asarray(double_sided)  # pyright: ignore[reportUnknownMemberType]
        )

        return self._replace_obj(object_id, new_obj)
