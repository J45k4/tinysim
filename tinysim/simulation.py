"""Thin simulator wrapper over TinySim's functional tensor core."""

from dataclasses import replace

from tinygrad import Tensor, TinyJit, dtypes

from .actuator import actuator_forces
from .articulated_contact import (
    contacts,
    smooth_free_body_generalized_force,
    smooth_generalized_force,
)
from .compile import CompiledModel, compile_model
from .constraint import joint_limit_rows, solve_contact_impulses
from .dynamics import (
    forward_dynamics,
    inverse_mass_matrix,
    mass_matrix,
)
from .integrator import semi_implicit_euler
from .kinematics import forward_kinematics
from .model import ModelSpec
from .state import (
    RuntimeParameters,
    State,
    make_runtime_parameters,
    make_state,
    validate_state,
)


def step(model: CompiledModel, state: State, control: Tensor | None = None) -> State:
    """Runs one articulated step using the model's compiled contact mode."""
    validate_state(model, state)
    command = state.ctrl if control is None else control
    if command.shape != state.ctrl.shape:
        raise ValueError(f"control must have shape {state.ctrl.shape}")
    if command.dtype != model.dtype or command.device != model.device:
        raise TypeError("control dtype and device must match the compiled model")
    parameters = state.parameters
    generalized_force = actuator_forces(
        model,
        state.qpos,
        state.qvel,
        command,
        None if parameters is None else parameters.actuator_gain,
    )
    contact_batch = None
    if model.contact.mode != "none" and model.collision_pairs:
        kinematics = forward_kinematics(model, state.qpos)
        if model.contact.mode == "smooth":
            independent_free = all(
                parent == -1
                and model.joints[model.joint_for_body[body]].kind == "free"
                for body, parent in enumerate(model.body_parent)
            )
            if independent_free:
                generalized_force = (
                    generalized_force
                    + smooth_free_body_generalized_force(
                        model,
                        kinematics,
                        state.qvel,
                        None
                        if parameters is None
                        else parameters.geom_friction,
                    )
                )
            else:
                contact_batch = contacts(model, kinematics)
                generalized_force = (
                    generalized_force
                    + smooth_generalized_force(
                        model,
                        contact_batch,
                        state.qvel,
                        None
                        if parameters is None
                        else parameters.geom_friction,
                    )
                )
        else:
            contact_batch = contacts(model, kinematics)
    acceleration = forward_dynamics(
        model,
        state.qpos,
        state.qvel,
        generalized_force,
        body_mass=None if parameters is None else parameters.body_mass,
    )
    if model.contact.mode != "constraint" or model.nconstraint == 0:
        return semi_implicit_euler(model, state, acceleration, control=command)

    predicted_velocity = state.qvel + model.timestep * acceleration
    limit_jacobian, limit_velocity, limit_penetration, limit_active = (
        joint_limit_rows(model, state.qpos, predicted_velocity)
    )
    if contact_batch is None:
        jacobian = limit_jacobian
        normal_velocity = limit_velocity
        penetration = limit_penetration
        active = limit_active
    else:
        contact_jacobian = contact_batch.constraint_jacobian
        contact_velocity = (
            contact_jacobian @ predicted_velocity.unsqueeze(-1)
        ).squeeze(-1)
        jacobian = Tensor.cat(contact_jacobian, limit_jacobian, dim=1)
        normal_velocity = Tensor.cat(
            contact_velocity, limit_velocity, dim=1
        )
        penetration = Tensor.cat(
            contact_batch.geometry.penetration, limit_penetration, dim=1
        )
        active = Tensor.cat(
            contact_batch.geometry.active, limit_active, dim=1
        )
    inverse_mass = inverse_mass_matrix(
        mass_matrix(
            model,
            state.qpos,
            body_mass=None if parameters is None else parameters.body_mass,
        )
    )
    initial = state.constraint_impulse
    if initial is None:
        initial = Tensor.zeros(
            state.qpos.shape[0],
            model.nconstraint,
            dtype=model.dtype,
            device=model.device,
        )
    impulses, delta_velocity = solve_contact_impulses(
        inverse_mass,
        jacobian,
        normal_velocity,
        penetration,
        active,
        timestep=model.timestep,
        iterations=model.contact.solver_iterations,
        stabilization=model.contact.stabilization,
        regularization=model.contact.regularization,
        initial_impulse=initial,
    )
    constrained_acceleration = (
        predicted_velocity + delta_velocity - state.qvel
    ) / model.timestep
    return semi_implicit_euler(
        model,
        state,
        constrained_acceleration,
        control=command,
        constraint_impulse=impulses,
    )


def _jitted_step(model: CompiledModel) -> TinyJit:
    def run(
        qpos: Tensor,
        qvel: Tensor,
        control: Tensor,
        time: Tensor,
        constraint_impulse: Tensor,
        body_mass: Tensor,
        geom_friction: Tensor,
        actuator_gain: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        state = State(
            qpos=qpos,
            qvel=qvel,
            ctrl=control,
            time=time,
            constraint_impulse=constraint_impulse,
            parameters=RuntimeParameters(
                body_mass, geom_friction, actuator_gain
            ),
        )
        result = step(model, state, control)
        assert result.constraint_impulse is not None
        Tensor.realize(
            result.qpos, result.qvel, result.time, result.constraint_impulse
        )
        return (
            result.qpos,
            result.qvel,
            result.time,
            result.constraint_impulse,
        )

    return TinyJit(run)


class Simulator:
    """Differentiable simulator specialized to a fixed model and world count."""

    def __init__(self, model: CompiledModel, worlds: int):
        if worlds < 1:
            raise ValueError("worlds must be positive")
        self.model = model
        self.worlds = worlds
        self.parameters = make_runtime_parameters(model, worlds)
        self._inference_step: TinyJit | None = None

    @classmethod
    def compile(
        cls,
        spec: ModelSpec,
        *,
        worlds: int = 1,
        device: str | None = None,
        dtype: object | str = dtypes.float32,
        contact: str | None = None,
        solver_iterations: int | None = None,
    ) -> "Simulator":
        """Compiles the eager path, which preserves tinygrad autodiff."""
        resolved_dtype = {
            "float32": dtypes.float32,
            "float64": dtypes.float64,
        }.get(dtype, dtype)
        if contact is not None or solver_iterations is not None:
            updates = {}
            if contact is not None:
                updates["mode"] = contact
            if solver_iterations is not None:
                updates["solver_iterations"] = solver_iterations
            spec = replace(spec, contact=replace(spec.contact, **updates))
        return cls(compile_model(spec, device=device, dtype=resolved_dtype), worlds)

    def make_state(self) -> State:
        return make_state(self.model, self.worlds)

    def zeros_control(self) -> Tensor:
        return Tensor.zeros(
            self.worlds,
            self.model.nu,
            device=self.model.device,
            dtype=self.model.dtype,
        )

    def step(self, state: State, control: Tensor | None = None) -> State:
        if state.qpos.shape[0] != self.worlds:
            raise ValueError(f"state world dimension must be {self.worlds}")
        return step(self.model, state, control)

    def inference_step(self, state: State, control: Tensor | None = None) -> State:
        """Runs a forward-only TinyJit step.

        This method deliberately detaches its outputs. Capture/replay does not
        preserve tinygrad's autograd graph across the JIT boundary; use
        :meth:`step` for differentiable simulation.
        """
        validate_state(self.model, state)
        command = state.ctrl if control is None else control
        if state.qpos.shape[0] != self.worlds:
            raise ValueError(f"state world dimension must be {self.worlds}")
        if command.shape != state.ctrl.shape:
            raise ValueError(f"control must have shape {state.ctrl.shape}")
        if command.dtype != self.model.dtype or command.device != self.model.device:
            raise TypeError("control dtype and device must match the compiled model")
        if self._inference_step is None:
            self._inference_step = _jitted_step(self.model)
        constraint_impulse = state.constraint_impulse
        if constraint_impulse is None:
            constraint_impulse = Tensor.zeros(
                self.worlds,
                self.model.nconstraint,
                dtype=self.model.dtype,
                device=self.model.device,
            )
        parameters = state.parameters or self.parameters
        qpos, qvel, time, constraint_impulse = self._inference_step(
            state.qpos,
            state.qvel,
            command,
            state.time,
            constraint_impulse,
            parameters.body_mass,
            parameters.geom_friction,
            parameters.actuator_gain,
        )
        return State(
            qpos=qpos.detach(),
            qvel=qvel.detach(),
            ctrl=command,
            time=time.detach(),
            constraint_impulse=constraint_impulse,
            parameters=parameters,
        )

    @staticmethod
    def observe(state: State) -> Tensor:
        return state.qpos.cat(state.qvel, dim=-1)
