import types

import torch
from torch import nn

from tangoflux import model as model_module
from tangoflux.model import TangoFlux


class DummyTextEncoder:
    device = torch.device("cpu")


class DummyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_states, **kwargs):
        return (self.scale * hidden_states,)


class DummyScheduler:
    def __init__(self, num_train_timesteps=10):
        self.sigmas = torch.linspace(1.0, 0.1, num_train_timesteps)
        self.timesteps = torch.arange(num_train_timesteps)
        self.config = types.SimpleNamespace(num_train_timesteps=num_train_timesteps)


def build_lightweight_model(in_channels=4, enabled=True, adapter_only=False):
    model = TangoFlux.__new__(TangoFlux)
    nn.Module.__init__(model)

    model.in_channels = in_channels
    model.audio_seq_len = 3
    model.uncondition = False
    model.commutative_adapter_enabled = enabled
    model.commutative_adapter_adapter_only = adapter_only
    model.commutative_adapter_residual_scale = 0.1
    model.lambda_comm = 1e-4
    model.transformer = DummyTransformer()
    model.fc = nn.Identity()
    model.duration_emebdder = nn.Linear(1, 1, bias=False)
    model.noise_scheduler_copy = DummyScheduler()
    model.text_encoder = DummyTextEncoder()

    if enabled:
        model.commutator_A_t = nn.Linear(in_channels, in_channels, bias=False)
        model.commutator_A_f = nn.Linear(in_channels, in_channels, bias=False)
    else:
        model.commutator_A_t = None
        model.commutator_A_f = None

    def encode_text(prompt):
        batch_size = len(prompt)
        encoder_hidden_states = torch.arange(
            batch_size * 2 * in_channels, dtype=torch.float32
        ).reshape(batch_size, 2, in_channels)
        boolean_encoder_mask = torch.ones(batch_size, 2, dtype=torch.bool)
        return encoder_hidden_states, boolean_encoder_mask

    def encode_duration(duration):
        return torch.zeros(duration.shape[0], 1, in_channels, dtype=torch.float32)

    model.encode_text = encode_text
    model.encode_duration = encode_duration
    return model


def test_comm_loss_is_zero_for_identity_operators():
    model = build_lightweight_model(in_channels=2, enabled=True)

    with torch.no_grad():
        nn.init.eye_(model.commutator_A_t.weight)
        nn.init.eye_(model.commutator_A_f.weight)

    comm_loss = model.compute_commutative_loss()

    assert torch.isclose(comm_loss, torch.tensor(0.0), atol=1e-8)


def test_comm_loss_is_positive_for_non_commuting_pair():
    model = build_lightweight_model(in_channels=2, enabled=True)

    with torch.no_grad():
        model.commutator_A_t.weight.copy_(torch.tensor([[0.0, 1.0], [0.0, 0.0]]))
        model.commutator_A_f.weight.copy_(torch.tensor([[0.0, 0.0], [1.0, 0.0]]))

    comm_loss = model.compute_commutative_loss()

    assert comm_loss.item() > 0


def test_adapter_only_mode_leaves_only_adapter_weights_trainable():
    model = build_lightweight_model(enabled=True, adapter_only=True)

    model.configure_trainable_parameters()
    model.validate_adapter_only_trainable()

    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert trainable_names == {"commutator_A_t.weight", "commutator_A_f.weight"}
    assert not any(parameter.requires_grad for parameter in model.transformer.parameters())
    assert not any(parameter.requires_grad for parameter in model.duration_emebdder.parameters())


def test_adapter_only_smoke_step_runs_forward_backward_and_optimizer_step(monkeypatch):
    model = build_lightweight_model(enabled=True, adapter_only=True)
    model.configure_trainable_parameters()
    model.validate_adapter_only_trainable()

    with torch.no_grad():
        nn.init.eye_(model.commutator_A_t.weight)
        nn.init.eye_(model.commutator_A_f.weight)

    monkeypatch.setattr(
        model_module,
        "compute_density_for_timestep_sampling",
        lambda **kwargs: torch.tensor([0.1, 0.4]),
    )

    optimizer = torch.optim.SGD(model.get_optimizer_parameters(), lr=0.1)
    latents = torch.randn(2, model.audio_seq_len, model.in_channels)
    duration = torch.tensor([1.0, 2.0])

    optimizer.zero_grad()
    loss, flow_loss, comm_loss, _ = model(
        latents, ["prompt a", "prompt b"], duration=duration
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(flow_loss)
    assert torch.isfinite(comm_loss)
    assert torch.isclose(loss, flow_loss + (model.lambda_comm * comm_loss))
    assert model.commutator_A_t.weight.grad is not None
    assert model.commutator_A_f.weight.grad is not None
    assert model.transformer.scale.grad is None

    before_step = model.commutator_A_t.weight.detach().clone()
    optimizer.step()

    assert not torch.allclose(before_step, model.commutator_A_t.weight.detach())


def test_strict_false_checkpoint_load_accepts_missing_adapter_weights():
    model = build_lightweight_model(enabled=True, adapter_only=True)
    state_dict = model.state_dict()
    state_dict.pop("commutator_A_t.weight")
    state_dict.pop("commutator_A_f.weight")

    incompatible = model.load_state_dict(state_dict, strict=False)

    assert set(incompatible.missing_keys) == {
        "commutator_A_t.weight",
        "commutator_A_f.weight",
    }
    assert incompatible.unexpected_keys == []
