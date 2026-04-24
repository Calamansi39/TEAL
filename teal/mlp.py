import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))

import types
from torch import nn

from utils.linear_input_stats import record_linear_input_stats
from utils.utils import ActivationModule, Distribution, SparsifyFn, get_module_device


def _monkeypatch_mlp(mlp, file_path, grabbing_mode=False):
    mlp.forward_old = mlp.forward
    mlp.forward = types.MethodType(_mlp_forward, mlp)

    mlp.file_path = file_path
    mlp.grabbing_mode = grabbing_mode
    mlp.arc_quant_bridge = None
    mlp.layer_idx = None

    if not grabbing_mode:
        mlp.distrs = {}
        mlp.distrs['h1'] = Distribution(file_path, hidden_type='h1')
        mlp.distrs['h2'] = Distribution(file_path, hidden_type='h2')


        mlp.sparse_fns = nn.ModuleDict({
            'gate': SparsifyFn(mlp.distrs['h1']).to(get_module_device(mlp)),
            'up': SparsifyFn(mlp.distrs['h1']).to(get_module_device(mlp)),
            'down': SparsifyFn(mlp.distrs['h2']).to(get_module_device(mlp)),
        })

    mlp.activation_module = ActivationModule(file_path)

    return mlp

def _mlp_forward(self, x, activation_module=None):
    if hasattr(self, 'config') and self.config.pretraining_tp > 1:
        # TODO: UNTESTED

        assert 1 == 0, "Pretraining TP > 1 not implemented yet"
    else:
        if self.grabbing_mode:
            self.activation_module.grab_activations(x, 'h1')

            intermediate_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            self.activation_module.grab_activations(intermediate_states, 'h2')
            down_proj = self.down_proj(intermediate_states)
        else:
            x_gate = self.sparse_fns['gate'](x)
            x_up = self.sparse_fns['up'](x)
            record_linear_input_stats(f"layer_{self.layer_idx}.gate", x_gate)
            record_linear_input_stats(f"layer_{self.layer_idx}.up", x_up)

            gate_key = f"layers.{self.layer_idx}.mlp.gate_proj.input"
            up_key = f"layers.{self.layer_idx}.mlp.up_proj.input"
            down_key = f"layers.{self.layer_idx}.mlp.down_proj.input"
            if self.arc_quant_bridge is not None:
                gate_out = self.arc_quant_bridge.linear(x_gate, self.gate_proj.weight, self.gate_proj.bias, gate_key)
                up_out = self.arc_quant_bridge.linear(x_up, self.up_proj.weight, self.up_proj.bias, up_key)
            else:
                gate_out = self.gate_proj(x_gate)
                up_out = self.up_proj(x_up)

            intermediate_states = self.act_fn(gate_out) * up_out
            intermediate_states = self.sparse_fns['down'](intermediate_states)
            record_linear_input_stats(f"layer_{self.layer_idx}.down", intermediate_states)
            if self.arc_quant_bridge is not None:
                down_proj = self.arc_quant_bridge.linear(
                    intermediate_states,
                    self.down_proj.weight,
                    self.down_proj.bias,
                    down_key,
                )
            else:
                down_proj = self.down_proj(intermediate_states)

    return down_proj
