graph():
    %x : [#users=1] = placeholder[target=x] | unfixed
    %_is_measure : [#users=0] = placeholder[target=_is_measure](default=False) | unfixed
    %blocks_0_0_layers_0_0 : [#users=1] = call_module[target=blocks.0.0.layers.0.0](args = (%x,), kwargs = {}) | unfixed
    %blocks_0_0_layers_0_1 : [#users=1] = call_module[target=blocks.0.0.layers.0.1](args = (%blocks_0_0_layers_0_0,), kwargs = {}) | unfixed
    %blocks_0_0_layers_0_2 : [#users=1] = call_module[target=blocks.0.0.layers.0.2](args = (%blocks_0_0_layers_0_1,), kwargs = {}) | unfixed
    %blocks_0_0_layers_0_3 : [#users=4] = call_module[target=blocks.0.0.layers.0.3](args = (%blocks_0_0_layers_0_2,), kwargs = {}) | unfixed
    %blocks_0_0_layers_1_net_0 : [#users=1] = call_module[target=blocks.0.0.layers.1.net.0](args = (%blocks_0_0_layers_0_3,), kwargs = {}) | unfixed
    %blocks_0_0_layers_1_net_1 : [#users=1] = call_module[target=blocks.0.0.layers.1.net.1](args = (%blocks_0_0_layers_1_net_0,), kwargs = {}) | unfixed
    %blocks_0_0_layers_1_net_2 : [#users=4] = call_module[target=blocks.0.0.layers.1.net.2](args = (%blocks_0_0_layers_1_net_1,), kwargs = {}) | unfixed
    %blocks_0_0_layers_2_net_0 : [#users=1] = call_module[target=blocks.0.0.layers.2.net.0](args = (%blocks_0_0_layers_1_net_2,), kwargs = {}) | unfixed
    %blocks_0_0_layers_2_net_1 : [#users=1] = call_module[target=blocks.0.0.layers.2.net.1](args = (%blocks_0_0_layers_2_net_0,), kwargs = {}) | unfixed
    %blocks_0_0_layers_2_net_2 : [#users=4] = call_module[target=blocks.0.0.layers.2.net.2](args = (%blocks_0_0_layers_2_net_1,), kwargs = {}) | unfixed
    %blocks_0_0_layers_3_net_0 : [#users=1] = call_module[target=blocks.0.0.layers.3.net.0](args = (%blocks_0_0_layers_2_net_2,), kwargs = {}) | unfixed
    %blocks_0_0_layers_3_net_1 : [#users=1] = call_module[target=blocks.0.0.layers.3.net.1](args = (%blocks_0_0_layers_3_net_0,), kwargs = {}) | unfixed
    %blocks_0_0_layers_3_net_2 : [#users=2] = call_module[target=blocks.0.0.layers.3.net.2](args = (%blocks_0_0_layers_3_net_1,), kwargs = {}) | unfixed
    %blocks_0_1_layers_0_conv_normal_net_0 : [#users=1] = call_module[target=blocks.0.1.layers.0.conv_normal.net.0](args = (%blocks_0_0_layers_0_3,), kwargs = {}) | unfixed
    %blocks_0_1_layers_0_conv_normal_net_1 : [#users=1] = call_module[target=blocks.0.1.layers.0.conv_normal.net.1](args = (%blocks_0_1_layers_0_conv_normal_net_0,), kwargs = {}) | unfixed
    %blocks_0_1_layers_0_conv_normal_net_2 : [#users=1] = call_module[target=blocks.0.1.layers.0.conv_normal.net.2](args = (%blocks_0_1_layers_0_conv_normal_net_1,), kwargs = {}) | unfixed
    %blocks_0_1_layers_0_conv_normal_net_3 : [#users=1] = call_module[target=blocks.0.1.layers.0.conv_normal.net.3](args = (%blocks_0_1_layers_0_conv_normal_net_2,), kwargs = {}) | unfixed
    %blocks_0_1_layers_0_conv_normal_net_4 : [#users=1] = call_module[target=blocks.0.1.layers.0.conv_normal.net.4](args = (%blocks_0_1_layers_0_conv_normal_net_3,), kwargs = {}) | unfixed
    %blocks_0_1_layers_0_conv_normal_net_5 : [#users=1] = call_module[target=blocks.0.1.layers.0.conv_normal.net.5](args = (%blocks_0_1_layers_0_conv_normal_net_4,), kwargs = {}) | unfixed
    %cat : [#users=3] = call_function[target=torch.cat](args = ([%blocks_0_0_layers_0_3, %blocks_0_1_layers_0_conv_normal_net_5],), kwargs = {dim: 1}) | unfixed
    %blocks_0_1_layers_1_conv_down_net_0 : [#users=1] = call_module[target=blocks.0.1.layers.1.conv_down.net.0](args = (%blocks_0_0_layers_0_3,), kwargs = {}) | unfixed
    %blocks_0_1_layers_1_conv_down_net_1 : [#users=1] = call_module[target=blocks.0.1.layers.1.conv_down.net.1](args = (%blocks_0_1_layers_1_conv_down_net_0,), kwargs = {}) | unfixed
    %blocks_0_1_layers_1_conv_down_net_2 : [#users=1] = call_module[target=blocks.0.1.layers.1.conv_down.net.2](args = (%blocks_0_1_layers_1_conv_down_net_1,), kwargs = {}) | unfixed
    %blocks_0_1_layers_1_conv_down_net_3 : [#users=1] = call_module[target=blocks.0.1.layers.1.conv_down.net.3](args = (%blocks_0_1_layers_1_conv_down_net_2,), kwargs = {}) | unfixed
    %blocks_0_1_layers_1_conv_down_net_4 : [#users=1] = call_module[target=blocks.0.1.layers.1.conv_down.net.4](args = (%blocks_0_1_layers_1_conv_down_net_3,), kwargs = {}) | unfixed
    %blocks_0_1_layers_1_conv_down_net_5 : [#users=1] = call_module[target=blocks.0.1.layers.1.conv_down.net.5](args = (%blocks_0_1_layers_1_conv_down_net_4,), kwargs = {}) | unfixed
    %blocks_0_1_layers_1_conv_normal_net_0 : [#users=1] = call_module[target=blocks.0.1.layers.1.conv_normal.net.0](args = (%blocks_0_0_layers_1_net_2,), kwargs = {}) | unfixed
    %blocks_0_1_layers_1_conv_normal_net_1 : [#users=1] = call_module[target=blocks.0.1.layers.1.conv_normal.net.1](args = (%blocks_0_1_layers_1_conv_normal_net_0,), kwargs = {}) | unfixed
    %blocks_0_1_layers_1_conv_normal_net_2 : [#users=1] = call_module[target=blocks.0.1.layers.1.conv_normal.net.2](args = (%blocks_0_1_layers_1_conv_normal_net_1,), kwargs = {}) | unfixed
    %blocks_0_1_layers_1_conv_normal_net_3 : [#users=1] = call_module[target=blocks.0.1.layers.1.conv_normal.net.3](args = (%blocks_0_1_layers_1_conv_normal_net_2,), kwargs = {}) | unfixed
    %blocks_0_1_layers_1_conv_normal_net_4 : [#users=1] = call_module[target=blocks.0.1.layers.1.conv_normal.net.4](args = (%blocks_0_1_layers_1_conv_normal_net_3,), kwargs = {}) | unfixed
    %blocks_0_1_layers_1_conv_normal_net_5 : [#users=1] = call_module[target=blocks.0.1.layers.1.conv_normal.net.5](args = (%blocks_0_1_layers_1_conv_normal_net_4,), kwargs = {}) | unfixed
    %cat_1 : [#users=3] = call_function[target=torch.cat](args = ([%blocks_0_0_layers_1_net_2, %blocks_0_1_layers_1_conv_down_net_5, %blocks_0_1_layers_1_conv_normal_net_5],), kwargs = {dim: 1}) | unfixed
    %blocks_0_1_layers_2_conv_down_net_0 : [#users=1] = call_module[target=blocks.0.1.layers.2.conv_down.net.0](args = (%blocks_0_0_layers_1_net_2,), kwargs = {}) | unfixed
    %blocks_0_1_layers_2_conv_down_net_1 : [#users=1] = call_module[target=blocks.0.1.layers.2.conv_down.net.1](args = (%blocks_0_1_layers_2_conv_down_net_0,), kwargs = {}) | unfixed
    %blocks_0_1_layers_2_conv_down_net_2 : [#users=1] = call_module[target=blocks.0.1.layers.2.conv_down.net.2](args = (%blocks_0_1_layers_2_conv_down_net_1,), kwargs = {}) | unfixed
    %blocks_0_1_layers_2_conv_down_net_3 : [#users=1] = call_module[target=blocks.0.1.layers.2.conv_down.net.3](args = (%blocks_0_1_layers_2_conv_down_net_2,), kwargs = {}) | unfixed
    %blocks_0_1_layers_2_conv_down_net_4 : [#users=1] = call_module[target=blocks.0.1.layers.2.conv_down.net.4](args = (%blocks_0_1_layers_2_conv_down_net_3,), kwargs = {}) | unfixed
    %blocks_0_1_layers_2_conv_down_net_5 : [#users=1] = call_module[target=blocks.0.1.layers.2.conv_down.net.5](args = (%blocks_0_1_layers_2_conv_down_net_4,), kwargs = {}) | unfixed
    %blocks_0_1_layers_2_conv_normal_net_0 : [#users=1] = call_module[target=blocks.0.1.layers.2.conv_normal.net.0](args = (%blocks_0_0_layers_2_net_2,), kwargs = {}) | unfixed
    %blocks_0_1_layers_2_conv_normal_net_1 : [#users=1] = call_module[target=blocks.0.1.layers.2.conv_normal.net.1](args = (%blocks_0_1_layers_2_conv_normal_net_0,), kwargs = {}) | unfixed
    %blocks_0_1_layers_2_conv_normal_net_2 : [#users=1] = call_module[target=blocks.0.1.layers.2.conv_normal.net.2](args = (%blocks_0_1_layers_2_conv_normal_net_1,), kwargs = {}) | unfixed
    %blocks_0_1_layers_2_conv_normal_net_3 : [#users=1] = call_module[target=blocks.0.1.layers.2.conv_normal.net.3](args = (%blocks_0_1_layers_2_conv_normal_net_2,), kwargs = {}) | unfixed
    %blocks_0_1_layers_2_conv_normal_net_4 : [#users=1] = call_module[target=blocks.0.1.layers.2.conv_normal.net.4](args = (%blocks_0_1_layers_2_conv_normal_net_3,), kwargs = {}) | unfixed
    %blocks_0_1_layers_2_conv_normal_net_5 : [#users=1] = call_module[target=blocks.0.1.layers.2.conv_normal.net.5](args = (%blocks_0_1_layers_2_conv_normal_net_4,), kwargs = {}) | unfixed
    %cat_2 : [#users=3] = call_function[target=torch.cat](args = ([%blocks_0_0_layers_2_net_2, %blocks_0_1_layers_2_conv_down_net_5, %blocks_0_1_layers_2_conv_normal_net_5],), kwargs = {dim: 1}) | unfixed
    %blocks_0_1_layers_3_conv_down_net_0 : [#users=1] = call_module[target=blocks.0.1.layers.3.conv_down.net.0](args = (%blocks_0_0_layers_2_net_2,), kwargs = {}) | unfixed
    %blocks_0_1_layers_3_conv_down_net_1 : [#users=1] = call_module[target=blocks.0.1.layers.3.conv_down.net.1](args = (%blocks_0_1_layers_3_conv_down_net_0,), kwargs = {}) | unfixed
    %blocks_0_1_layers_3_conv_down_net_2 : [#users=1] = call_module[target=blocks.0.1.layers.3.conv_down.net.2](args = (%blocks_0_1_layers_3_conv_down_net_1,), kwargs = {}) | unfixed
    %blocks_0_1_layers_3_conv_down_net_3 : [#users=1] = call_module[target=blocks.0.1.layers.3.conv_down.net.3](args = (%blocks_0_1_layers_3_conv_down_net_2,), kwargs = {}) | unfixed
    %blocks_0_1_layers_3_conv_down_net_4 : [#users=1] = call_module[target=blocks.0.1.layers.3.conv_down.net.4](args = (%blocks_0_1_layers_3_conv_down_net_3,), kwargs = {}) | unfixed
    %blocks_0_1_layers_3_conv_down_net_5 : [#users=1] = call_module[target=blocks.0.1.layers.3.conv_down.net.5](args = (%blocks_0_1_layers_3_conv_down_net_4,), kwargs = {}) | unfixed
    %blocks_0_1_layers_3_conv_normal_net_0 : [#users=1] = call_module[target=blocks.0.1.layers.3.conv_normal.net.0](args = (%blocks_0_0_layers_3_net_2,), kwargs = {}) | unfixed
    %blocks_0_1_layers_3_conv_normal_net_1 : [#users=1] = call_module[target=blocks.0.1.layers.3.conv_normal.net.1](args = (%blocks_0_1_layers_3_conv_normal_net_0,), kwargs = {}) | unfixed
    %blocks_0_1_layers_3_conv_normal_net_2 : [#users=1] = call_module[target=blocks.0.1.layers.3.conv_normal.net.2](args = (%blocks_0_1_layers_3_conv_normal_net_1,), kwargs = {}) | unfixed
    %blocks_0_1_layers_3_conv_normal_net_3 : [#users=1] = call_module[target=blocks.0.1.layers.3.conv_normal.net.3](args = (%blocks_0_1_layers_3_conv_normal_net_2,), kwargs = {}) | unfixed
    %blocks_0_1_layers_3_conv_normal_net_4 : [#users=1] = call_module[target=blocks.0.1.layers.3.conv_normal.net.4](args = (%blocks_0_1_layers_3_conv_normal_net_3,), kwargs = {}) | unfixed
    %blocks_0_1_layers_3_conv_normal_net_5 : [#users=1] = call_module[target=blocks.0.1.layers.3.conv_normal.net.5](args = (%blocks_0_1_layers_3_conv_normal_net_4,), kwargs = {}) | unfixed
    %cat_3 : [#users=2] = call_function[target=torch.cat](args = ([%blocks_0_0_layers_3_net_2, %blocks_0_1_layers_3_conv_down_net_5, %blocks_0_1_layers_3_conv_normal_net_5],), kwargs = {dim: 1}) | unfixed
    %blocks_0_2_layers_0_conv_normal_net_0 : [#users=1] = call_module[target=blocks.0.2.layers.0.conv_normal.net.0](args = (%cat,), kwargs = {}) | unfixed
    %blocks_0_2_layers_0_conv_normal_net_1 : [#users=1] = call_module[target=blocks.0.2.layers.0.conv_normal.net.1](args = (%blocks_0_2_layers_0_conv_normal_net_0,), kwargs = {}) | unfixed
    %blocks_0_2_layers_0_conv_normal_net_2 : [#users=1] = call_module[target=blocks.0.2.layers.0.conv_normal.net.2](args = (%blocks_0_2_layers_0_conv_normal_net_1,), kwargs = {}) | unfixed
    %blocks_0_2_layers_0_conv_normal_net_3 : [#users=1] = call_module[target=blocks.0.2.layers.0.conv_normal.net.3](args = (%blocks_0_2_layers_0_conv_normal_net_2,), kwargs = {}) | unfixed
    %blocks_0_2_layers_0_conv_normal_net_4 : [#users=1] = call_module[target=blocks.0.2.layers.0.conv_normal.net.4](args = (%blocks_0_2_layers_0_conv_normal_net_3,), kwargs = {}) | unfixed
    %blocks_0_2_layers_0_conv_normal_net_5 : [#users=1] = call_module[target=blocks.0.2.layers.0.conv_normal.net.5](args = (%blocks_0_2_layers_0_conv_normal_net_4,), kwargs = {}) | unfixed
    %cat_4 : [#users=3] = call_function[target=torch.cat](args = ([%cat, %blocks_0_2_layers_0_conv_normal_net_5],), kwargs = {dim: 1}) | unfixed
    %blocks_0_2_layers_1_conv_down_net_0 : [#users=1] = call_module[target=blocks.0.2.layers.1.conv_down.net.0](args = (%cat,), kwargs = {}) | unfixed
    %blocks_0_2_layers_1_conv_down_net_1 : [#users=1] = call_module[target=blocks.0.2.layers.1.conv_down.net.1](args = (%blocks_0_2_layers_1_conv_down_net_0,), kwargs = {}) | unfixed
    %blocks_0_2_layers_1_conv_down_net_2 : [#users=1] = call_module[target=blocks.0.2.layers.1.conv_down.net.2](args = (%blocks_0_2_layers_1_conv_down_net_1,), kwargs = {}) | unfixed
    %blocks_0_2_layers_1_conv_down_net_3 : [#users=1] = call_module[target=blocks.0.2.layers.1.conv_down.net.3](args = (%blocks_0_2_layers_1_conv_down_net_2,), kwargs = {}) | unfixed
    %blocks_0_2_layers_1_conv_down_net_4 : [#users=1] = call_module[target=blocks.0.2.layers.1.conv_down.net.4](args = (%blocks_0_2_layers_1_conv_down_net_3,), kwargs = {}) | unfixed
    %blocks_0_2_layers_1_conv_down_net_5 : [#users=1] = call_module[target=blocks.0.2.layers.1.conv_down.net.5](args = (%blocks_0_2_layers_1_conv_down_net_4,), kwargs = {}) | unfixed
    %blocks_0_2_layers_1_conv_normal_net_0 : [#users=1] = call_module[target=blocks.0.2.layers.1.conv_normal.net.0](args = (%cat_1,), kwargs = {}) | unfixed
    %blocks_0_2_layers_1_conv_normal_net_1 : [#users=1] = call_module[target=blocks.0.2.layers.1.conv_normal.net.1](args = (%blocks_0_2_layers_1_conv_normal_net_0,), kwargs = {}) | unfixed
    %blocks_0_2_layers_1_conv_normal_net_2 : [#users=1] = call_module[target=blocks.0.2.layers.1.conv_normal.net.2](args = (%blocks_0_2_layers_1_conv_normal_net_1,), kwargs = {}) | unfixed
    %blocks_0_2_layers_1_conv_normal_net_3 : [#users=1] = call_module[target=blocks.0.2.layers.1.conv_normal.net.3](args = (%blocks_0_2_layers_1_conv_normal_net_2,), kwargs = {}) | unfixed
    %blocks_0_2_layers_1_conv_normal_net_4 : [#users=1] = call_module[target=blocks.0.2.layers.1.conv_normal.net.4](args = (%blocks_0_2_layers_1_conv_normal_net_3,), kwargs = {}) | unfixed
    %blocks_0_2_layers_1_conv_normal_net_5 : [#users=1] = call_module[target=blocks.0.2.layers.1.conv_normal.net.5](args = (%blocks_0_2_layers_1_conv_normal_net_4,), kwargs = {}) | unfixed
    %cat_5 : [#users=3] = call_function[target=torch.cat](args = ([%cat_1, %blocks_0_2_layers_1_conv_down_net_5, %blocks_0_2_layers_1_conv_normal_net_5],), kwargs = {dim: 1}) | unfixed
    %blocks_0_2_layers_2_conv_down_net_0 : [#users=1] = call_module[target=blocks.0.2.layers.2.conv_down.net.0](args = (%cat_1,), kwargs = {}) | unfixed
    %blocks_0_2_layers_2_conv_down_net_1 : [#users=1] = call_module[target=blocks.0.2.layers.2.conv_down.net.1](args = (%blocks_0_2_layers_2_conv_down_net_0,), kwargs = {}) | unfixed
    %blocks_0_2_layers_2_conv_down_net_2 : [#users=1] = call_module[target=blocks.0.2.layers.2.conv_down.net.2](args = (%blocks_0_2_layers_2_conv_down_net_1,), kwargs = {}) | unfixed
    %blocks_0_2_layers_2_conv_down_net_3 : [#users=1] = call_module[target=blocks.0.2.layers.2.conv_down.net.3](args = (%blocks_0_2_layers_2_conv_down_net_2,), kwargs = {}) | unfixed
    %blocks_0_2_layers_2_conv_down_net_4 : [#users=1] = call_module[target=blocks.0.2.layers.2.conv_down.net.4](args = (%blocks_0_2_layers_2_conv_down_net_3,), kwargs = {}) | unfixed
    %blocks_0_2_layers_2_conv_down_net_5 : [#users=1] = call_module[target=blocks.0.2.layers.2.conv_down.net.5](args = (%blocks_0_2_layers_2_conv_down_net_4,), kwargs = {}) | unfixed
    %blocks_0_2_layers_2_conv_normal_net_0 : [#users=1] = call_module[target=blocks.0.2.layers.2.conv_normal.net.0](args = (%cat_2,), kwargs = {}) | unfixed
    %blocks_0_2_layers_2_conv_normal_net_1 : [#users=1] = call_module[target=blocks.0.2.layers.2.conv_normal.net.1](args = (%blocks_0_2_layers_2_conv_normal_net_0,), kwargs = {}) | unfixed
    %blocks_0_2_layers_2_conv_normal_net_2 : [#users=1] = call_module[target=blocks.0.2.layers.2.conv_normal.net.2](args = (%blocks_0_2_layers_2_conv_normal_net_1,), kwargs = {}) | unfixed
    %blocks_0_2_layers_2_conv_normal_net_3 : [#users=1] = call_module[target=blocks.0.2.layers.2.conv_normal.net.3](args = (%blocks_0_2_layers_2_conv_normal_net_2,), kwargs = {}) | unfixed
    %blocks_0_2_layers_2_conv_normal_net_4 : [#users=1] = call_module[target=blocks.0.2.layers.2.conv_normal.net.4](args = (%blocks_0_2_layers_2_conv_normal_net_3,), kwargs = {}) | unfixed
    %blocks_0_2_layers_2_conv_normal_net_5 : [#users=1] = call_module[target=blocks.0.2.layers.2.conv_normal.net.5](args = (%blocks_0_2_layers_2_conv_normal_net_4,), kwargs = {}) | unfixed
    %cat_6 : [#users=3] = call_function[target=torch.cat](args = ([%cat_2, %blocks_0_2_layers_2_conv_down_net_5, %blocks_0_2_layers_2_conv_normal_net_5],), kwargs = {dim: 1}) | unfixed
    %blocks_0_2_layers_3_conv_down_net_0 : [#users=1] = call_module[target=blocks.0.2.layers.3.conv_down.net.0](args = (%cat_2,), kwargs = {}) | unfixed
    %blocks_0_2_layers_3_conv_down_net_1 : [#users=1] = call_module[target=blocks.0.2.layers.3.conv_down.net.1](args = (%blocks_0_2_layers_3_conv_down_net_0,), kwargs = {}) | unfixed
    %blocks_0_2_layers_3_conv_down_net_2 : [#users=1] = call_module[target=blocks.0.2.layers.3.conv_down.net.2](args = (%blocks_0_2_layers_3_conv_down_net_1,), kwargs = {}) | unfixed
    %blocks_0_2_layers_3_conv_down_net_3 : [#users=1] = call_module[target=blocks.0.2.layers.3.conv_down.net.3](args = (%blocks_0_2_layers_3_conv_down_net_2,), kwargs = {}) | unfixed
    %blocks_0_2_layers_3_conv_down_net_4 : [#users=1] = call_module[target=blocks.0.2.layers.3.conv_down.net.4](args = (%blocks_0_2_layers_3_conv_down_net_3,), kwargs = {}) | unfixed
    %blocks_0_2_layers_3_conv_down_net_5 : [#users=1] = call_module[target=blocks.0.2.layers.3.conv_down.net.5](args = (%blocks_0_2_layers_3_conv_down_net_4,), kwargs = {}) | unfixed
    %blocks_0_2_layers_3_conv_normal_net_0 : [#users=1] = call_module[target=blocks.0.2.layers.3.conv_normal.net.0](args = (%cat_3,), kwargs = {}) | unfixed
    %blocks_0_2_layers_3_conv_normal_net_1 : [#users=1] = call_module[target=blocks.0.2.layers.3.conv_normal.net.1](args = (%blocks_0_2_layers_3_conv_normal_net_0,), kwargs = {}) | unfixed
    %blocks_0_2_layers_3_conv_normal_net_2 : [#users=1] = call_module[target=blocks.0.2.layers.3.conv_normal.net.2](args = (%blocks_0_2_layers_3_conv_normal_net_1,), kwargs = {}) | unfixed
    %blocks_0_2_layers_3_conv_normal_net_3 : [#users=1] = call_module[target=blocks.0.2.layers.3.conv_normal.net.3](args = (%blocks_0_2_layers_3_conv_normal_net_2,), kwargs = {}) | unfixed
    %blocks_0_2_layers_3_conv_normal_net_4 : [#users=1] = call_module[target=blocks.0.2.layers.3.conv_normal.net.4](args = (%blocks_0_2_layers_3_conv_normal_net_3,), kwargs = {}) | unfixed
    %blocks_0_2_layers_3_conv_normal_net_5 : [#users=1] = call_module[target=blocks.0.2.layers.3.conv_normal.net.5](args = (%blocks_0_2_layers_3_conv_normal_net_4,), kwargs = {}) | unfixed
    %cat_7 : [#users=2] = call_function[target=torch.cat](args = ([%cat_3, %blocks_0_2_layers_3_conv_down_net_5, %blocks_0_2_layers_3_conv_normal_net_5],), kwargs = {dim: 1}) | unfixed
    %blocks_0_3_layers_0_conv_normal_net_0 : [#users=1] = call_module[target=blocks.0.3.layers.0.conv_normal.net.0](args = (%cat_4,), kwargs = {}) | unfixed
    %blocks_0_3_layers_0_conv_normal_net_1 : [#users=1] = call_module[target=blocks.0.3.layers.0.conv_normal.net.1](args = (%blocks_0_3_layers_0_conv_normal_net_0,), kwargs = {}) | unfixed
    %blocks_0_3_layers_0_conv_normal_net_2 : [#users=1] = call_module[target=blocks.0.3.layers.0.conv_normal.net.2](args = (%blocks_0_3_layers_0_conv_normal_net_1,), kwargs = {}) | unfixed
    %blocks_0_3_layers_0_conv_normal_net_3 : [#users=1] = call_module[target=blocks.0.3.layers.0.conv_normal.net.3](args = (%blocks_0_3_layers_0_conv_normal_net_2,), kwargs = {}) | unfixed
    %blocks_0_3_layers_0_conv_normal_net_4 : [#users=1] = call_module[target=blocks.0.3.layers.0.conv_normal.net.4](args = (%blocks_0_3_layers_0_conv_normal_net_3,), kwargs = {}) | unfixed
    %blocks_0_3_layers_0_conv_normal_net_5 : [#users=1] = call_module[target=blocks.0.3.layers.0.conv_normal.net.5](args = (%blocks_0_3_layers_0_conv_normal_net_4,), kwargs = {}) | unfixed
    %cat_8 : [#users=3] = call_function[target=torch.cat](args = ([%cat_4, %blocks_0_3_layers_0_conv_normal_net_5],), kwargs = {dim: 1}) | unfixed
    %blocks_0_3_layers_1_conv_down_net_0 : [#users=1] = call_module[target=blocks.0.3.layers.1.conv_down.net.0](args = (%cat_4,), kwargs = {}) | unfixed
    %blocks_0_3_layers_1_conv_down_net_1 : [#users=1] = call_module[target=blocks.0.3.layers.1.conv_down.net.1](args = (%blocks_0_3_layers_1_conv_down_net_0,), kwargs = {}) | unfixed
    %blocks_0_3_layers_1_conv_down_net_2 : [#users=1] = call_module[target=blocks.0.3.layers.1.conv_down.net.2](args = (%blocks_0_3_layers_1_conv_down_net_1,), kwargs = {}) | unfixed
    %blocks_0_3_layers_1_conv_down_net_3 : [#users=1] = call_module[target=blocks.0.3.layers.1.conv_down.net.3](args = (%blocks_0_3_layers_1_conv_down_net_2,), kwargs = {}) | unfixed
    %blocks_0_3_layers_1_conv_down_net_4 : [#users=1] = call_module[target=blocks.0.3.layers.1.conv_down.net.4](args = (%blocks_0_3_layers_1_conv_down_net_3,), kwargs = {}) | unfixed
    %blocks_0_3_layers_1_conv_down_net_5 : [#users=1] = call_module[target=blocks.0.3.layers.1.conv_down.net.5](args = (%blocks_0_3_layers_1_conv_down_net_4,), kwargs = {}) | unfixed
    %blocks_0_3_layers_1_conv_normal_net_0 : [#users=1] = call_module[target=blocks.0.3.layers.1.conv_normal.net.0](args = (%cat_5,), kwargs = {}) | unfixed
    %blocks_0_3_layers_1_conv_normal_net_1 : [#users=1] = call_module[target=blocks.0.3.layers.1.conv_normal.net.1](args = (%blocks_0_3_layers_1_conv_normal_net_0,), kwargs = {}) | unfixed
    %blocks_0_3_layers_1_conv_normal_net_2 : [#users=1] = call_module[target=blocks.0.3.layers.1.conv_normal.net.2](args = (%blocks_0_3_layers_1_conv_normal_net_1,), kwargs = {}) | unfixed
    %blocks_0_3_layers_1_conv_normal_net_3 : [#users=1] = call_module[target=blocks.0.3.layers.1.conv_normal.net.3](args = (%blocks_0_3_layers_1_conv_normal_net_2,), kwargs = {}) | unfixed
    %blocks_0_3_layers_1_conv_normal_net_4 : [#users=1] = call_module[target=blocks.0.3.layers.1.conv_normal.net.4](args = (%blocks_0_3_layers_1_conv_normal_net_3,), kwargs = {}) | unfixed
    %blocks_0_3_layers_1_conv_normal_net_5 : [#users=1] = call_module[target=blocks.0.3.layers.1.conv_normal.net.5](args = (%blocks_0_3_layers_1_conv_normal_net_4,), kwargs = {}) | unfixed
    %cat_9 : [#users=3] = call_function[target=torch.cat](args = ([%cat_5, %blocks_0_3_layers_1_conv_down_net_5, %blocks_0_3_layers_1_conv_normal_net_5],), kwargs = {dim: 1}) | unfixed
    %blocks_0_3_layers_2_conv_down_net_0 : [#users=1] = call_module[target=blocks.0.3.layers.2.conv_down.net.0](args = (%cat_5,), kwargs = {}) | unfixed
    %blocks_0_3_layers_2_conv_down_net_1 : [#users=1] = call_module[target=blocks.0.3.layers.2.conv_down.net.1](args = (%blocks_0_3_layers_2_conv_down_net_0,), kwargs = {}) | unfixed
    %blocks_0_3_layers_2_conv_down_net_2 : [#users=1] = call_module[target=blocks.0.3.layers.2.conv_down.net.2](args = (%blocks_0_3_layers_2_conv_down_net_1,), kwargs = {}) | unfixed
    %blocks_0_3_layers_2_conv_down_net_3 : [#users=1] = call_module[target=blocks.0.3.layers.2.conv_down.net.3](args = (%blocks_0_3_layers_2_conv_down_net_2,), kwargs = {}) | unfixed
    %blocks_0_3_layers_2_conv_down_net_4 : [#users=1] = call_module[target=blocks.0.3.layers.2.conv_down.net.4](args = (%blocks_0_3_layers_2_conv_down_net_3,), kwargs = {}) | unfixed
    %blocks_0_3_layers_2_conv_down_net_5 : [#users=1] = call_module[target=blocks.0.3.layers.2.conv_down.net.5](args = (%blocks_0_3_layers_2_conv_down_net_4,), kwargs = {}) | unfixed
    %blocks_0_3_layers_2_conv_normal_net_0 : [#users=1] = call_module[target=blocks.0.3.layers.2.conv_normal.net.0](args = (%cat_6,), kwargs = {}) | unfixed
    %blocks_0_3_layers_2_conv_normal_net_1 : [#users=1] = call_module[target=blocks.0.3.layers.2.conv_normal.net.1](args = (%blocks_0_3_layers_2_conv_normal_net_0,), kwargs = {}) | unfixed
    %blocks_0_3_layers_2_conv_normal_net_2 : [#users=1] = call_module[target=blocks.0.3.layers.2.conv_normal.net.2](args = (%blocks_0_3_layers_2_conv_normal_net_1,), kwargs = {}) | unfixed
    %blocks_0_3_layers_2_conv_normal_net_3 : [#users=1] = call_module[target=blocks.0.3.layers.2.conv_normal.net.3](args = (%blocks_0_3_layers_2_conv_normal_net_2,), kwargs = {}) | unfixed
    %blocks_0_3_layers_2_conv_normal_net_4 : [#users=1] = call_module[target=blocks.0.3.layers.2.conv_normal.net.4](args = (%blocks_0_3_layers_2_conv_normal_net_3,), kwargs = {}) | unfixed
    %blocks_0_3_layers_2_conv_normal_net_5 : [#users=1] = call_module[target=blocks.0.3.layers.2.conv_normal.net.5](args = (%blocks_0_3_layers_2_conv_normal_net_4,), kwargs = {}) | unfixed
    %cat_10 : [#users=3] = call_function[target=torch.cat](args = ([%cat_6, %blocks_0_3_layers_2_conv_down_net_5, %blocks_0_3_layers_2_conv_normal_net_5],), kwargs = {dim: 1}) | unfixed
    %blocks_0_3_layers_3_conv_down_net_0 : [#users=1] = call_module[target=blocks.0.3.layers.3.conv_down.net.0](args = (%cat_6,), kwargs = {}) | unfixed
    %blocks_0_3_layers_3_conv_down_net_1 : [#users=1] = call_module[target=blocks.0.3.layers.3.conv_down.net.1](args = (%blocks_0_3_layers_3_conv_down_net_0,), kwargs = {}) | unfixed
    %blocks_0_3_layers_3_conv_down_net_2 : [#users=1] = call_module[target=blocks.0.3.layers.3.conv_down.net.2](args = (%blocks_0_3_layers_3_conv_down_net_1,), kwargs = {}) | unfixed
    %blocks_0_3_layers_3_conv_down_net_3 : [#users=1] = call_module[target=blocks.0.3.layers.3.conv_down.net.3](args = (%blocks_0_3_layers_3_conv_down_net_2,), kwargs = {}) | unfixed
    %blocks_0_3_layers_3_conv_down_net_4 : [#users=1] = call_module[target=blocks.0.3.layers.3.conv_down.net.4](args = (%blocks_0_3_layers_3_conv_down_net_3,), kwargs = {}) | unfixed
    %blocks_0_3_layers_3_conv_down_net_5 : [#users=1] = call_module[target=blocks.0.3.layers.3.conv_down.net.5](args = (%blocks_0_3_layers_3_conv_down_net_4,), kwargs = {}) | unfixed
    %blocks_0_3_layers_3_conv_normal_net_0 : [#users=1] = call_module[target=blocks.0.3.layers.3.conv_normal.net.0](args = (%cat_7,), kwargs = {}) | unfixed
    %blocks_0_3_layers_3_conv_normal_net_1 : [#users=1] = call_module[target=blocks.0.3.layers.3.conv_normal.net.1](args = (%blocks_0_3_layers_3_conv_normal_net_0,), kwargs = {}) | unfixed
    %blocks_0_3_layers_3_conv_normal_net_2 : [#users=1] = call_module[target=blocks.0.3.layers.3.conv_normal.net.2](args = (%blocks_0_3_layers_3_conv_normal_net_1,), kwargs = {}) | unfixed
    %blocks_0_3_layers_3_conv_normal_net_3 : [#users=1] = call_module[target=blocks.0.3.layers.3.conv_normal.net.3](args = (%blocks_0_3_layers_3_conv_normal_net_2,), kwargs = {}) | unfixed
    %blocks_0_3_layers_3_conv_normal_net_4 : [#users=1] = call_module[target=blocks.0.3.layers.3.conv_normal.net.4](args = (%blocks_0_3_layers_3_conv_normal_net_3,), kwargs = {}) | unfixed
    %blocks_0_3_layers_3_conv_normal_net_5 : [#users=1] = call_module[target=blocks.0.3.layers.3.conv_normal.net.5](args = (%blocks_0_3_layers_3_conv_normal_net_4,), kwargs = {}) | unfixed
    %cat_11 : [#users=2] = call_function[target=torch.cat](args = ([%cat_7, %blocks_0_3_layers_3_conv_down_net_5, %blocks_0_3_layers_3_conv_normal_net_5],), kwargs = {dim: 1}) | unfixed
    %blocks_0_4_layers_0_conv_normal_net_0 : [#users=1] = call_module[target=blocks.0.4.layers.0.conv_normal.net.0](args = (%cat_8,), kwargs = {}) | unfixed
    %blocks_0_4_layers_0_conv_normal_net_1 : [#users=1] = call_module[target=blocks.0.4.layers.0.conv_normal.net.1](args = (%blocks_0_4_layers_0_conv_normal_net_0,), kwargs = {}) | unfixed
    %blocks_0_4_layers_0_conv_normal_net_2 : [#users=1] = call_module[target=blocks.0.4.layers.0.conv_normal.net.2](args = (%blocks_0_4_layers_0_conv_normal_net_1,), kwargs = {}) | unfixed
    %blocks_0_4_layers_0_conv_normal_net_3 : [#users=1] = call_module[target=blocks.0.4.layers.0.conv_normal.net.3](args = (%blocks_0_4_layers_0_conv_normal_net_2,), kwargs = {}) | unfixed
    %blocks_0_4_layers_0_conv_normal_net_4 : [#users=1] = call_module[target=blocks.0.4.layers.0.conv_normal.net.4](args = (%blocks_0_4_layers_0_conv_normal_net_3,), kwargs = {}) | unfixed
    %blocks_0_4_layers_0_conv_normal_net_5 : [#users=1] = call_module[target=blocks.0.4.layers.0.conv_normal.net.5](args = (%blocks_0_4_layers_0_conv_normal_net_4,), kwargs = {}) | unfixed
    %cat_12 : [#users=1] = call_function[target=torch.cat](args = ([%cat_8, %blocks_0_4_layers_0_conv_normal_net_5],), kwargs = {dim: 1}) | unfixed
    %blocks_0_4_layers_1_conv_down_net_0 : [#users=1] = call_module[target=blocks.0.4.layers.1.conv_down.net.0](args = (%cat_8,), kwargs = {}) | unfixed
    %blocks_0_4_layers_1_conv_down_net_1 : [#users=1] = call_module[target=blocks.0.4.layers.1.conv_down.net.1](args = (%blocks_0_4_layers_1_conv_down_net_0,), kwargs = {}) | unfixed
    %blocks_0_4_layers_1_conv_down_net_2 : [#users=1] = call_module[target=blocks.0.4.layers.1.conv_down.net.2](args = (%blocks_0_4_layers_1_conv_down_net_1,), kwargs = {}) | unfixed
    %blocks_0_4_layers_1_conv_down_net_3 : [#users=1] = call_module[target=blocks.0.4.layers.1.conv_down.net.3](args = (%blocks_0_4_layers_1_conv_down_net_2,), kwargs = {}) | unfixed
    %blocks_0_4_layers_1_conv_down_net_4 : [#users=1] = call_module[target=blocks.0.4.layers.1.conv_down.net.4](args = (%blocks_0_4_layers_1_conv_down_net_3,), kwargs = {}) | unfixed
    %blocks_0_4_layers_1_conv_down_net_5 : [#users=1] = call_module[target=blocks.0.4.layers.1.conv_down.net.5](args = (%blocks_0_4_layers_1_conv_down_net_4,), kwargs = {}) | unfixed
    %blocks_0_4_layers_1_conv_normal_net_0 : [#users=1] = call_module[target=blocks.0.4.layers.1.conv_normal.net.0](args = (%cat_9,), kwargs = {}) | unfixed
    %blocks_0_4_layers_1_conv_normal_net_1 : [#users=1] = call_module[target=blocks.0.4.layers.1.conv_normal.net.1](args = (%blocks_0_4_layers_1_conv_normal_net_0,), kwargs = {}) | unfixed
    %blocks_0_4_layers_1_conv_normal_net_2 : [#users=1] = call_module[target=blocks.0.4.layers.1.conv_normal.net.2](args = (%blocks_0_4_layers_1_conv_normal_net_1,), kwargs = {}) | unfixed
    %blocks_0_4_layers_1_conv_normal_net_3 : [#users=1] = call_module[target=blocks.0.4.layers.1.conv_normal.net.3](args = (%blocks_0_4_layers_1_conv_normal_net_2,), kwargs = {}) | unfixed
    %blocks_0_4_layers_1_conv_normal_net_4 : [#users=1] = call_module[target=blocks.0.4.layers.1.conv_normal.net.4](args = (%blocks_0_4_layers_1_conv_normal_net_3,), kwargs = {}) | unfixed
    %blocks_0_4_layers_1_conv_normal_net_5 : [#users=1] = call_module[target=blocks.0.4.layers.1.conv_normal.net.5](args = (%blocks_0_4_layers_1_conv_normal_net_4,), kwargs = {}) | unfixed
    %cat_13 : [#users=1] = call_function[target=torch.cat](args = ([%cat_9, %blocks_0_4_layers_1_conv_down_net_5, %blocks_0_4_layers_1_conv_normal_net_5],), kwargs = {dim: 1}) | unfixed
    %blocks_0_4_layers_2_conv_down_net_0 : [#users=1] = call_module[target=blocks.0.4.layers.2.conv_down.net.0](args = (%cat_9,), kwargs = {}) | unfixed
    %blocks_0_4_layers_2_conv_down_net_1 : [#users=1] = call_module[target=blocks.0.4.layers.2.conv_down.net.1](args = (%blocks_0_4_layers_2_conv_down_net_0,), kwargs = {}) | unfixed
    %blocks_0_4_layers_2_conv_down_net_2 : [#users=1] = call_module[target=blocks.0.4.layers.2.conv_down.net.2](args = (%blocks_0_4_layers_2_conv_down_net_1,), kwargs = {}) | unfixed
    %blocks_0_4_layers_2_conv_down_net_3 : [#users=1] = call_module[target=blocks.0.4.layers.2.conv_down.net.3](args = (%blocks_0_4_layers_2_conv_down_net_2,), kwargs = {}) | unfixed
    %blocks_0_4_layers_2_conv_down_net_4 : [#users=1] = call_module[target=blocks.0.4.layers.2.conv_down.net.4](args = (%blocks_0_4_layers_2_conv_down_net_3,), kwargs = {}) | unfixed
    %blocks_0_4_layers_2_conv_down_net_5 : [#users=1] = call_module[target=blocks.0.4.layers.2.conv_down.net.5](args = (%blocks_0_4_layers_2_conv_down_net_4,), kwargs = {}) | unfixed
    %blocks_0_4_layers_2_conv_normal_net_0 : [#users=1] = call_module[target=blocks.0.4.layers.2.conv_normal.net.0](args = (%cat_10,), kwargs = {}) | unfixed
    %blocks_0_4_layers_2_conv_normal_net_1 : [#users=1] = call_module[target=blocks.0.4.layers.2.conv_normal.net.1](args = (%blocks_0_4_layers_2_conv_normal_net_0,), kwargs = {}) | unfixed
    %blocks_0_4_layers_2_conv_normal_net_2 : [#users=1] = call_module[target=blocks.0.4.layers.2.conv_normal.net.2](args = (%blocks_0_4_layers_2_conv_normal_net_1,), kwargs = {}) | unfixed
    %blocks_0_4_layers_2_conv_normal_net_3 : [#users=1] = call_module[target=blocks.0.4.layers.2.conv_normal.net.3](args = (%blocks_0_4_layers_2_conv_normal_net_2,), kwargs = {}) | unfixed
    %blocks_0_4_layers_2_conv_normal_net_4 : [#users=1] = call_module[target=blocks.0.4.layers.2.conv_normal.net.4](args = (%blocks_0_4_layers_2_conv_normal_net_3,), kwargs = {}) | unfixed
    %blocks_0_4_layers_2_conv_normal_net_5 : [#users=1] = call_module[target=blocks.0.4.layers.2.conv_normal.net.5](args = (%blocks_0_4_layers_2_conv_normal_net_4,), kwargs = {}) | unfixed
    %cat_14 : [#users=1] = call_function[target=torch.cat](args = ([%cat_10, %blocks_0_4_layers_2_conv_down_net_5, %blocks_0_4_layers_2_conv_normal_net_5],), kwargs = {dim: 1}) | unfixed
    %blocks_0_4_layers_3_conv_down_net_0 : [#users=1] = call_module[target=blocks.0.4.layers.3.conv_down.net.0](args = (%cat_10,), kwargs = {}) | unfixed
    %blocks_0_4_layers_3_conv_down_net_1 : [#users=1] = call_module[target=blocks.0.4.layers.3.conv_down.net.1](args = (%blocks_0_4_layers_3_conv_down_net_0,), kwargs = {}) | unfixed
    %blocks_0_4_layers_3_conv_down_net_2 : [#users=1] = call_module[target=blocks.0.4.layers.3.conv_down.net.2](args = (%blocks_0_4_layers_3_conv_down_net_1,), kwargs = {}) | unfixed
    %blocks_0_4_layers_3_conv_down_net_3 : [#users=1] = call_module[target=blocks.0.4.layers.3.conv_down.net.3](args = (%blocks_0_4_layers_3_conv_down_net_2,), kwargs = {}) | unfixed
    %blocks_0_4_layers_3_conv_down_net_4 : [#users=1] = call_module[target=blocks.0.4.layers.3.conv_down.net.4](args = (%blocks_0_4_layers_3_conv_down_net_3,), kwargs = {}) | unfixed
    %blocks_0_4_layers_3_conv_down_net_5 : [#users=1] = call_module[target=blocks.0.4.layers.3.conv_down.net.5](args = (%blocks_0_4_layers_3_conv_down_net_4,), kwargs = {}) | unfixed
    %blocks_0_4_layers_3_conv_normal_net_0 : [#users=1] = call_module[target=blocks.0.4.layers.3.conv_normal.net.0](args = (%cat_11,), kwargs = {}) | unfixed
    %blocks_0_4_layers_3_conv_normal_net_1 : [#users=1] = call_module[target=blocks.0.4.layers.3.conv_normal.net.1](args = (%blocks_0_4_layers_3_conv_normal_net_0,), kwargs = {}) | unfixed
    %blocks_0_4_layers_3_conv_normal_net_2 : [#users=1] = call_module[target=blocks.0.4.layers.3.conv_normal.net.2](args = (%blocks_0_4_layers_3_conv_normal_net_1,), kwargs = {}) | unfixed
    %blocks_0_4_layers_3_conv_normal_net_3 : [#users=1] = call_module[target=blocks.0.4.layers.3.conv_normal.net.3](args = (%blocks_0_4_layers_3_conv_normal_net_2,), kwargs = {}) | unfixed
    %blocks_0_4_layers_3_conv_normal_net_4 : [#users=1] = call_module[target=blocks.0.4.layers.3.conv_normal.net.4](args = (%blocks_0_4_layers_3_conv_normal_net_3,), kwargs = {}) | unfixed
    %blocks_0_4_layers_3_conv_normal_net_5 : [#users=1] = call_module[target=blocks.0.4.layers.3.conv_normal.net.5](args = (%blocks_0_4_layers_3_conv_normal_net_4,), kwargs = {}) | unfixed
    %cat_15 : [#users=2] = call_function[target=torch.cat](args = ([%cat_11, %blocks_0_4_layers_3_conv_down_net_5, %blocks_0_4_layers_3_conv_normal_net_5],), kwargs = {dim: 1}) | unfixed
    %classifier_0_m_0_net_0 : [#users=1] = call_module[target=classifier.0.m.0.net.0](args = (%cat_15,), kwargs = {}) | unfixed
    %classifier_0_m_0_net_1 : [#users=1] = call_module[target=classifier.0.m.0.net.1](args = (%classifier_0_m_0_net_0,), kwargs = {}) | unfixed
    %classifier_0_m_0_net_2 : [#users=1] = call_module[target=classifier.0.m.0.net.2](args = (%classifier_0_m_0_net_1,), kwargs = {}) | unfixed
    %classifier_0_m_1_net_0 : [#users=1] = call_module[target=classifier.0.m.1.net.0](args = (%classifier_0_m_0_net_2,), kwargs = {}) | unfixed
    %classifier_0_m_1_net_1 : [#users=1] = call_module[target=classifier.0.m.1.net.1](args = (%classifier_0_m_1_net_0,), kwargs = {}) | unfixed
    %classifier_0_m_1_net_2 : [#users=1] = call_module[target=classifier.0.m.1.net.2](args = (%classifier_0_m_1_net_1,), kwargs = {}) | unfixed
    %classifier_0_m_2 : [#users=2] = call_module[target=classifier.0.m.2](args = (%classifier_0_m_1_net_2,), kwargs = {}) | unfixed
    %size : [#users=1] = call_method[target=size](args = (%classifier_0_m_2, 0), kwargs = {}) | unfixed
    %view : [#users=1] = call_method[target=view](args = (%classifier_0_m_2, %size, 384), kwargs = {}) | unfixed
    %classifier_0_linear : [#users=1] = call_module[target=classifier.0.linear](args = (%view,), kwargs = {}) | unfixed
    %scatters_0 : [#users=5] = call_module[target=scatters.0](args = ([%cat_12, %cat_13, %cat_14, %cat_15, %classifier_0_linear], %classifier_0_linear), kwargs = {}) | fixed
    %getitem : [#users=1] = call_function[target=operator.getitem](args = (%scatters_0, 0), kwargs = {}) | fixed
    %getitem_1 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_0, 1), kwargs = {}) | fixed
    %getitem_2 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_0, 2), kwargs = {}) | fixed
    %getitem_3 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_0, 3), kwargs = {}) | fixed
    %getitem_4 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_0, 4), kwargs = {}) | fixed
    %getitem_5 : [#users=3] = call_function[target=operator.getitem](args = (%getitem, 1), kwargs = {}) | fixed
    %getitem_6 : [#users=3] = call_function[target=operator.getitem](args = (%getitem_1, 1), kwargs = {}) | fixed
    %getitem_7 : [#users=3] = call_function[target=operator.getitem](args = (%getitem_2, 1), kwargs = {}) | fixed
    %getitem_8 : [#users=2] = call_function[target=operator.getitem](args = (%getitem_3, 1), kwargs = {}) | fixed
    %getitem_9 : [#users=1] = call_function[target=operator.getitem](args = (%getitem_4, 0), kwargs = {}) | fixed
    %blocks_1_0_layers_0_conv_normal_net_0 : [#users=1] = call_module[target=blocks.1.0.layers.0.conv_normal.net.0](args = (%getitem_5,), kwargs = {}) | fixed
    %blocks_1_0_layers_0_conv_normal_net_1 : [#users=1] = call_module[target=blocks.1.0.layers.0.conv_normal.net.1](args = (%blocks_1_0_layers_0_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_1_0_layers_0_conv_normal_net_2 : [#users=1] = call_module[target=blocks.1.0.layers.0.conv_normal.net.2](args = (%blocks_1_0_layers_0_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_1_0_layers_0_conv_normal_net_3 : [#users=1] = call_module[target=blocks.1.0.layers.0.conv_normal.net.3](args = (%blocks_1_0_layers_0_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_1_0_layers_0_conv_normal_net_4 : [#users=1] = call_module[target=blocks.1.0.layers.0.conv_normal.net.4](args = (%blocks_1_0_layers_0_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_1_0_layers_0_conv_normal_net_5 : [#users=1] = call_module[target=blocks.1.0.layers.0.conv_normal.net.5](args = (%blocks_1_0_layers_0_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_16 : [#users=1] = call_function[target=torch.cat](args = ([%getitem_5, %blocks_1_0_layers_0_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_1_0_layers_1_conv_down_net_0 : [#users=1] = call_module[target=blocks.1.0.layers.1.conv_down.net.0](args = (%getitem_5,), kwargs = {}) | fixed
    %blocks_1_0_layers_1_conv_down_net_1 : [#users=1] = call_module[target=blocks.1.0.layers.1.conv_down.net.1](args = (%blocks_1_0_layers_1_conv_down_net_0,), kwargs = {}) | fixed
    %blocks_1_0_layers_1_conv_down_net_2 : [#users=1] = call_module[target=blocks.1.0.layers.1.conv_down.net.2](args = (%blocks_1_0_layers_1_conv_down_net_1,), kwargs = {}) | fixed
    %blocks_1_0_layers_1_conv_down_net_3 : [#users=1] = call_module[target=blocks.1.0.layers.1.conv_down.net.3](args = (%blocks_1_0_layers_1_conv_down_net_2,), kwargs = {}) | fixed
    %blocks_1_0_layers_1_conv_down_net_4 : [#users=1] = call_module[target=blocks.1.0.layers.1.conv_down.net.4](args = (%blocks_1_0_layers_1_conv_down_net_3,), kwargs = {}) | fixed
    %blocks_1_0_layers_1_conv_down_net_5 : [#users=1] = call_module[target=blocks.1.0.layers.1.conv_down.net.5](args = (%blocks_1_0_layers_1_conv_down_net_4,), kwargs = {}) | fixed
    %blocks_1_0_layers_1_conv_normal_net_0 : [#users=1] = call_module[target=blocks.1.0.layers.1.conv_normal.net.0](args = (%getitem_6,), kwargs = {}) | fixed
    %blocks_1_0_layers_1_conv_normal_net_1 : [#users=1] = call_module[target=blocks.1.0.layers.1.conv_normal.net.1](args = (%blocks_1_0_layers_1_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_1_0_layers_1_conv_normal_net_2 : [#users=1] = call_module[target=blocks.1.0.layers.1.conv_normal.net.2](args = (%blocks_1_0_layers_1_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_1_0_layers_1_conv_normal_net_3 : [#users=1] = call_module[target=blocks.1.0.layers.1.conv_normal.net.3](args = (%blocks_1_0_layers_1_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_1_0_layers_1_conv_normal_net_4 : [#users=1] = call_module[target=blocks.1.0.layers.1.conv_normal.net.4](args = (%blocks_1_0_layers_1_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_1_0_layers_1_conv_normal_net_5 : [#users=1] = call_module[target=blocks.1.0.layers.1.conv_normal.net.5](args = (%blocks_1_0_layers_1_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_17 : [#users=3] = call_function[target=torch.cat](args = ([%getitem_6, %blocks_1_0_layers_1_conv_down_net_5, %blocks_1_0_layers_1_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_1_0_layers_2_conv_down_net_0 : [#users=1] = call_module[target=blocks.1.0.layers.2.conv_down.net.0](args = (%getitem_6,), kwargs = {}) | fixed
    %blocks_1_0_layers_2_conv_down_net_1 : [#users=1] = call_module[target=blocks.1.0.layers.2.conv_down.net.1](args = (%blocks_1_0_layers_2_conv_down_net_0,), kwargs = {}) | fixed
    %blocks_1_0_layers_2_conv_down_net_2 : [#users=1] = call_module[target=blocks.1.0.layers.2.conv_down.net.2](args = (%blocks_1_0_layers_2_conv_down_net_1,), kwargs = {}) | fixed
    %blocks_1_0_layers_2_conv_down_net_3 : [#users=1] = call_module[target=blocks.1.0.layers.2.conv_down.net.3](args = (%blocks_1_0_layers_2_conv_down_net_2,), kwargs = {}) | fixed
    %blocks_1_0_layers_2_conv_down_net_4 : [#users=1] = call_module[target=blocks.1.0.layers.2.conv_down.net.4](args = (%blocks_1_0_layers_2_conv_down_net_3,), kwargs = {}) | fixed
    %blocks_1_0_layers_2_conv_down_net_5 : [#users=1] = call_module[target=blocks.1.0.layers.2.conv_down.net.5](args = (%blocks_1_0_layers_2_conv_down_net_4,), kwargs = {}) | fixed
    %blocks_1_0_layers_2_conv_normal_net_0 : [#users=1] = call_module[target=blocks.1.0.layers.2.conv_normal.net.0](args = (%getitem_7,), kwargs = {}) | fixed
    %blocks_1_0_layers_2_conv_normal_net_1 : [#users=1] = call_module[target=blocks.1.0.layers.2.conv_normal.net.1](args = (%blocks_1_0_layers_2_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_1_0_layers_2_conv_normal_net_2 : [#users=1] = call_module[target=blocks.1.0.layers.2.conv_normal.net.2](args = (%blocks_1_0_layers_2_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_1_0_layers_2_conv_normal_net_3 : [#users=1] = call_module[target=blocks.1.0.layers.2.conv_normal.net.3](args = (%blocks_1_0_layers_2_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_1_0_layers_2_conv_normal_net_4 : [#users=1] = call_module[target=blocks.1.0.layers.2.conv_normal.net.4](args = (%blocks_1_0_layers_2_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_1_0_layers_2_conv_normal_net_5 : [#users=1] = call_module[target=blocks.1.0.layers.2.conv_normal.net.5](args = (%blocks_1_0_layers_2_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_18 : [#users=3] = call_function[target=torch.cat](args = ([%getitem_7, %blocks_1_0_layers_2_conv_down_net_5, %blocks_1_0_layers_2_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_1_0_layers_3_conv_down_net_0 : [#users=1] = call_module[target=blocks.1.0.layers.3.conv_down.net.0](args = (%getitem_7,), kwargs = {}) | fixed
    %blocks_1_0_layers_3_conv_down_net_1 : [#users=1] = call_module[target=blocks.1.0.layers.3.conv_down.net.1](args = (%blocks_1_0_layers_3_conv_down_net_0,), kwargs = {}) | fixed
    %blocks_1_0_layers_3_conv_down_net_2 : [#users=1] = call_module[target=blocks.1.0.layers.3.conv_down.net.2](args = (%blocks_1_0_layers_3_conv_down_net_1,), kwargs = {}) | fixed
    %blocks_1_0_layers_3_conv_down_net_3 : [#users=1] = call_module[target=blocks.1.0.layers.3.conv_down.net.3](args = (%blocks_1_0_layers_3_conv_down_net_2,), kwargs = {}) | fixed
    %blocks_1_0_layers_3_conv_down_net_4 : [#users=1] = call_module[target=blocks.1.0.layers.3.conv_down.net.4](args = (%blocks_1_0_layers_3_conv_down_net_3,), kwargs = {}) | fixed
    %blocks_1_0_layers_3_conv_down_net_5 : [#users=1] = call_module[target=blocks.1.0.layers.3.conv_down.net.5](args = (%blocks_1_0_layers_3_conv_down_net_4,), kwargs = {}) | fixed
    %blocks_1_0_layers_3_conv_normal_net_0 : [#users=1] = call_module[target=blocks.1.0.layers.3.conv_normal.net.0](args = (%getitem_8,), kwargs = {}) | fixed
    %blocks_1_0_layers_3_conv_normal_net_1 : [#users=1] = call_module[target=blocks.1.0.layers.3.conv_normal.net.1](args = (%blocks_1_0_layers_3_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_1_0_layers_3_conv_normal_net_2 : [#users=1] = call_module[target=blocks.1.0.layers.3.conv_normal.net.2](args = (%blocks_1_0_layers_3_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_1_0_layers_3_conv_normal_net_3 : [#users=1] = call_module[target=blocks.1.0.layers.3.conv_normal.net.3](args = (%blocks_1_0_layers_3_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_1_0_layers_3_conv_normal_net_4 : [#users=1] = call_module[target=blocks.1.0.layers.3.conv_normal.net.4](args = (%blocks_1_0_layers_3_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_1_0_layers_3_conv_normal_net_5 : [#users=1] = call_module[target=blocks.1.0.layers.3.conv_normal.net.5](args = (%blocks_1_0_layers_3_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_19 : [#users=2] = call_function[target=torch.cat](args = ([%getitem_8, %blocks_1_0_layers_3_conv_down_net_5, %blocks_1_0_layers_3_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_1_1_layers_0_conv_down_net_0 : [#users=1] = call_module[target=blocks.1.1.layers.0.conv_down.net.0](args = (%cat_16,), kwargs = {}) | fixed
    %blocks_1_1_layers_0_conv_down_net_1 : [#users=1] = call_module[target=blocks.1.1.layers.0.conv_down.net.1](args = (%blocks_1_1_layers_0_conv_down_net_0,), kwargs = {}) | fixed
    %blocks_1_1_layers_0_conv_down_net_2 : [#users=1] = call_module[target=blocks.1.1.layers.0.conv_down.net.2](args = (%blocks_1_1_layers_0_conv_down_net_1,), kwargs = {}) | fixed
    %blocks_1_1_layers_0_conv_down_net_3 : [#users=1] = call_module[target=blocks.1.1.layers.0.conv_down.net.3](args = (%blocks_1_1_layers_0_conv_down_net_2,), kwargs = {}) | fixed
    %blocks_1_1_layers_0_conv_down_net_4 : [#users=1] = call_module[target=blocks.1.1.layers.0.conv_down.net.4](args = (%blocks_1_1_layers_0_conv_down_net_3,), kwargs = {}) | fixed
    %blocks_1_1_layers_0_conv_down_net_5 : [#users=1] = call_module[target=blocks.1.1.layers.0.conv_down.net.5](args = (%blocks_1_1_layers_0_conv_down_net_4,), kwargs = {}) | fixed
    %blocks_1_1_layers_0_conv_normal_net_0 : [#users=1] = call_module[target=blocks.1.1.layers.0.conv_normal.net.0](args = (%cat_17,), kwargs = {}) | fixed
    %blocks_1_1_layers_0_conv_normal_net_1 : [#users=1] = call_module[target=blocks.1.1.layers.0.conv_normal.net.1](args = (%blocks_1_1_layers_0_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_1_1_layers_0_conv_normal_net_2 : [#users=1] = call_module[target=blocks.1.1.layers.0.conv_normal.net.2](args = (%blocks_1_1_layers_0_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_1_1_layers_0_conv_normal_net_3 : [#users=1] = call_module[target=blocks.1.1.layers.0.conv_normal.net.3](args = (%blocks_1_1_layers_0_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_1_1_layers_0_conv_normal_net_4 : [#users=1] = call_module[target=blocks.1.1.layers.0.conv_normal.net.4](args = (%blocks_1_1_layers_0_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_1_1_layers_0_conv_normal_net_5 : [#users=1] = call_module[target=blocks.1.1.layers.0.conv_normal.net.5](args = (%blocks_1_1_layers_0_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_20 : [#users=1] = call_function[target=torch.cat](args = ([%cat_17, %blocks_1_1_layers_0_conv_down_net_5, %blocks_1_1_layers_0_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_1_1_layers_1_conv_down_net_0 : [#users=1] = call_module[target=blocks.1.1.layers.1.conv_down.net.0](args = (%cat_17,), kwargs = {}) | fixed
    %blocks_1_1_layers_1_conv_down_net_1 : [#users=1] = call_module[target=blocks.1.1.layers.1.conv_down.net.1](args = (%blocks_1_1_layers_1_conv_down_net_0,), kwargs = {}) | fixed
    %blocks_1_1_layers_1_conv_down_net_2 : [#users=1] = call_module[target=blocks.1.1.layers.1.conv_down.net.2](args = (%blocks_1_1_layers_1_conv_down_net_1,), kwargs = {}) | fixed
    %blocks_1_1_layers_1_conv_down_net_3 : [#users=1] = call_module[target=blocks.1.1.layers.1.conv_down.net.3](args = (%blocks_1_1_layers_1_conv_down_net_2,), kwargs = {}) | fixed
    %blocks_1_1_layers_1_conv_down_net_4 : [#users=1] = call_module[target=blocks.1.1.layers.1.conv_down.net.4](args = (%blocks_1_1_layers_1_conv_down_net_3,), kwargs = {}) | fixed
    %blocks_1_1_layers_1_conv_down_net_5 : [#users=1] = call_module[target=blocks.1.1.layers.1.conv_down.net.5](args = (%blocks_1_1_layers_1_conv_down_net_4,), kwargs = {}) | fixed
    %blocks_1_1_layers_1_conv_normal_net_0 : [#users=1] = call_module[target=blocks.1.1.layers.1.conv_normal.net.0](args = (%cat_18,), kwargs = {}) | fixed
    %blocks_1_1_layers_1_conv_normal_net_1 : [#users=1] = call_module[target=blocks.1.1.layers.1.conv_normal.net.1](args = (%blocks_1_1_layers_1_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_1_1_layers_1_conv_normal_net_2 : [#users=1] = call_module[target=blocks.1.1.layers.1.conv_normal.net.2](args = (%blocks_1_1_layers_1_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_1_1_layers_1_conv_normal_net_3 : [#users=1] = call_module[target=blocks.1.1.layers.1.conv_normal.net.3](args = (%blocks_1_1_layers_1_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_1_1_layers_1_conv_normal_net_4 : [#users=1] = call_module[target=blocks.1.1.layers.1.conv_normal.net.4](args = (%blocks_1_1_layers_1_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_1_1_layers_1_conv_normal_net_5 : [#users=1] = call_module[target=blocks.1.1.layers.1.conv_normal.net.5](args = (%blocks_1_1_layers_1_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_21 : [#users=1] = call_function[target=torch.cat](args = ([%cat_18, %blocks_1_1_layers_1_conv_down_net_5, %blocks_1_1_layers_1_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_1_1_layers_2_conv_down_net_0 : [#users=1] = call_module[target=blocks.1.1.layers.2.conv_down.net.0](args = (%cat_18,), kwargs = {}) | fixed
    %blocks_1_1_layers_2_conv_down_net_1 : [#users=1] = call_module[target=blocks.1.1.layers.2.conv_down.net.1](args = (%blocks_1_1_layers_2_conv_down_net_0,), kwargs = {}) | fixed
    %blocks_1_1_layers_2_conv_down_net_2 : [#users=1] = call_module[target=blocks.1.1.layers.2.conv_down.net.2](args = (%blocks_1_1_layers_2_conv_down_net_1,), kwargs = {}) | fixed
    %blocks_1_1_layers_2_conv_down_net_3 : [#users=1] = call_module[target=blocks.1.1.layers.2.conv_down.net.3](args = (%blocks_1_1_layers_2_conv_down_net_2,), kwargs = {}) | fixed
    %blocks_1_1_layers_2_conv_down_net_4 : [#users=1] = call_module[target=blocks.1.1.layers.2.conv_down.net.4](args = (%blocks_1_1_layers_2_conv_down_net_3,), kwargs = {}) | fixed
    %blocks_1_1_layers_2_conv_down_net_5 : [#users=1] = call_module[target=blocks.1.1.layers.2.conv_down.net.5](args = (%blocks_1_1_layers_2_conv_down_net_4,), kwargs = {}) | fixed
    %blocks_1_1_layers_2_conv_normal_net_0 : [#users=1] = call_module[target=blocks.1.1.layers.2.conv_normal.net.0](args = (%cat_19,), kwargs = {}) | fixed
    %blocks_1_1_layers_2_conv_normal_net_1 : [#users=1] = call_module[target=blocks.1.1.layers.2.conv_normal.net.1](args = (%blocks_1_1_layers_2_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_1_1_layers_2_conv_normal_net_2 : [#users=1] = call_module[target=blocks.1.1.layers.2.conv_normal.net.2](args = (%blocks_1_1_layers_2_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_1_1_layers_2_conv_normal_net_3 : [#users=1] = call_module[target=blocks.1.1.layers.2.conv_normal.net.3](args = (%blocks_1_1_layers_2_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_1_1_layers_2_conv_normal_net_4 : [#users=1] = call_module[target=blocks.1.1.layers.2.conv_normal.net.4](args = (%blocks_1_1_layers_2_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_1_1_layers_2_conv_normal_net_5 : [#users=1] = call_module[target=blocks.1.1.layers.2.conv_normal.net.5](args = (%blocks_1_1_layers_2_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_22 : [#users=1] = call_function[target=torch.cat](args = ([%cat_19, %blocks_1_1_layers_2_conv_down_net_5, %blocks_1_1_layers_2_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_1_2_m_0_net_0 : [#users=1] = call_module[target=blocks.1.2.m.0.net.0](args = (%cat_20,), kwargs = {}) | fixed
    %blocks_1_2_m_0_net_1 : [#users=1] = call_module[target=blocks.1.2.m.0.net.1](args = (%blocks_1_2_m_0_net_0,), kwargs = {}) | fixed
    %blocks_1_2_m_0_net_2 : [#users=3] = call_module[target=blocks.1.2.m.0.net.2](args = (%blocks_1_2_m_0_net_1,), kwargs = {}) | fixed
    %blocks_1_2_m_1_net_0 : [#users=1] = call_module[target=blocks.1.2.m.1.net.0](args = (%cat_21,), kwargs = {}) | fixed
    %blocks_1_2_m_1_net_1 : [#users=1] = call_module[target=blocks.1.2.m.1.net.1](args = (%blocks_1_2_m_1_net_0,), kwargs = {}) | fixed
    %blocks_1_2_m_1_net_2 : [#users=3] = call_module[target=blocks.1.2.m.1.net.2](args = (%blocks_1_2_m_1_net_1,), kwargs = {}) | fixed
    %blocks_1_2_m_2_net_0 : [#users=1] = call_module[target=blocks.1.2.m.2.net.0](args = (%cat_22,), kwargs = {}) | fixed
    %blocks_1_2_m_2_net_1 : [#users=1] = call_module[target=blocks.1.2.m.2.net.1](args = (%blocks_1_2_m_2_net_0,), kwargs = {}) | fixed
    %blocks_1_2_m_2_net_2 : [#users=2] = call_module[target=blocks.1.2.m.2.net.2](args = (%blocks_1_2_m_2_net_1,), kwargs = {}) | fixed
    %blocks_1_3_layers_0_conv_normal_net_0 : [#users=1] = call_module[target=blocks.1.3.layers.0.conv_normal.net.0](args = (%blocks_1_2_m_0_net_2,), kwargs = {}) | fixed
    %blocks_1_3_layers_0_conv_normal_net_1 : [#users=1] = call_module[target=blocks.1.3.layers.0.conv_normal.net.1](args = (%blocks_1_3_layers_0_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_1_3_layers_0_conv_normal_net_2 : [#users=1] = call_module[target=blocks.1.3.layers.0.conv_normal.net.2](args = (%blocks_1_3_layers_0_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_1_3_layers_0_conv_normal_net_3 : [#users=1] = call_module[target=blocks.1.3.layers.0.conv_normal.net.3](args = (%blocks_1_3_layers_0_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_1_3_layers_0_conv_normal_net_4 : [#users=1] = call_module[target=blocks.1.3.layers.0.conv_normal.net.4](args = (%blocks_1_3_layers_0_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_1_3_layers_0_conv_normal_net_5 : [#users=1] = call_module[target=blocks.1.3.layers.0.conv_normal.net.5](args = (%blocks_1_3_layers_0_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_23 : [#users=3] = call_function[target=torch.cat](args = ([%blocks_1_2_m_0_net_2, %blocks_1_3_layers_0_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_1_3_layers_1_conv_down_net_0 : [#users=1] = call_module[target=blocks.1.3.layers.1.conv_down.net.0](args = (%blocks_1_2_m_0_net_2,), kwargs = {}) | fixed
    %blocks_1_3_layers_1_conv_down_net_1 : [#users=1] = call_module[target=blocks.1.3.layers.1.conv_down.net.1](args = (%blocks_1_3_layers_1_conv_down_net_0,), kwargs = {}) | fixed
    %blocks_1_3_layers_1_conv_down_net_2 : [#users=1] = call_module[target=blocks.1.3.layers.1.conv_down.net.2](args = (%blocks_1_3_layers_1_conv_down_net_1,), kwargs = {}) | fixed
    %blocks_1_3_layers_1_conv_down_net_3 : [#users=1] = call_module[target=blocks.1.3.layers.1.conv_down.net.3](args = (%blocks_1_3_layers_1_conv_down_net_2,), kwargs = {}) | fixed
    %blocks_1_3_layers_1_conv_down_net_4 : [#users=1] = call_module[target=blocks.1.3.layers.1.conv_down.net.4](args = (%blocks_1_3_layers_1_conv_down_net_3,), kwargs = {}) | fixed
    %blocks_1_3_layers_1_conv_down_net_5 : [#users=1] = call_module[target=blocks.1.3.layers.1.conv_down.net.5](args = (%blocks_1_3_layers_1_conv_down_net_4,), kwargs = {}) | fixed
    %blocks_1_3_layers_1_conv_normal_net_0 : [#users=1] = call_module[target=blocks.1.3.layers.1.conv_normal.net.0](args = (%blocks_1_2_m_1_net_2,), kwargs = {}) | fixed
    %blocks_1_3_layers_1_conv_normal_net_1 : [#users=1] = call_module[target=blocks.1.3.layers.1.conv_normal.net.1](args = (%blocks_1_3_layers_1_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_1_3_layers_1_conv_normal_net_2 : [#users=1] = call_module[target=blocks.1.3.layers.1.conv_normal.net.2](args = (%blocks_1_3_layers_1_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_1_3_layers_1_conv_normal_net_3 : [#users=1] = call_module[target=blocks.1.3.layers.1.conv_normal.net.3](args = (%blocks_1_3_layers_1_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_1_3_layers_1_conv_normal_net_4 : [#users=1] = call_module[target=blocks.1.3.layers.1.conv_normal.net.4](args = (%blocks_1_3_layers_1_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_1_3_layers_1_conv_normal_net_5 : [#users=1] = call_module[target=blocks.1.3.layers.1.conv_normal.net.5](args = (%blocks_1_3_layers_1_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_24 : [#users=3] = call_function[target=torch.cat](args = ([%blocks_1_2_m_1_net_2, %blocks_1_3_layers_1_conv_down_net_5, %blocks_1_3_layers_1_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_1_3_layers_2_conv_down_net_0 : [#users=1] = call_module[target=blocks.1.3.layers.2.conv_down.net.0](args = (%blocks_1_2_m_1_net_2,), kwargs = {}) | fixed
    %blocks_1_3_layers_2_conv_down_net_1 : [#users=1] = call_module[target=blocks.1.3.layers.2.conv_down.net.1](args = (%blocks_1_3_layers_2_conv_down_net_0,), kwargs = {}) | fixed
    %blocks_1_3_layers_2_conv_down_net_2 : [#users=1] = call_module[target=blocks.1.3.layers.2.conv_down.net.2](args = (%blocks_1_3_layers_2_conv_down_net_1,), kwargs = {}) | fixed
    %blocks_1_3_layers_2_conv_down_net_3 : [#users=1] = call_module[target=blocks.1.3.layers.2.conv_down.net.3](args = (%blocks_1_3_layers_2_conv_down_net_2,), kwargs = {}) | fixed
    %blocks_1_3_layers_2_conv_down_net_4 : [#users=1] = call_module[target=blocks.1.3.layers.2.conv_down.net.4](args = (%blocks_1_3_layers_2_conv_down_net_3,), kwargs = {}) | fixed
    %blocks_1_3_layers_2_conv_down_net_5 : [#users=1] = call_module[target=blocks.1.3.layers.2.conv_down.net.5](args = (%blocks_1_3_layers_2_conv_down_net_4,), kwargs = {}) | fixed
    %blocks_1_3_layers_2_conv_normal_net_0 : [#users=1] = call_module[target=blocks.1.3.layers.2.conv_normal.net.0](args = (%blocks_1_2_m_2_net_2,), kwargs = {}) | fixed
    %blocks_1_3_layers_2_conv_normal_net_1 : [#users=1] = call_module[target=blocks.1.3.layers.2.conv_normal.net.1](args = (%blocks_1_3_layers_2_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_1_3_layers_2_conv_normal_net_2 : [#users=1] = call_module[target=blocks.1.3.layers.2.conv_normal.net.2](args = (%blocks_1_3_layers_2_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_1_3_layers_2_conv_normal_net_3 : [#users=1] = call_module[target=blocks.1.3.layers.2.conv_normal.net.3](args = (%blocks_1_3_layers_2_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_1_3_layers_2_conv_normal_net_4 : [#users=1] = call_module[target=blocks.1.3.layers.2.conv_normal.net.4](args = (%blocks_1_3_layers_2_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_1_3_layers_2_conv_normal_net_5 : [#users=1] = call_module[target=blocks.1.3.layers.2.conv_normal.net.5](args = (%blocks_1_3_layers_2_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_25 : [#users=2] = call_function[target=torch.cat](args = ([%blocks_1_2_m_2_net_2, %blocks_1_3_layers_2_conv_down_net_5, %blocks_1_3_layers_2_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_1_4_layers_0_conv_normal_net_0 : [#users=1] = call_module[target=blocks.1.4.layers.0.conv_normal.net.0](args = (%cat_23,), kwargs = {}) | fixed
    %blocks_1_4_layers_0_conv_normal_net_1 : [#users=1] = call_module[target=blocks.1.4.layers.0.conv_normal.net.1](args = (%blocks_1_4_layers_0_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_1_4_layers_0_conv_normal_net_2 : [#users=1] = call_module[target=blocks.1.4.layers.0.conv_normal.net.2](args = (%blocks_1_4_layers_0_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_1_4_layers_0_conv_normal_net_3 : [#users=1] = call_module[target=blocks.1.4.layers.0.conv_normal.net.3](args = (%blocks_1_4_layers_0_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_1_4_layers_0_conv_normal_net_4 : [#users=1] = call_module[target=blocks.1.4.layers.0.conv_normal.net.4](args = (%blocks_1_4_layers_0_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_1_4_layers_0_conv_normal_net_5 : [#users=1] = call_module[target=blocks.1.4.layers.0.conv_normal.net.5](args = (%blocks_1_4_layers_0_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_26 : [#users=1] = call_function[target=torch.cat](args = ([%cat_23, %blocks_1_4_layers_0_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_1_4_layers_1_conv_down_net_0 : [#users=1] = call_module[target=blocks.1.4.layers.1.conv_down.net.0](args = (%cat_23,), kwargs = {}) | fixed
    %blocks_1_4_layers_1_conv_down_net_1 : [#users=1] = call_module[target=blocks.1.4.layers.1.conv_down.net.1](args = (%blocks_1_4_layers_1_conv_down_net_0,), kwargs = {}) | fixed
    %blocks_1_4_layers_1_conv_down_net_2 : [#users=1] = call_module[target=blocks.1.4.layers.1.conv_down.net.2](args = (%blocks_1_4_layers_1_conv_down_net_1,), kwargs = {}) | fixed
    %blocks_1_4_layers_1_conv_down_net_3 : [#users=1] = call_module[target=blocks.1.4.layers.1.conv_down.net.3](args = (%blocks_1_4_layers_1_conv_down_net_2,), kwargs = {}) | fixed
    %blocks_1_4_layers_1_conv_down_net_4 : [#users=1] = call_module[target=blocks.1.4.layers.1.conv_down.net.4](args = (%blocks_1_4_layers_1_conv_down_net_3,), kwargs = {}) | fixed
    %blocks_1_4_layers_1_conv_down_net_5 : [#users=1] = call_module[target=blocks.1.4.layers.1.conv_down.net.5](args = (%blocks_1_4_layers_1_conv_down_net_4,), kwargs = {}) | fixed
    %blocks_1_4_layers_1_conv_normal_net_0 : [#users=1] = call_module[target=blocks.1.4.layers.1.conv_normal.net.0](args = (%cat_24,), kwargs = {}) | fixed
    %blocks_1_4_layers_1_conv_normal_net_1 : [#users=1] = call_module[target=blocks.1.4.layers.1.conv_normal.net.1](args = (%blocks_1_4_layers_1_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_1_4_layers_1_conv_normal_net_2 : [#users=1] = call_module[target=blocks.1.4.layers.1.conv_normal.net.2](args = (%blocks_1_4_layers_1_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_1_4_layers_1_conv_normal_net_3 : [#users=1] = call_module[target=blocks.1.4.layers.1.conv_normal.net.3](args = (%blocks_1_4_layers_1_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_1_4_layers_1_conv_normal_net_4 : [#users=1] = call_module[target=blocks.1.4.layers.1.conv_normal.net.4](args = (%blocks_1_4_layers_1_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_1_4_layers_1_conv_normal_net_5 : [#users=1] = call_module[target=blocks.1.4.layers.1.conv_normal.net.5](args = (%blocks_1_4_layers_1_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_27 : [#users=1] = call_function[target=torch.cat](args = ([%cat_24, %blocks_1_4_layers_1_conv_down_net_5, %blocks_1_4_layers_1_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_1_4_layers_2_conv_down_net_0 : [#users=1] = call_module[target=blocks.1.4.layers.2.conv_down.net.0](args = (%cat_24,), kwargs = {}) | fixed
    %blocks_1_4_layers_2_conv_down_net_1 : [#users=1] = call_module[target=blocks.1.4.layers.2.conv_down.net.1](args = (%blocks_1_4_layers_2_conv_down_net_0,), kwargs = {}) | fixed
    %blocks_1_4_layers_2_conv_down_net_2 : [#users=1] = call_module[target=blocks.1.4.layers.2.conv_down.net.2](args = (%blocks_1_4_layers_2_conv_down_net_1,), kwargs = {}) | fixed
    %blocks_1_4_layers_2_conv_down_net_3 : [#users=1] = call_module[target=blocks.1.4.layers.2.conv_down.net.3](args = (%blocks_1_4_layers_2_conv_down_net_2,), kwargs = {}) | fixed
    %blocks_1_4_layers_2_conv_down_net_4 : [#users=1] = call_module[target=blocks.1.4.layers.2.conv_down.net.4](args = (%blocks_1_4_layers_2_conv_down_net_3,), kwargs = {}) | fixed
    %blocks_1_4_layers_2_conv_down_net_5 : [#users=1] = call_module[target=blocks.1.4.layers.2.conv_down.net.5](args = (%blocks_1_4_layers_2_conv_down_net_4,), kwargs = {}) | fixed
    %blocks_1_4_layers_2_conv_normal_net_0 : [#users=1] = call_module[target=blocks.1.4.layers.2.conv_normal.net.0](args = (%cat_25,), kwargs = {}) | fixed
    %blocks_1_4_layers_2_conv_normal_net_1 : [#users=1] = call_module[target=blocks.1.4.layers.2.conv_normal.net.1](args = (%blocks_1_4_layers_2_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_1_4_layers_2_conv_normal_net_2 : [#users=1] = call_module[target=blocks.1.4.layers.2.conv_normal.net.2](args = (%blocks_1_4_layers_2_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_1_4_layers_2_conv_normal_net_3 : [#users=1] = call_module[target=blocks.1.4.layers.2.conv_normal.net.3](args = (%blocks_1_4_layers_2_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_1_4_layers_2_conv_normal_net_4 : [#users=1] = call_module[target=blocks.1.4.layers.2.conv_normal.net.4](args = (%blocks_1_4_layers_2_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_1_4_layers_2_conv_normal_net_5 : [#users=1] = call_module[target=blocks.1.4.layers.2.conv_normal.net.5](args = (%blocks_1_4_layers_2_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_28 : [#users=3] = call_function[target=torch.cat](args = ([%cat_25, %blocks_1_4_layers_2_conv_down_net_5, %blocks_1_4_layers_2_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %classifier_1_m_0_net_0 : [#users=1] = call_module[target=classifier.1.m.0.net.0](args = (%cat_28,), kwargs = {}) | fixed
    %classifier_1_m_0_net_1 : [#users=1] = call_module[target=classifier.1.m.0.net.1](args = (%classifier_1_m_0_net_0,), kwargs = {}) | fixed
    %classifier_1_m_0_net_2 : [#users=1] = call_module[target=classifier.1.m.0.net.2](args = (%classifier_1_m_0_net_1,), kwargs = {}) | fixed
    %classifier_1_m_1_net_0 : [#users=1] = call_module[target=classifier.1.m.1.net.0](args = (%classifier_1_m_0_net_2,), kwargs = {}) | fixed
    %classifier_1_m_1_net_1 : [#users=1] = call_module[target=classifier.1.m.1.net.1](args = (%classifier_1_m_1_net_0,), kwargs = {}) | fixed
    %classifier_1_m_1_net_2 : [#users=1] = call_module[target=classifier.1.m.1.net.2](args = (%classifier_1_m_1_net_1,), kwargs = {}) | fixed
    %classifier_1_m_2 : [#users=2] = call_module[target=classifier.1.m.2](args = (%classifier_1_m_1_net_2,), kwargs = {}) | fixed
    %size_1 : [#users=1] = call_method[target=size](args = (%classifier_1_m_2, 0), kwargs = {}) | fixed
    %view_1 : [#users=1] = call_method[target=view](args = (%classifier_1_m_2, %size_1, 384), kwargs = {}) | unfixed
    %classifier_1_linear : [#users=1] = call_module[target=classifier.1.linear](args = (%view_1,), kwargs = {}) | unfixed
    %zeros_like : [#users=1] = call_function[target=torch.zeros_like](args = (%cat_28,), kwargs = {}) | fixed
    %scatters_1 : [#users=5] = call_module[target=scatters.1](args = ([%cat_26, %cat_27, %cat_28, %zeros_like, %classifier_1_linear], %classifier_1_linear), kwargs = {}) | fixed
    %getitem_10 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_1, 0), kwargs = {}) | fixed
    %getitem_11 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_1, 1), kwargs = {}) | fixed
    %getitem_12 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_1, 2), kwargs = {}) | fixed
    %getitem_13 : [#users=0] = call_function[target=operator.getitem](args = (%scatters_1, 3), kwargs = {}) | fixed
    %getitem_14 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_1, 4), kwargs = {}) | fixed
    %getitem_15 : [#users=3] = call_function[target=operator.getitem](args = (%getitem_10, 1), kwargs = {}) | fixed
    %getitem_16 : [#users=3] = call_function[target=operator.getitem](args = (%getitem_11, 1), kwargs = {}) | fixed
    %getitem_17 : [#users=2] = call_function[target=operator.getitem](args = (%getitem_12, 1), kwargs = {}) | fixed
    %getitem_18 : [#users=1] = call_function[target=operator.getitem](args = (%getitem_14, 0), kwargs = {}) | fixed
    %blocks_2_0_layers_0_conv_normal_net_0 : [#users=1] = call_module[target=blocks.2.0.layers.0.conv_normal.net.0](args = (%getitem_15,), kwargs = {}) | fixed
    %blocks_2_0_layers_0_conv_normal_net_1 : [#users=1] = call_module[target=blocks.2.0.layers.0.conv_normal.net.1](args = (%blocks_2_0_layers_0_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_2_0_layers_0_conv_normal_net_2 : [#users=1] = call_module[target=blocks.2.0.layers.0.conv_normal.net.2](args = (%blocks_2_0_layers_0_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_2_0_layers_0_conv_normal_net_3 : [#users=1] = call_module[target=blocks.2.0.layers.0.conv_normal.net.3](args = (%blocks_2_0_layers_0_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_2_0_layers_0_conv_normal_net_4 : [#users=1] = call_module[target=blocks.2.0.layers.0.conv_normal.net.4](args = (%blocks_2_0_layers_0_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_2_0_layers_0_conv_normal_net_5 : [#users=1] = call_module[target=blocks.2.0.layers.0.conv_normal.net.5](args = (%blocks_2_0_layers_0_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_29 : [#users=3] = call_function[target=torch.cat](args = ([%getitem_15, %blocks_2_0_layers_0_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_2_0_layers_1_conv_down_net_0 : [#users=1] = call_module[target=blocks.2.0.layers.1.conv_down.net.0](args = (%getitem_15,), kwargs = {}) | fixed
    %blocks_2_0_layers_1_conv_down_net_1 : [#users=1] = call_module[target=blocks.2.0.layers.1.conv_down.net.1](args = (%blocks_2_0_layers_1_conv_down_net_0,), kwargs = {}) | fixed
    %blocks_2_0_layers_1_conv_down_net_2 : [#users=1] = call_module[target=blocks.2.0.layers.1.conv_down.net.2](args = (%blocks_2_0_layers_1_conv_down_net_1,), kwargs = {}) | fixed
    %blocks_2_0_layers_1_conv_down_net_3 : [#users=1] = call_module[target=blocks.2.0.layers.1.conv_down.net.3](args = (%blocks_2_0_layers_1_conv_down_net_2,), kwargs = {}) | fixed
    %blocks_2_0_layers_1_conv_down_net_4 : [#users=1] = call_module[target=blocks.2.0.layers.1.conv_down.net.4](args = (%blocks_2_0_layers_1_conv_down_net_3,), kwargs = {}) | fixed
    %blocks_2_0_layers_1_conv_down_net_5 : [#users=1] = call_module[target=blocks.2.0.layers.1.conv_down.net.5](args = (%blocks_2_0_layers_1_conv_down_net_4,), kwargs = {}) | fixed
    %blocks_2_0_layers_1_conv_normal_net_0 : [#users=1] = call_module[target=blocks.2.0.layers.1.conv_normal.net.0](args = (%getitem_16,), kwargs = {}) | fixed
    %blocks_2_0_layers_1_conv_normal_net_1 : [#users=1] = call_module[target=blocks.2.0.layers.1.conv_normal.net.1](args = (%blocks_2_0_layers_1_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_2_0_layers_1_conv_normal_net_2 : [#users=1] = call_module[target=blocks.2.0.layers.1.conv_normal.net.2](args = (%blocks_2_0_layers_1_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_2_0_layers_1_conv_normal_net_3 : [#users=1] = call_module[target=blocks.2.0.layers.1.conv_normal.net.3](args = (%blocks_2_0_layers_1_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_2_0_layers_1_conv_normal_net_4 : [#users=1] = call_module[target=blocks.2.0.layers.1.conv_normal.net.4](args = (%blocks_2_0_layers_1_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_2_0_layers_1_conv_normal_net_5 : [#users=1] = call_module[target=blocks.2.0.layers.1.conv_normal.net.5](args = (%blocks_2_0_layers_1_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_30 : [#users=3] = call_function[target=torch.cat](args = ([%getitem_16, %blocks_2_0_layers_1_conv_down_net_5, %blocks_2_0_layers_1_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_2_0_layers_2_conv_down_net_0 : [#users=1] = call_module[target=blocks.2.0.layers.2.conv_down.net.0](args = (%getitem_16,), kwargs = {}) | fixed
    %blocks_2_0_layers_2_conv_down_net_1 : [#users=1] = call_module[target=blocks.2.0.layers.2.conv_down.net.1](args = (%blocks_2_0_layers_2_conv_down_net_0,), kwargs = {}) | fixed
    %blocks_2_0_layers_2_conv_down_net_2 : [#users=1] = call_module[target=blocks.2.0.layers.2.conv_down.net.2](args = (%blocks_2_0_layers_2_conv_down_net_1,), kwargs = {}) | fixed
    %blocks_2_0_layers_2_conv_down_net_3 : [#users=1] = call_module[target=blocks.2.0.layers.2.conv_down.net.3](args = (%blocks_2_0_layers_2_conv_down_net_2,), kwargs = {}) | fixed
    %blocks_2_0_layers_2_conv_down_net_4 : [#users=1] = call_module[target=blocks.2.0.layers.2.conv_down.net.4](args = (%blocks_2_0_layers_2_conv_down_net_3,), kwargs = {}) | fixed
    %blocks_2_0_layers_2_conv_down_net_5 : [#users=1] = call_module[target=blocks.2.0.layers.2.conv_down.net.5](args = (%blocks_2_0_layers_2_conv_down_net_4,), kwargs = {}) | fixed
    %blocks_2_0_layers_2_conv_normal_net_0 : [#users=1] = call_module[target=blocks.2.0.layers.2.conv_normal.net.0](args = (%getitem_17,), kwargs = {}) | fixed
    %blocks_2_0_layers_2_conv_normal_net_1 : [#users=1] = call_module[target=blocks.2.0.layers.2.conv_normal.net.1](args = (%blocks_2_0_layers_2_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_2_0_layers_2_conv_normal_net_2 : [#users=1] = call_module[target=blocks.2.0.layers.2.conv_normal.net.2](args = (%blocks_2_0_layers_2_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_2_0_layers_2_conv_normal_net_3 : [#users=1] = call_module[target=blocks.2.0.layers.2.conv_normal.net.3](args = (%blocks_2_0_layers_2_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_2_0_layers_2_conv_normal_net_4 : [#users=1] = call_module[target=blocks.2.0.layers.2.conv_normal.net.4](args = (%blocks_2_0_layers_2_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_2_0_layers_2_conv_normal_net_5 : [#users=1] = call_module[target=blocks.2.0.layers.2.conv_normal.net.5](args = (%blocks_2_0_layers_2_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_31 : [#users=2] = call_function[target=torch.cat](args = ([%getitem_17, %blocks_2_0_layers_2_conv_down_net_5, %blocks_2_0_layers_2_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_2_1_layers_0_conv_normal_net_0 : [#users=1] = call_module[target=blocks.2.1.layers.0.conv_normal.net.0](args = (%cat_29,), kwargs = {}) | fixed
    %blocks_2_1_layers_0_conv_normal_net_1 : [#users=1] = call_module[target=blocks.2.1.layers.0.conv_normal.net.1](args = (%blocks_2_1_layers_0_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_2_1_layers_0_conv_normal_net_2 : [#users=1] = call_module[target=blocks.2.1.layers.0.conv_normal.net.2](args = (%blocks_2_1_layers_0_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_2_1_layers_0_conv_normal_net_3 : [#users=1] = call_module[target=blocks.2.1.layers.0.conv_normal.net.3](args = (%blocks_2_1_layers_0_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_2_1_layers_0_conv_normal_net_4 : [#users=1] = call_module[target=blocks.2.1.layers.0.conv_normal.net.4](args = (%blocks_2_1_layers_0_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_2_1_layers_0_conv_normal_net_5 : [#users=1] = call_module[target=blocks.2.1.layers.0.conv_normal.net.5](args = (%blocks_2_1_layers_0_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_32 : [#users=1] = call_function[target=torch.cat](args = ([%cat_29, %blocks_2_1_layers_0_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_2_1_layers_1_conv_down_net_0 : [#users=1] = call_module[target=blocks.2.1.layers.1.conv_down.net.0](args = (%cat_29,), kwargs = {}) | fixed
    %blocks_2_1_layers_1_conv_down_net_1 : [#users=1] = call_module[target=blocks.2.1.layers.1.conv_down.net.1](args = (%blocks_2_1_layers_1_conv_down_net_0,), kwargs = {}) | fixed
    %blocks_2_1_layers_1_conv_down_net_2 : [#users=1] = call_module[target=blocks.2.1.layers.1.conv_down.net.2](args = (%blocks_2_1_layers_1_conv_down_net_1,), kwargs = {}) | fixed
    %blocks_2_1_layers_1_conv_down_net_3 : [#users=1] = call_module[target=blocks.2.1.layers.1.conv_down.net.3](args = (%blocks_2_1_layers_1_conv_down_net_2,), kwargs = {}) | fixed
    %blocks_2_1_layers_1_conv_down_net_4 : [#users=1] = call_module[target=blocks.2.1.layers.1.conv_down.net.4](args = (%blocks_2_1_layers_1_conv_down_net_3,), kwargs = {}) | fixed
    %blocks_2_1_layers_1_conv_down_net_5 : [#users=1] = call_module[target=blocks.2.1.layers.1.conv_down.net.5](args = (%blocks_2_1_layers_1_conv_down_net_4,), kwargs = {}) | fixed
    %blocks_2_1_layers_1_conv_normal_net_0 : [#users=1] = call_module[target=blocks.2.1.layers.1.conv_normal.net.0](args = (%cat_30,), kwargs = {}) | fixed
    %blocks_2_1_layers_1_conv_normal_net_1 : [#users=1] = call_module[target=blocks.2.1.layers.1.conv_normal.net.1](args = (%blocks_2_1_layers_1_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_2_1_layers_1_conv_normal_net_2 : [#users=1] = call_module[target=blocks.2.1.layers.1.conv_normal.net.2](args = (%blocks_2_1_layers_1_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_2_1_layers_1_conv_normal_net_3 : [#users=1] = call_module[target=blocks.2.1.layers.1.conv_normal.net.3](args = (%blocks_2_1_layers_1_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_2_1_layers_1_conv_normal_net_4 : [#users=1] = call_module[target=blocks.2.1.layers.1.conv_normal.net.4](args = (%blocks_2_1_layers_1_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_2_1_layers_1_conv_normal_net_5 : [#users=1] = call_module[target=blocks.2.1.layers.1.conv_normal.net.5](args = (%blocks_2_1_layers_1_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_33 : [#users=3] = call_function[target=torch.cat](args = ([%cat_30, %blocks_2_1_layers_1_conv_down_net_5, %blocks_2_1_layers_1_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_2_1_layers_2_conv_down_net_0 : [#users=1] = call_module[target=blocks.2.1.layers.2.conv_down.net.0](args = (%cat_30,), kwargs = {}) | fixed
    %blocks_2_1_layers_2_conv_down_net_1 : [#users=1] = call_module[target=blocks.2.1.layers.2.conv_down.net.1](args = (%blocks_2_1_layers_2_conv_down_net_0,), kwargs = {}) | fixed
    %blocks_2_1_layers_2_conv_down_net_2 : [#users=1] = call_module[target=blocks.2.1.layers.2.conv_down.net.2](args = (%blocks_2_1_layers_2_conv_down_net_1,), kwargs = {}) | fixed
    %blocks_2_1_layers_2_conv_down_net_3 : [#users=1] = call_module[target=blocks.2.1.layers.2.conv_down.net.3](args = (%blocks_2_1_layers_2_conv_down_net_2,), kwargs = {}) | fixed
    %blocks_2_1_layers_2_conv_down_net_4 : [#users=1] = call_module[target=blocks.2.1.layers.2.conv_down.net.4](args = (%blocks_2_1_layers_2_conv_down_net_3,), kwargs = {}) | fixed
    %blocks_2_1_layers_2_conv_down_net_5 : [#users=1] = call_module[target=blocks.2.1.layers.2.conv_down.net.5](args = (%blocks_2_1_layers_2_conv_down_net_4,), kwargs = {}) | fixed
    %blocks_2_1_layers_2_conv_normal_net_0 : [#users=1] = call_module[target=blocks.2.1.layers.2.conv_normal.net.0](args = (%cat_31,), kwargs = {}) | fixed
    %blocks_2_1_layers_2_conv_normal_net_1 : [#users=1] = call_module[target=blocks.2.1.layers.2.conv_normal.net.1](args = (%blocks_2_1_layers_2_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_2_1_layers_2_conv_normal_net_2 : [#users=1] = call_module[target=blocks.2.1.layers.2.conv_normal.net.2](args = (%blocks_2_1_layers_2_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_2_1_layers_2_conv_normal_net_3 : [#users=1] = call_module[target=blocks.2.1.layers.2.conv_normal.net.3](args = (%blocks_2_1_layers_2_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_2_1_layers_2_conv_normal_net_4 : [#users=1] = call_module[target=blocks.2.1.layers.2.conv_normal.net.4](args = (%blocks_2_1_layers_2_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_2_1_layers_2_conv_normal_net_5 : [#users=1] = call_module[target=blocks.2.1.layers.2.conv_normal.net.5](args = (%blocks_2_1_layers_2_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_34 : [#users=2] = call_function[target=torch.cat](args = ([%cat_31, %blocks_2_1_layers_2_conv_down_net_5, %blocks_2_1_layers_2_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_2_2_layers_0_conv_down_net_0 : [#users=1] = call_module[target=blocks.2.2.layers.0.conv_down.net.0](args = (%cat_32,), kwargs = {}) | fixed
    %blocks_2_2_layers_0_conv_down_net_1 : [#users=1] = call_module[target=blocks.2.2.layers.0.conv_down.net.1](args = (%blocks_2_2_layers_0_conv_down_net_0,), kwargs = {}) | fixed
    %blocks_2_2_layers_0_conv_down_net_2 : [#users=1] = call_module[target=blocks.2.2.layers.0.conv_down.net.2](args = (%blocks_2_2_layers_0_conv_down_net_1,), kwargs = {}) | fixed
    %blocks_2_2_layers_0_conv_down_net_3 : [#users=1] = call_module[target=blocks.2.2.layers.0.conv_down.net.3](args = (%blocks_2_2_layers_0_conv_down_net_2,), kwargs = {}) | fixed
    %blocks_2_2_layers_0_conv_down_net_4 : [#users=1] = call_module[target=blocks.2.2.layers.0.conv_down.net.4](args = (%blocks_2_2_layers_0_conv_down_net_3,), kwargs = {}) | fixed
    %blocks_2_2_layers_0_conv_down_net_5 : [#users=1] = call_module[target=blocks.2.2.layers.0.conv_down.net.5](args = (%blocks_2_2_layers_0_conv_down_net_4,), kwargs = {}) | fixed
    %blocks_2_2_layers_0_conv_normal_net_0 : [#users=1] = call_module[target=blocks.2.2.layers.0.conv_normal.net.0](args = (%cat_33,), kwargs = {}) | fixed
    %blocks_2_2_layers_0_conv_normal_net_1 : [#users=1] = call_module[target=blocks.2.2.layers.0.conv_normal.net.1](args = (%blocks_2_2_layers_0_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_2_2_layers_0_conv_normal_net_2 : [#users=1] = call_module[target=blocks.2.2.layers.0.conv_normal.net.2](args = (%blocks_2_2_layers_0_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_2_2_layers_0_conv_normal_net_3 : [#users=1] = call_module[target=blocks.2.2.layers.0.conv_normal.net.3](args = (%blocks_2_2_layers_0_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_2_2_layers_0_conv_normal_net_4 : [#users=1] = call_module[target=blocks.2.2.layers.0.conv_normal.net.4](args = (%blocks_2_2_layers_0_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_2_2_layers_0_conv_normal_net_5 : [#users=1] = call_module[target=blocks.2.2.layers.0.conv_normal.net.5](args = (%blocks_2_2_layers_0_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_35 : [#users=1] = call_function[target=torch.cat](args = ([%cat_33, %blocks_2_2_layers_0_conv_down_net_5, %blocks_2_2_layers_0_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_2_2_layers_1_conv_down_net_0 : [#users=1] = call_module[target=blocks.2.2.layers.1.conv_down.net.0](args = (%cat_33,), kwargs = {}) | fixed
    %blocks_2_2_layers_1_conv_down_net_1 : [#users=1] = call_module[target=blocks.2.2.layers.1.conv_down.net.1](args = (%blocks_2_2_layers_1_conv_down_net_0,), kwargs = {}) | fixed
    %blocks_2_2_layers_1_conv_down_net_2 : [#users=1] = call_module[target=blocks.2.2.layers.1.conv_down.net.2](args = (%blocks_2_2_layers_1_conv_down_net_1,), kwargs = {}) | fixed
    %blocks_2_2_layers_1_conv_down_net_3 : [#users=1] = call_module[target=blocks.2.2.layers.1.conv_down.net.3](args = (%blocks_2_2_layers_1_conv_down_net_2,), kwargs = {}) | fixed
    %blocks_2_2_layers_1_conv_down_net_4 : [#users=1] = call_module[target=blocks.2.2.layers.1.conv_down.net.4](args = (%blocks_2_2_layers_1_conv_down_net_3,), kwargs = {}) | fixed
    %blocks_2_2_layers_1_conv_down_net_5 : [#users=1] = call_module[target=blocks.2.2.layers.1.conv_down.net.5](args = (%blocks_2_2_layers_1_conv_down_net_4,), kwargs = {}) | fixed
    %blocks_2_2_layers_1_conv_normal_net_0 : [#users=1] = call_module[target=blocks.2.2.layers.1.conv_normal.net.0](args = (%cat_34,), kwargs = {}) | fixed
    %blocks_2_2_layers_1_conv_normal_net_1 : [#users=1] = call_module[target=blocks.2.2.layers.1.conv_normal.net.1](args = (%blocks_2_2_layers_1_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_2_2_layers_1_conv_normal_net_2 : [#users=1] = call_module[target=blocks.2.2.layers.1.conv_normal.net.2](args = (%blocks_2_2_layers_1_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_2_2_layers_1_conv_normal_net_3 : [#users=1] = call_module[target=blocks.2.2.layers.1.conv_normal.net.3](args = (%blocks_2_2_layers_1_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_2_2_layers_1_conv_normal_net_4 : [#users=1] = call_module[target=blocks.2.2.layers.1.conv_normal.net.4](args = (%blocks_2_2_layers_1_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_2_2_layers_1_conv_normal_net_5 : [#users=1] = call_module[target=blocks.2.2.layers.1.conv_normal.net.5](args = (%blocks_2_2_layers_1_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_36 : [#users=1] = call_function[target=torch.cat](args = ([%cat_34, %blocks_2_2_layers_1_conv_down_net_5, %blocks_2_2_layers_1_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_2_3_m_0_net_0 : [#users=1] = call_module[target=blocks.2.3.m.0.net.0](args = (%cat_35,), kwargs = {}) | fixed
    %blocks_2_3_m_0_net_1 : [#users=1] = call_module[target=blocks.2.3.m.0.net.1](args = (%blocks_2_3_m_0_net_0,), kwargs = {}) | fixed
    %blocks_2_3_m_0_net_2 : [#users=3] = call_module[target=blocks.2.3.m.0.net.2](args = (%blocks_2_3_m_0_net_1,), kwargs = {}) | fixed
    %blocks_2_3_m_1_net_0 : [#users=1] = call_module[target=blocks.2.3.m.1.net.0](args = (%cat_36,), kwargs = {}) | fixed
    %blocks_2_3_m_1_net_1 : [#users=1] = call_module[target=blocks.2.3.m.1.net.1](args = (%blocks_2_3_m_1_net_0,), kwargs = {}) | fixed
    %blocks_2_3_m_1_net_2 : [#users=2] = call_module[target=blocks.2.3.m.1.net.2](args = (%blocks_2_3_m_1_net_1,), kwargs = {}) | fixed
    %blocks_2_4_layers_0_conv_normal_net_0 : [#users=1] = call_module[target=blocks.2.4.layers.0.conv_normal.net.0](args = (%blocks_2_3_m_0_net_2,), kwargs = {}) | fixed
    %blocks_2_4_layers_0_conv_normal_net_1 : [#users=1] = call_module[target=blocks.2.4.layers.0.conv_normal.net.1](args = (%blocks_2_4_layers_0_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_2_4_layers_0_conv_normal_net_2 : [#users=1] = call_module[target=blocks.2.4.layers.0.conv_normal.net.2](args = (%blocks_2_4_layers_0_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_2_4_layers_0_conv_normal_net_3 : [#users=1] = call_module[target=blocks.2.4.layers.0.conv_normal.net.3](args = (%blocks_2_4_layers_0_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_2_4_layers_0_conv_normal_net_4 : [#users=1] = call_module[target=blocks.2.4.layers.0.conv_normal.net.4](args = (%blocks_2_4_layers_0_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_2_4_layers_0_conv_normal_net_5 : [#users=1] = call_module[target=blocks.2.4.layers.0.conv_normal.net.5](args = (%blocks_2_4_layers_0_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_37 : [#users=1] = call_function[target=torch.cat](args = ([%blocks_2_3_m_0_net_2, %blocks_2_4_layers_0_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_2_4_layers_1_conv_down_net_0 : [#users=1] = call_module[target=blocks.2.4.layers.1.conv_down.net.0](args = (%blocks_2_3_m_0_net_2,), kwargs = {}) | fixed
    %blocks_2_4_layers_1_conv_down_net_1 : [#users=1] = call_module[target=blocks.2.4.layers.1.conv_down.net.1](args = (%blocks_2_4_layers_1_conv_down_net_0,), kwargs = {}) | fixed
    %blocks_2_4_layers_1_conv_down_net_2 : [#users=1] = call_module[target=blocks.2.4.layers.1.conv_down.net.2](args = (%blocks_2_4_layers_1_conv_down_net_1,), kwargs = {}) | fixed
    %blocks_2_4_layers_1_conv_down_net_3 : [#users=1] = call_module[target=blocks.2.4.layers.1.conv_down.net.3](args = (%blocks_2_4_layers_1_conv_down_net_2,), kwargs = {}) | fixed
    %blocks_2_4_layers_1_conv_down_net_4 : [#users=1] = call_module[target=blocks.2.4.layers.1.conv_down.net.4](args = (%blocks_2_4_layers_1_conv_down_net_3,), kwargs = {}) | fixed
    %blocks_2_4_layers_1_conv_down_net_5 : [#users=1] = call_module[target=blocks.2.4.layers.1.conv_down.net.5](args = (%blocks_2_4_layers_1_conv_down_net_4,), kwargs = {}) | fixed
    %blocks_2_4_layers_1_conv_normal_net_0 : [#users=1] = call_module[target=blocks.2.4.layers.1.conv_normal.net.0](args = (%blocks_2_3_m_1_net_2,), kwargs = {}) | fixed
    %blocks_2_4_layers_1_conv_normal_net_1 : [#users=1] = call_module[target=blocks.2.4.layers.1.conv_normal.net.1](args = (%blocks_2_4_layers_1_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_2_4_layers_1_conv_normal_net_2 : [#users=1] = call_module[target=blocks.2.4.layers.1.conv_normal.net.2](args = (%blocks_2_4_layers_1_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_2_4_layers_1_conv_normal_net_3 : [#users=1] = call_module[target=blocks.2.4.layers.1.conv_normal.net.3](args = (%blocks_2_4_layers_1_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_2_4_layers_1_conv_normal_net_4 : [#users=1] = call_module[target=blocks.2.4.layers.1.conv_normal.net.4](args = (%blocks_2_4_layers_1_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_2_4_layers_1_conv_normal_net_5 : [#users=1] = call_module[target=blocks.2.4.layers.1.conv_normal.net.5](args = (%blocks_2_4_layers_1_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_38 : [#users=4] = call_function[target=torch.cat](args = ([%blocks_2_3_m_1_net_2, %blocks_2_4_layers_1_conv_down_net_5, %blocks_2_4_layers_1_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %classifier_2_m_0_net_0 : [#users=1] = call_module[target=classifier.2.m.0.net.0](args = (%cat_38,), kwargs = {}) | fixed
    %classifier_2_m_0_net_1 : [#users=1] = call_module[target=classifier.2.m.0.net.1](args = (%classifier_2_m_0_net_0,), kwargs = {}) | fixed
    %classifier_2_m_0_net_2 : [#users=1] = call_module[target=classifier.2.m.0.net.2](args = (%classifier_2_m_0_net_1,), kwargs = {}) | fixed
    %classifier_2_m_1_net_0 : [#users=1] = call_module[target=classifier.2.m.1.net.0](args = (%classifier_2_m_0_net_2,), kwargs = {}) | fixed
    %classifier_2_m_1_net_1 : [#users=1] = call_module[target=classifier.2.m.1.net.1](args = (%classifier_2_m_1_net_0,), kwargs = {}) | fixed
    %classifier_2_m_1_net_2 : [#users=1] = call_module[target=classifier.2.m.1.net.2](args = (%classifier_2_m_1_net_1,), kwargs = {}) | fixed
    %classifier_2_m_2 : [#users=2] = call_module[target=classifier.2.m.2](args = (%classifier_2_m_1_net_2,), kwargs = {}) | fixed
    %size_2 : [#users=1] = call_method[target=size](args = (%classifier_2_m_2, 0), kwargs = {}) | fixed
    %view_2 : [#users=1] = call_method[target=view](args = (%classifier_2_m_2, %size_2, 352), kwargs = {}) | unfixed
    %classifier_2_linear : [#users=1] = call_module[target=classifier.2.linear](args = (%view_2,), kwargs = {}) | unfixed
    %zeros_like_1 : [#users=1] = call_function[target=torch.zeros_like](args = (%cat_38,), kwargs = {}) | fixed
    %zeros_like_2 : [#users=1] = call_function[target=torch.zeros_like](args = (%cat_38,), kwargs = {}) | fixed
    %scatters_2 : [#users=5] = call_module[target=scatters.2](args = ([%cat_37, %cat_38, %zeros_like_1, %zeros_like_2, %classifier_2_linear], %classifier_2_linear), kwargs = {}) | fixed
    %getitem_19 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_2, 0), kwargs = {}) | fixed
    %getitem_20 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_2, 1), kwargs = {}) | fixed
    %getitem_21 : [#users=0] = call_function[target=operator.getitem](args = (%scatters_2, 2), kwargs = {}) | fixed
    %getitem_22 : [#users=0] = call_function[target=operator.getitem](args = (%scatters_2, 3), kwargs = {}) | fixed
    %getitem_23 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_2, 4), kwargs = {}) | fixed
    %getitem_24 : [#users=3] = call_function[target=operator.getitem](args = (%getitem_19, 1), kwargs = {}) | fixed
    %getitem_25 : [#users=2] = call_function[target=operator.getitem](args = (%getitem_20, 1), kwargs = {}) | fixed
    %getitem_26 : [#users=1] = call_function[target=operator.getitem](args = (%getitem_23, 0), kwargs = {}) | fixed
    %blocks_3_0_layers_0_conv_normal_net_0 : [#users=1] = call_module[target=blocks.3.0.layers.0.conv_normal.net.0](args = (%getitem_24,), kwargs = {}) | fixed
    %blocks_3_0_layers_0_conv_normal_net_1 : [#users=1] = call_module[target=blocks.3.0.layers.0.conv_normal.net.1](args = (%blocks_3_0_layers_0_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_3_0_layers_0_conv_normal_net_2 : [#users=1] = call_module[target=blocks.3.0.layers.0.conv_normal.net.2](args = (%blocks_3_0_layers_0_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_3_0_layers_0_conv_normal_net_3 : [#users=1] = call_module[target=blocks.3.0.layers.0.conv_normal.net.3](args = (%blocks_3_0_layers_0_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_3_0_layers_0_conv_normal_net_4 : [#users=1] = call_module[target=blocks.3.0.layers.0.conv_normal.net.4](args = (%blocks_3_0_layers_0_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_3_0_layers_0_conv_normal_net_5 : [#users=1] = call_module[target=blocks.3.0.layers.0.conv_normal.net.5](args = (%blocks_3_0_layers_0_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_39 : [#users=3] = call_function[target=torch.cat](args = ([%getitem_24, %blocks_3_0_layers_0_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_3_0_layers_1_conv_down_net_0 : [#users=1] = call_module[target=blocks.3.0.layers.1.conv_down.net.0](args = (%getitem_24,), kwargs = {}) | fixed
    %blocks_3_0_layers_1_conv_down_net_1 : [#users=1] = call_module[target=blocks.3.0.layers.1.conv_down.net.1](args = (%blocks_3_0_layers_1_conv_down_net_0,), kwargs = {}) | fixed
    %blocks_3_0_layers_1_conv_down_net_2 : [#users=1] = call_module[target=blocks.3.0.layers.1.conv_down.net.2](args = (%blocks_3_0_layers_1_conv_down_net_1,), kwargs = {}) | fixed
    %blocks_3_0_layers_1_conv_down_net_3 : [#users=1] = call_module[target=blocks.3.0.layers.1.conv_down.net.3](args = (%blocks_3_0_layers_1_conv_down_net_2,), kwargs = {}) | fixed
    %blocks_3_0_layers_1_conv_down_net_4 : [#users=1] = call_module[target=blocks.3.0.layers.1.conv_down.net.4](args = (%blocks_3_0_layers_1_conv_down_net_3,), kwargs = {}) | fixed
    %blocks_3_0_layers_1_conv_down_net_5 : [#users=1] = call_module[target=blocks.3.0.layers.1.conv_down.net.5](args = (%blocks_3_0_layers_1_conv_down_net_4,), kwargs = {}) | fixed
    %blocks_3_0_layers_1_conv_normal_net_0 : [#users=1] = call_module[target=blocks.3.0.layers.1.conv_normal.net.0](args = (%getitem_25,), kwargs = {}) | fixed
    %blocks_3_0_layers_1_conv_normal_net_1 : [#users=1] = call_module[target=blocks.3.0.layers.1.conv_normal.net.1](args = (%blocks_3_0_layers_1_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_3_0_layers_1_conv_normal_net_2 : [#users=1] = call_module[target=blocks.3.0.layers.1.conv_normal.net.2](args = (%blocks_3_0_layers_1_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_3_0_layers_1_conv_normal_net_3 : [#users=1] = call_module[target=blocks.3.0.layers.1.conv_normal.net.3](args = (%blocks_3_0_layers_1_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_3_0_layers_1_conv_normal_net_4 : [#users=1] = call_module[target=blocks.3.0.layers.1.conv_normal.net.4](args = (%blocks_3_0_layers_1_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_3_0_layers_1_conv_normal_net_5 : [#users=1] = call_module[target=blocks.3.0.layers.1.conv_normal.net.5](args = (%blocks_3_0_layers_1_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_40 : [#users=2] = call_function[target=torch.cat](args = ([%getitem_25, %blocks_3_0_layers_1_conv_down_net_5, %blocks_3_0_layers_1_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_3_1_layers_0_conv_normal_net_0 : [#users=1] = call_module[target=blocks.3.1.layers.0.conv_normal.net.0](args = (%cat_39,), kwargs = {}) | fixed
    %blocks_3_1_layers_0_conv_normal_net_1 : [#users=1] = call_module[target=blocks.3.1.layers.0.conv_normal.net.1](args = (%blocks_3_1_layers_0_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_3_1_layers_0_conv_normal_net_2 : [#users=1] = call_module[target=blocks.3.1.layers.0.conv_normal.net.2](args = (%blocks_3_1_layers_0_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_3_1_layers_0_conv_normal_net_3 : [#users=1] = call_module[target=blocks.3.1.layers.0.conv_normal.net.3](args = (%blocks_3_1_layers_0_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_3_1_layers_0_conv_normal_net_4 : [#users=1] = call_module[target=blocks.3.1.layers.0.conv_normal.net.4](args = (%blocks_3_1_layers_0_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_3_1_layers_0_conv_normal_net_5 : [#users=1] = call_module[target=blocks.3.1.layers.0.conv_normal.net.5](args = (%blocks_3_1_layers_0_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_41 : [#users=3] = call_function[target=torch.cat](args = ([%cat_39, %blocks_3_1_layers_0_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_3_1_layers_1_conv_down_net_0 : [#users=1] = call_module[target=blocks.3.1.layers.1.conv_down.net.0](args = (%cat_39,), kwargs = {}) | fixed
    %blocks_3_1_layers_1_conv_down_net_1 : [#users=1] = call_module[target=blocks.3.1.layers.1.conv_down.net.1](args = (%blocks_3_1_layers_1_conv_down_net_0,), kwargs = {}) | fixed
    %blocks_3_1_layers_1_conv_down_net_2 : [#users=1] = call_module[target=blocks.3.1.layers.1.conv_down.net.2](args = (%blocks_3_1_layers_1_conv_down_net_1,), kwargs = {}) | fixed
    %blocks_3_1_layers_1_conv_down_net_3 : [#users=1] = call_module[target=blocks.3.1.layers.1.conv_down.net.3](args = (%blocks_3_1_layers_1_conv_down_net_2,), kwargs = {}) | fixed
    %blocks_3_1_layers_1_conv_down_net_4 : [#users=1] = call_module[target=blocks.3.1.layers.1.conv_down.net.4](args = (%blocks_3_1_layers_1_conv_down_net_3,), kwargs = {}) | fixed
    %blocks_3_1_layers_1_conv_down_net_5 : [#users=1] = call_module[target=blocks.3.1.layers.1.conv_down.net.5](args = (%blocks_3_1_layers_1_conv_down_net_4,), kwargs = {}) | fixed
    %blocks_3_1_layers_1_conv_normal_net_0 : [#users=1] = call_module[target=blocks.3.1.layers.1.conv_normal.net.0](args = (%cat_40,), kwargs = {}) | fixed
    %blocks_3_1_layers_1_conv_normal_net_1 : [#users=1] = call_module[target=blocks.3.1.layers.1.conv_normal.net.1](args = (%blocks_3_1_layers_1_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_3_1_layers_1_conv_normal_net_2 : [#users=1] = call_module[target=blocks.3.1.layers.1.conv_normal.net.2](args = (%blocks_3_1_layers_1_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_3_1_layers_1_conv_normal_net_3 : [#users=1] = call_module[target=blocks.3.1.layers.1.conv_normal.net.3](args = (%blocks_3_1_layers_1_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_3_1_layers_1_conv_normal_net_4 : [#users=1] = call_module[target=blocks.3.1.layers.1.conv_normal.net.4](args = (%blocks_3_1_layers_1_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_3_1_layers_1_conv_normal_net_5 : [#users=1] = call_module[target=blocks.3.1.layers.1.conv_normal.net.5](args = (%blocks_3_1_layers_1_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_42 : [#users=2] = call_function[target=torch.cat](args = ([%cat_40, %blocks_3_1_layers_1_conv_down_net_5, %blocks_3_1_layers_1_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_3_2_layers_0_conv_normal_net_0 : [#users=1] = call_module[target=blocks.3.2.layers.0.conv_normal.net.0](args = (%cat_41,), kwargs = {}) | fixed
    %blocks_3_2_layers_0_conv_normal_net_1 : [#users=1] = call_module[target=blocks.3.2.layers.0.conv_normal.net.1](args = (%blocks_3_2_layers_0_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_3_2_layers_0_conv_normal_net_2 : [#users=1] = call_module[target=blocks.3.2.layers.0.conv_normal.net.2](args = (%blocks_3_2_layers_0_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_3_2_layers_0_conv_normal_net_3 : [#users=1] = call_module[target=blocks.3.2.layers.0.conv_normal.net.3](args = (%blocks_3_2_layers_0_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_3_2_layers_0_conv_normal_net_4 : [#users=1] = call_module[target=blocks.3.2.layers.0.conv_normal.net.4](args = (%blocks_3_2_layers_0_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_3_2_layers_0_conv_normal_net_5 : [#users=1] = call_module[target=blocks.3.2.layers.0.conv_normal.net.5](args = (%blocks_3_2_layers_0_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_43 : [#users=1] = call_function[target=torch.cat](args = ([%cat_41, %blocks_3_2_layers_0_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_3_2_layers_1_conv_down_net_0 : [#users=1] = call_module[target=blocks.3.2.layers.1.conv_down.net.0](args = (%cat_41,), kwargs = {}) | fixed
    %blocks_3_2_layers_1_conv_down_net_1 : [#users=1] = call_module[target=blocks.3.2.layers.1.conv_down.net.1](args = (%blocks_3_2_layers_1_conv_down_net_0,), kwargs = {}) | fixed
    %blocks_3_2_layers_1_conv_down_net_2 : [#users=1] = call_module[target=blocks.3.2.layers.1.conv_down.net.2](args = (%blocks_3_2_layers_1_conv_down_net_1,), kwargs = {}) | fixed
    %blocks_3_2_layers_1_conv_down_net_3 : [#users=1] = call_module[target=blocks.3.2.layers.1.conv_down.net.3](args = (%blocks_3_2_layers_1_conv_down_net_2,), kwargs = {}) | fixed
    %blocks_3_2_layers_1_conv_down_net_4 : [#users=1] = call_module[target=blocks.3.2.layers.1.conv_down.net.4](args = (%blocks_3_2_layers_1_conv_down_net_3,), kwargs = {}) | fixed
    %blocks_3_2_layers_1_conv_down_net_5 : [#users=1] = call_module[target=blocks.3.2.layers.1.conv_down.net.5](args = (%blocks_3_2_layers_1_conv_down_net_4,), kwargs = {}) | fixed
    %blocks_3_2_layers_1_conv_normal_net_0 : [#users=1] = call_module[target=blocks.3.2.layers.1.conv_normal.net.0](args = (%cat_42,), kwargs = {}) | fixed
    %blocks_3_2_layers_1_conv_normal_net_1 : [#users=1] = call_module[target=blocks.3.2.layers.1.conv_normal.net.1](args = (%blocks_3_2_layers_1_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_3_2_layers_1_conv_normal_net_2 : [#users=1] = call_module[target=blocks.3.2.layers.1.conv_normal.net.2](args = (%blocks_3_2_layers_1_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_3_2_layers_1_conv_normal_net_3 : [#users=1] = call_module[target=blocks.3.2.layers.1.conv_normal.net.3](args = (%blocks_3_2_layers_1_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_3_2_layers_1_conv_normal_net_4 : [#users=1] = call_module[target=blocks.3.2.layers.1.conv_normal.net.4](args = (%blocks_3_2_layers_1_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_3_2_layers_1_conv_normal_net_5 : [#users=1] = call_module[target=blocks.3.2.layers.1.conv_normal.net.5](args = (%blocks_3_2_layers_1_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_44 : [#users=2] = call_function[target=torch.cat](args = ([%cat_42, %blocks_3_2_layers_1_conv_down_net_5, %blocks_3_2_layers_1_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_3_3_layers_0_conv_down_net_0 : [#users=1] = call_module[target=blocks.3.3.layers.0.conv_down.net.0](args = (%cat_43,), kwargs = {}) | fixed
    %blocks_3_3_layers_0_conv_down_net_1 : [#users=1] = call_module[target=blocks.3.3.layers.0.conv_down.net.1](args = (%blocks_3_3_layers_0_conv_down_net_0,), kwargs = {}) | fixed
    %blocks_3_3_layers_0_conv_down_net_2 : [#users=1] = call_module[target=blocks.3.3.layers.0.conv_down.net.2](args = (%blocks_3_3_layers_0_conv_down_net_1,), kwargs = {}) | fixed
    %blocks_3_3_layers_0_conv_down_net_3 : [#users=1] = call_module[target=blocks.3.3.layers.0.conv_down.net.3](args = (%blocks_3_3_layers_0_conv_down_net_2,), kwargs = {}) | fixed
    %blocks_3_3_layers_0_conv_down_net_4 : [#users=1] = call_module[target=blocks.3.3.layers.0.conv_down.net.4](args = (%blocks_3_3_layers_0_conv_down_net_3,), kwargs = {}) | fixed
    %blocks_3_3_layers_0_conv_down_net_5 : [#users=1] = call_module[target=blocks.3.3.layers.0.conv_down.net.5](args = (%blocks_3_3_layers_0_conv_down_net_4,), kwargs = {}) | fixed
    %blocks_3_3_layers_0_conv_normal_net_0 : [#users=1] = call_module[target=blocks.3.3.layers.0.conv_normal.net.0](args = (%cat_44,), kwargs = {}) | fixed
    %blocks_3_3_layers_0_conv_normal_net_1 : [#users=1] = call_module[target=blocks.3.3.layers.0.conv_normal.net.1](args = (%blocks_3_3_layers_0_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_3_3_layers_0_conv_normal_net_2 : [#users=1] = call_module[target=blocks.3.3.layers.0.conv_normal.net.2](args = (%blocks_3_3_layers_0_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_3_3_layers_0_conv_normal_net_3 : [#users=1] = call_module[target=blocks.3.3.layers.0.conv_normal.net.3](args = (%blocks_3_3_layers_0_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_3_3_layers_0_conv_normal_net_4 : [#users=1] = call_module[target=blocks.3.3.layers.0.conv_normal.net.4](args = (%blocks_3_3_layers_0_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_3_3_layers_0_conv_normal_net_5 : [#users=1] = call_module[target=blocks.3.3.layers.0.conv_normal.net.5](args = (%blocks_3_3_layers_0_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_45 : [#users=1] = call_function[target=torch.cat](args = ([%cat_44, %blocks_3_3_layers_0_conv_down_net_5, %blocks_3_3_layers_0_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_3_4_m_0_net_0 : [#users=1] = call_module[target=blocks.3.4.m.0.net.0](args = (%cat_45,), kwargs = {}) | fixed
    %blocks_3_4_m_0_net_1 : [#users=1] = call_module[target=blocks.3.4.m.0.net.1](args = (%blocks_3_4_m_0_net_0,), kwargs = {}) | fixed
    %blocks_3_4_m_0_net_2 : [#users=5] = call_module[target=blocks.3.4.m.0.net.2](args = (%blocks_3_4_m_0_net_1,), kwargs = {}) | fixed
    %classifier_3_m_0_net_0 : [#users=1] = call_module[target=classifier.3.m.0.net.0](args = (%blocks_3_4_m_0_net_2,), kwargs = {}) | fixed
    %classifier_3_m_0_net_1 : [#users=1] = call_module[target=classifier.3.m.0.net.1](args = (%classifier_3_m_0_net_0,), kwargs = {}) | fixed
    %classifier_3_m_0_net_2 : [#users=1] = call_module[target=classifier.3.m.0.net.2](args = (%classifier_3_m_0_net_1,), kwargs = {}) | fixed
    %classifier_3_m_1_net_0 : [#users=1] = call_module[target=classifier.3.m.1.net.0](args = (%classifier_3_m_0_net_2,), kwargs = {}) | fixed
    %classifier_3_m_1_net_1 : [#users=1] = call_module[target=classifier.3.m.1.net.1](args = (%classifier_3_m_1_net_0,), kwargs = {}) | fixed
    %classifier_3_m_1_net_2 : [#users=1] = call_module[target=classifier.3.m.1.net.2](args = (%classifier_3_m_1_net_1,), kwargs = {}) | fixed
    %classifier_3_m_2 : [#users=2] = call_module[target=classifier.3.m.2](args = (%classifier_3_m_1_net_2,), kwargs = {}) | fixed
    %size_3 : [#users=1] = call_method[target=size](args = (%classifier_3_m_2, 0), kwargs = {}) | fixed
    %view_3 : [#users=1] = call_method[target=view](args = (%classifier_3_m_2, %size_3, 304), kwargs = {}) | unfixed
    %classifier_3_linear : [#users=1] = call_module[target=classifier.3.linear](args = (%view_3,), kwargs = {}) | unfixed
    %zeros_like_3 : [#users=1] = call_function[target=torch.zeros_like](args = (%blocks_3_4_m_0_net_2,), kwargs = {}) | fixed
    %zeros_like_4 : [#users=1] = call_function[target=torch.zeros_like](args = (%blocks_3_4_m_0_net_2,), kwargs = {}) | fixed
    %zeros_like_5 : [#users=1] = call_function[target=torch.zeros_like](args = (%blocks_3_4_m_0_net_2,), kwargs = {}) | fixed
    %scatters_3 : [#users=5] = call_module[target=scatters.3](args = ([%blocks_3_4_m_0_net_2, %zeros_like_3, %zeros_like_4, %zeros_like_5, %classifier_3_linear], %classifier_3_linear), kwargs = {}) | fixed
    %getitem_27 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_3, 0), kwargs = {}) | fixed
    %getitem_28 : [#users=0] = call_function[target=operator.getitem](args = (%scatters_3, 1), kwargs = {}) | fixed
    %getitem_29 : [#users=0] = call_function[target=operator.getitem](args = (%scatters_3, 2), kwargs = {}) | fixed
    %getitem_30 : [#users=0] = call_function[target=operator.getitem](args = (%scatters_3, 3), kwargs = {}) | fixed
    %getitem_31 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_3, 4), kwargs = {}) | fixed
    %getitem_32 : [#users=2] = call_function[target=operator.getitem](args = (%getitem_27, 1), kwargs = {}) | fixed
    %getitem_33 : [#users=1] = call_function[target=operator.getitem](args = (%getitem_31, 0), kwargs = {}) | fixed
    %blocks_4_0_layers_0_conv_normal_net_0 : [#users=1] = call_module[target=blocks.4.0.layers.0.conv_normal.net.0](args = (%getitem_32,), kwargs = {}) | fixed
    %blocks_4_0_layers_0_conv_normal_net_1 : [#users=1] = call_module[target=blocks.4.0.layers.0.conv_normal.net.1](args = (%blocks_4_0_layers_0_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_4_0_layers_0_conv_normal_net_2 : [#users=1] = call_module[target=blocks.4.0.layers.0.conv_normal.net.2](args = (%blocks_4_0_layers_0_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_4_0_layers_0_conv_normal_net_3 : [#users=1] = call_module[target=blocks.4.0.layers.0.conv_normal.net.3](args = (%blocks_4_0_layers_0_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_4_0_layers_0_conv_normal_net_4 : [#users=1] = call_module[target=blocks.4.0.layers.0.conv_normal.net.4](args = (%blocks_4_0_layers_0_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_4_0_layers_0_conv_normal_net_5 : [#users=1] = call_module[target=blocks.4.0.layers.0.conv_normal.net.5](args = (%blocks_4_0_layers_0_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_46 : [#users=2] = call_function[target=torch.cat](args = ([%getitem_32, %blocks_4_0_layers_0_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_4_1_layers_0_conv_normal_net_0 : [#users=1] = call_module[target=blocks.4.1.layers.0.conv_normal.net.0](args = (%cat_46,), kwargs = {}) | fixed
    %blocks_4_1_layers_0_conv_normal_net_1 : [#users=1] = call_module[target=blocks.4.1.layers.0.conv_normal.net.1](args = (%blocks_4_1_layers_0_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_4_1_layers_0_conv_normal_net_2 : [#users=1] = call_module[target=blocks.4.1.layers.0.conv_normal.net.2](args = (%blocks_4_1_layers_0_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_4_1_layers_0_conv_normal_net_3 : [#users=1] = call_module[target=blocks.4.1.layers.0.conv_normal.net.3](args = (%blocks_4_1_layers_0_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_4_1_layers_0_conv_normal_net_4 : [#users=1] = call_module[target=blocks.4.1.layers.0.conv_normal.net.4](args = (%blocks_4_1_layers_0_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_4_1_layers_0_conv_normal_net_5 : [#users=1] = call_module[target=blocks.4.1.layers.0.conv_normal.net.5](args = (%blocks_4_1_layers_0_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_47 : [#users=2] = call_function[target=torch.cat](args = ([%cat_46, %blocks_4_1_layers_0_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_4_2_layers_0_conv_normal_net_0 : [#users=1] = call_module[target=blocks.4.2.layers.0.conv_normal.net.0](args = (%cat_47,), kwargs = {}) | fixed
    %blocks_4_2_layers_0_conv_normal_net_1 : [#users=1] = call_module[target=blocks.4.2.layers.0.conv_normal.net.1](args = (%blocks_4_2_layers_0_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_4_2_layers_0_conv_normal_net_2 : [#users=1] = call_module[target=blocks.4.2.layers.0.conv_normal.net.2](args = (%blocks_4_2_layers_0_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_4_2_layers_0_conv_normal_net_3 : [#users=1] = call_module[target=blocks.4.2.layers.0.conv_normal.net.3](args = (%blocks_4_2_layers_0_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_4_2_layers_0_conv_normal_net_4 : [#users=1] = call_module[target=blocks.4.2.layers.0.conv_normal.net.4](args = (%blocks_4_2_layers_0_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_4_2_layers_0_conv_normal_net_5 : [#users=1] = call_module[target=blocks.4.2.layers.0.conv_normal.net.5](args = (%blocks_4_2_layers_0_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_48 : [#users=2] = call_function[target=torch.cat](args = ([%cat_47, %blocks_4_2_layers_0_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %blocks_4_3_layers_0_conv_normal_net_0 : [#users=1] = call_module[target=blocks.4.3.layers.0.conv_normal.net.0](args = (%cat_48,), kwargs = {}) | fixed
    %blocks_4_3_layers_0_conv_normal_net_1 : [#users=1] = call_module[target=blocks.4.3.layers.0.conv_normal.net.1](args = (%blocks_4_3_layers_0_conv_normal_net_0,), kwargs = {}) | fixed
    %blocks_4_3_layers_0_conv_normal_net_2 : [#users=1] = call_module[target=blocks.4.3.layers.0.conv_normal.net.2](args = (%blocks_4_3_layers_0_conv_normal_net_1,), kwargs = {}) | fixed
    %blocks_4_3_layers_0_conv_normal_net_3 : [#users=1] = call_module[target=blocks.4.3.layers.0.conv_normal.net.3](args = (%blocks_4_3_layers_0_conv_normal_net_2,), kwargs = {}) | fixed
    %blocks_4_3_layers_0_conv_normal_net_4 : [#users=1] = call_module[target=blocks.4.3.layers.0.conv_normal.net.4](args = (%blocks_4_3_layers_0_conv_normal_net_3,), kwargs = {}) | fixed
    %blocks_4_3_layers_0_conv_normal_net_5 : [#users=1] = call_module[target=blocks.4.3.layers.0.conv_normal.net.5](args = (%blocks_4_3_layers_0_conv_normal_net_4,), kwargs = {}) | fixed
    %cat_49 : [#users=1] = call_function[target=torch.cat](args = ([%cat_48, %blocks_4_3_layers_0_conv_normal_net_5],), kwargs = {dim: 1}) | fixed
    %classifier_4_m_0_net_0 : [#users=1] = call_module[target=classifier.4.m.0.net.0](args = (%cat_49,), kwargs = {}) | fixed
    %classifier_4_m_0_net_1 : [#users=1] = call_module[target=classifier.4.m.0.net.1](args = (%classifier_4_m_0_net_0,), kwargs = {}) | fixed
    %classifier_4_m_0_net_2 : [#users=1] = call_module[target=classifier.4.m.0.net.2](args = (%classifier_4_m_0_net_1,), kwargs = {}) | fixed
    %classifier_4_m_1_net_0 : [#users=1] = call_module[target=classifier.4.m.1.net.0](args = (%classifier_4_m_0_net_2,), kwargs = {}) | fixed
    %classifier_4_m_1_net_1 : [#users=1] = call_module[target=classifier.4.m.1.net.1](args = (%classifier_4_m_1_net_0,), kwargs = {}) | fixed
    %classifier_4_m_1_net_2 : [#users=1] = call_module[target=classifier.4.m.1.net.2](args = (%classifier_4_m_1_net_1,), kwargs = {}) | fixed
    %classifier_4_m_2 : [#users=2] = call_module[target=classifier.4.m.2](args = (%classifier_4_m_1_net_2,), kwargs = {}) | fixed
    %size_4 : [#users=1] = call_method[target=size](args = (%classifier_4_m_2, 0), kwargs = {}) | fixed
    %view_4 : [#users=1] = call_method[target=view](args = (%classifier_4_m_2, %size_4, 560), kwargs = {}) | unfixed
    %classifier_4_linear : [#users=1] = call_module[target=classifier.4.linear](args = (%view_4,), kwargs = {}) | unfixed
    %final_gather : [#users=1] = call_module[target=final_gather](args = ([%getitem_9, %getitem_18, %getitem_26, %getitem_33, %classifier_4_linear],), kwargs = {}) | unfixed
    return final_gather | unfixed
