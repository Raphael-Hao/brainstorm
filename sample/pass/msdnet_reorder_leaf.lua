graph():
    %x : [#users=1] = placeholder[target=x]
    %_is_measure : [#users=0] = placeholder[target=_is_measure](default=False)
    %blocks_0_0_layers_0_0 : [#users=1] = call_module[target=blocks.0.0.layers.0.0](args = (%x,), kwargs = {})
    %blocks_0_0_layers_0_1 : [#users=1] = call_module[target=blocks.0.0.layers.0.1](args = (%blocks_0_0_layers_0_0,), kwargs = {})
    %blocks_0_0_layers_0_2 : [#users=1] = call_module[target=blocks.0.0.layers.0.2](args = (%blocks_0_0_layers_0_1,), kwargs = {})
    %blocks_0_0_layers_0_3 : [#users=4] = call_module[target=blocks.0.0.layers.0.3](args = (%blocks_0_0_layers_0_2,), kwargs = {})
    %blocks_0_0_layers_1_net_0 : [#users=1] = call_module[target=blocks.0.0.layers.1.net.0](args = (%blocks_0_0_layers_0_3,), kwargs = {})
    %blocks_0_1_layers_0_conv_normal : [#users=1] = call_module[target=blocks.0.1.layers.0.conv_normal](args = (%blocks_0_0_layers_0_3,), kwargs = {})
    %blocks_0_1_layers_1_conv_down : [#users=1] = call_module[target=blocks.0.1.layers.1.conv_down](args = (%blocks_0_0_layers_0_3,), kwargs = {})
    %blocks_0_0_layers_1_net_1 : [#users=1] = call_module[target=blocks.0.0.layers.1.net.1](args = (%blocks_0_0_layers_1_net_0,), kwargs = {})
    %cat : [#users=2] = call_function[target=torch.cat](args = ([%blocks_0_0_layers_0_3, %blocks_0_1_layers_0_conv_normal],), kwargs = {dim: 1})
    %blocks_0_0_layers_1_net_2 : [#users=4] = call_module[target=blocks.0.0.layers.1.net.2](args = (%blocks_0_0_layers_1_net_1,), kwargs = {})
    %blocks_0_2_layers_1_conv_down : [#users=1] = call_module[target=blocks.0.2.layers.1.conv_down](args = (%cat,), kwargs = {})
    %blocks_0_0_layers_2_net_0 : [#users=1] = call_module[target=blocks.0.0.layers.2.net.0](args = (%blocks_0_0_layers_1_net_2,), kwargs = {})
    %blocks_0_1_layers_1_conv_normal : [#users=1] = call_module[target=blocks.0.1.layers.1.conv_normal](args = (%blocks_0_0_layers_1_net_2,), kwargs = {})
    %blocks_0_1_layers_2_conv_down : [#users=1] = call_module[target=blocks.0.1.layers.2.conv_down](args = (%blocks_0_0_layers_1_net_2,), kwargs = {})
    %blocks_0_0_layers_2_net_1 : [#users=1] = call_module[target=blocks.0.0.layers.2.net.1](args = (%blocks_0_0_layers_2_net_0,), kwargs = {})
    %cat_1 : [#users=3] = call_function[target=torch.cat](args = ([%blocks_0_0_layers_1_net_2, %blocks_0_1_layers_1_conv_down, %blocks_0_1_layers_1_conv_normal],), kwargs = {dim: 1})
    %blocks_0_0_layers_2_net_2 : [#users=4] = call_module[target=blocks.0.0.layers.2.net.2](args = (%blocks_0_0_layers_2_net_1,), kwargs = {})
    %blocks_0_2_layers_1_conv_normal : [#users=1] = call_module[target=blocks.0.2.layers.1.conv_normal](args = (%cat_1,), kwargs = {})
    %blocks_0_2_layers_2_conv_down : [#users=1] = call_module[target=blocks.0.2.layers.2.conv_down](args = (%cat_1,), kwargs = {})
    %blocks_0_0_layers_3_net_0 : [#users=1] = call_module[target=blocks.0.0.layers.3.net.0](args = (%blocks_0_0_layers_2_net_2,), kwargs = {})
    %blocks_0_1_layers_2_conv_normal : [#users=1] = call_module[target=blocks.0.1.layers.2.conv_normal](args = (%blocks_0_0_layers_2_net_2,), kwargs = {})
    %blocks_0_1_layers_3_conv_down : [#users=1] = call_module[target=blocks.0.1.layers.3.conv_down](args = (%blocks_0_0_layers_2_net_2,), kwargs = {})
    %cat_5 : [#users=2] = call_function[target=torch.cat](args = ([%cat_1, %blocks_0_2_layers_1_conv_down, %blocks_0_2_layers_1_conv_normal],), kwargs = {dim: 1})
    %blocks_0_0_layers_3_net_1 : [#users=1] = call_module[target=blocks.0.0.layers.3.net.1](args = (%blocks_0_0_layers_3_net_0,), kwargs = {})
    %cat_2 : [#users=3] = call_function[target=torch.cat](args = ([%blocks_0_0_layers_2_net_2, %blocks_0_1_layers_2_conv_down, %blocks_0_1_layers_2_conv_normal],), kwargs = {dim: 1})
    %blocks_0_3_layers_2_conv_down : [#users=1] = call_module[target=blocks.0.3.layers.2.conv_down](args = (%cat_5,), kwargs = {})
    %blocks_0_0_layers_3_net_2 : [#users=2] = call_module[target=blocks.0.0.layers.3.net.2](args = (%blocks_0_0_layers_3_net_1,), kwargs = {})
    %blocks_0_2_layers_2_conv_normal : [#users=1] = call_module[target=blocks.0.2.layers.2.conv_normal](args = (%cat_2,), kwargs = {})
    %blocks_0_2_layers_3_conv_down : [#users=1] = call_module[target=blocks.0.2.layers.3.conv_down](args = (%cat_2,), kwargs = {})
    %blocks_0_1_layers_3_conv_normal : [#users=1] = call_module[target=blocks.0.1.layers.3.conv_normal](args = (%blocks_0_0_layers_3_net_2,), kwargs = {})
    %cat_6 : [#users=3] = call_function[target=torch.cat](args = ([%cat_2, %blocks_0_2_layers_2_conv_down, %blocks_0_2_layers_2_conv_normal],), kwargs = {dim: 1})
    %cat_3 : [#users=2] = call_function[target=torch.cat](args = ([%blocks_0_0_layers_3_net_2, %blocks_0_1_layers_3_conv_down, %blocks_0_1_layers_3_conv_normal],), kwargs = {dim: 1})
    %blocks_0_3_layers_2_conv_normal : [#users=1] = call_module[target=blocks.0.3.layers.2.conv_normal](args = (%cat_6,), kwargs = {})
    %blocks_0_3_layers_3_conv_down : [#users=1] = call_module[target=blocks.0.3.layers.3.conv_down](args = (%cat_6,), kwargs = {})
    %blocks_0_2_layers_3_conv_normal : [#users=1] = call_module[target=blocks.0.2.layers.3.conv_normal](args = (%cat_3,), kwargs = {})
    %cat_10 : [#users=2] = call_function[target=torch.cat](args = ([%cat_6, %blocks_0_3_layers_2_conv_down, %blocks_0_3_layers_2_conv_normal],), kwargs = {dim: 1})
    %cat_7 : [#users=2] = call_function[target=torch.cat](args = ([%cat_3, %blocks_0_2_layers_3_conv_down, %blocks_0_2_layers_3_conv_normal],), kwargs = {dim: 1})
    %blocks_0_4_layers_3_conv_down : [#users=1] = call_module[target=blocks.0.4.layers.3.conv_down](args = (%cat_10,), kwargs = {})
    %blocks_0_3_layers_3_conv_normal : [#users=1] = call_module[target=blocks.0.3.layers.3.conv_normal](args = (%cat_7,), kwargs = {})
    %cat_11 : [#users=2] = call_function[target=torch.cat](args = ([%cat_7, %blocks_0_3_layers_3_conv_down, %blocks_0_3_layers_3_conv_normal],), kwargs = {dim: 1})
    %blocks_0_4_layers_3_conv_normal : [#users=1] = call_module[target=blocks.0.4.layers.3.conv_normal](args = (%cat_11,), kwargs = {})
    %cat_15 : [#users=2] = call_function[target=torch.cat](args = ([%cat_11, %blocks_0_4_layers_3_conv_down, %blocks_0_4_layers_3_conv_normal],), kwargs = {dim: 1})
    %classifier_0_m_0_net_0 : [#users=1] = call_module[target=classifier.0.m.0.net.0](args = (%cat_15,), kwargs = {})
    %classifier_0_m_0_net_1 : [#users=1] = call_module[target=classifier.0.m.0.net.1](args = (%classifier_0_m_0_net_0,), kwargs = {})
    %classifier_0_m_0_net_2 : [#users=1] = call_module[target=classifier.0.m.0.net.2](args = (%classifier_0_m_0_net_1,), kwargs = {})
    %classifier_0_m_1_net_0 : [#users=1] = call_module[target=classifier.0.m.1.net.0](args = (%classifier_0_m_0_net_2,), kwargs = {})
    %classifier_0_m_1_net_1 : [#users=1] = call_module[target=classifier.0.m.1.net.1](args = (%classifier_0_m_1_net_0,), kwargs = {})
    %classifier_0_m_1_net_2 : [#users=1] = call_module[target=classifier.0.m.1.net.2](args = (%classifier_0_m_1_net_1,), kwargs = {})
    %classifier_0_m_2 : [#users=2] = call_module[target=classifier.0.m.2](args = (%classifier_0_m_1_net_2,), kwargs = {})
    %size : [#users=1] = call_method[target=size](args = (%classifier_0_m_2, 0), kwargs = {})
    %view : [#users=1] = call_method[target=view](args = (%classifier_0_m_2, %size, 384), kwargs = {})
    %classifier_0_linear : [#users=1] = call_module[target=classifier.0.linear](args = (%view,), kwargs = {})
    %scatters_0 : [#users=5] = call_module[target=scatters.0](args = ([%cat_10, %cat, %cat_5, %cat_15, %classifier_0_linear], %classifier_0_linear), kwargs = {})
    %getitem : [#users=1] = call_function[target=operator.getitem](args = (%scatters_0, 0), kwargs = {})
    %getitem_1 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_0, 1), kwargs = {})
    %getitem_2 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_0, 2), kwargs = {})
    %getitem_3 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_0, 3), kwargs = {})
    %getitem_4 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_0, 4), kwargs = {})
    %getitem_5 : [#users=2] = call_function[target=operator.getitem](args = (%getitem, 1), kwargs = {})
    %getitem_6 : [#users=2] = call_function[target=operator.getitem](args = (%getitem_1, 1), kwargs = {})
    %getitem_7 : [#users=2] = call_function[target=operator.getitem](args = (%getitem_2, 1), kwargs = {})
    %getitem_8 : [#users=2] = call_function[target=operator.getitem](args = (%getitem_3, 1), kwargs = {})
    %getitem_9 : [#users=1] = call_function[target=operator.getitem](args = (%getitem_4, 0), kwargs = {})
    %blocks_0_4_layers_2_conv_normal : [#users=1] = call_module[target=blocks.0.4.layers.2.conv_normal](args = (%getitem_5,), kwargs = {})
    %blocks_0_2_layers_0_conv_normal : [#users=1] = call_module[target=blocks.0.2.layers.0.conv_normal](args = (%getitem_6,), kwargs = {})
    %blocks_0_3_layers_1_conv_normal : [#users=1] = call_module[target=blocks.0.3.layers.1.conv_normal](args = (%getitem_7,), kwargs = {})
    %blocks_1_0_layers_3_conv_normal : [#users=1] = call_module[target=blocks.1.0.layers.3.conv_normal](args = (%getitem_8,), kwargs = {})
    %cat_4 : [#users=3] = call_function[target=torch.cat](args = ([%getitem_6, %blocks_0_2_layers_0_conv_normal],), kwargs = {dim: 1})
    %blocks_0_3_layers_0_conv_normal : [#users=1] = call_module[target=blocks.0.3.layers.0.conv_normal](args = (%cat_4,), kwargs = {})
    %blocks_0_3_layers_1_conv_down : [#users=1] = call_module[target=blocks.0.3.layers.1.conv_down](args = (%cat_4,), kwargs = {})
    %cat_8 : [#users=3] = call_function[target=torch.cat](args = ([%cat_4, %blocks_0_3_layers_0_conv_normal],), kwargs = {dim: 1})
    %cat_9 : [#users=3] = call_function[target=torch.cat](args = ([%getitem_7, %blocks_0_3_layers_1_conv_down, %blocks_0_3_layers_1_conv_normal],), kwargs = {dim: 1})
    %blocks_0_4_layers_0_conv_normal : [#users=1] = call_module[target=blocks.0.4.layers.0.conv_normal](args = (%cat_8,), kwargs = {})
    %blocks_0_4_layers_1_conv_down : [#users=1] = call_module[target=blocks.0.4.layers.1.conv_down](args = (%cat_8,), kwargs = {})
    %blocks_0_4_layers_1_conv_normal : [#users=1] = call_module[target=blocks.0.4.layers.1.conv_normal](args = (%cat_9,), kwargs = {})
    %blocks_0_4_layers_2_conv_down : [#users=1] = call_module[target=blocks.0.4.layers.2.conv_down](args = (%cat_9,), kwargs = {})
    %cat_12 : [#users=3] = call_function[target=torch.cat](args = ([%cat_8, %blocks_0_4_layers_0_conv_normal],), kwargs = {dim: 1})
    %cat_13 : [#users=3] = call_function[target=torch.cat](args = ([%cat_9, %blocks_0_4_layers_1_conv_down, %blocks_0_4_layers_1_conv_normal],), kwargs = {dim: 1})
    %cat_14 : [#users=3] = call_function[target=torch.cat](args = ([%getitem_5, %blocks_0_4_layers_2_conv_down, %blocks_0_4_layers_2_conv_normal],), kwargs = {dim: 1})
    %blocks_1_0_layers_0_conv_normal : [#users=1] = call_module[target=blocks.1.0.layers.0.conv_normal](args = (%cat_12,), kwargs = {})
    %blocks_1_0_layers_1_conv_down : [#users=1] = call_module[target=blocks.1.0.layers.1.conv_down](args = (%cat_12,), kwargs = {})
    %blocks_1_0_layers_1_conv_normal : [#users=1] = call_module[target=blocks.1.0.layers.1.conv_normal](args = (%cat_13,), kwargs = {})
    %blocks_1_0_layers_2_conv_down : [#users=1] = call_module[target=blocks.1.0.layers.2.conv_down](args = (%cat_13,), kwargs = {})
    %blocks_1_0_layers_2_conv_normal : [#users=1] = call_module[target=blocks.1.0.layers.2.conv_normal](args = (%cat_14,), kwargs = {})
    %blocks_1_0_layers_3_conv_down : [#users=1] = call_module[target=blocks.1.0.layers.3.conv_down](args = (%cat_14,), kwargs = {})
    %cat_16 : [#users=1] = call_function[target=torch.cat](args = ([%cat_12, %blocks_1_0_layers_0_conv_normal],), kwargs = {dim: 1})
    %cat_17 : [#users=3] = call_function[target=torch.cat](args = ([%cat_13, %blocks_1_0_layers_1_conv_down, %blocks_1_0_layers_1_conv_normal],), kwargs = {dim: 1})
    %cat_18 : [#users=3] = call_function[target=torch.cat](args = ([%cat_14, %blocks_1_0_layers_2_conv_down, %blocks_1_0_layers_2_conv_normal],), kwargs = {dim: 1})
    %cat_19 : [#users=2] = call_function[target=torch.cat](args = ([%getitem_8, %blocks_1_0_layers_3_conv_down, %blocks_1_0_layers_3_conv_normal],), kwargs = {dim: 1})
    %blocks_1_1_layers_0_conv_down : [#users=1] = call_module[target=blocks.1.1.layers.0.conv_down](args = (%cat_16,), kwargs = {})
    %blocks_1_1_layers_0_conv_normal : [#users=1] = call_module[target=blocks.1.1.layers.0.conv_normal](args = (%cat_17,), kwargs = {})
    %blocks_1_1_layers_1_conv_down : [#users=1] = call_module[target=blocks.1.1.layers.1.conv_down](args = (%cat_17,), kwargs = {})
    %blocks_1_1_layers_1_conv_normal : [#users=1] = call_module[target=blocks.1.1.layers.1.conv_normal](args = (%cat_18,), kwargs = {})
    %blocks_1_1_layers_2_conv_down : [#users=1] = call_module[target=blocks.1.1.layers.2.conv_down](args = (%cat_18,), kwargs = {})
    %blocks_1_1_layers_2_conv_normal : [#users=1] = call_module[target=blocks.1.1.layers.2.conv_normal](args = (%cat_19,), kwargs = {})
    %cat_20 : [#users=1] = call_function[target=torch.cat](args = ([%cat_17, %blocks_1_1_layers_0_conv_down, %blocks_1_1_layers_0_conv_normal],), kwargs = {dim: 1})
    %cat_21 : [#users=1] = call_function[target=torch.cat](args = ([%cat_18, %blocks_1_1_layers_1_conv_down, %blocks_1_1_layers_1_conv_normal],), kwargs = {dim: 1})
    %cat_22 : [#users=1] = call_function[target=torch.cat](args = ([%cat_19, %blocks_1_1_layers_2_conv_down, %blocks_1_1_layers_2_conv_normal],), kwargs = {dim: 1})
    %blocks_1_2_m_0_net_0 : [#users=1] = call_module[target=blocks.1.2.m.0.net.0](args = (%cat_20,), kwargs = {})
    %blocks_1_2_m_1_net_0 : [#users=1] = call_module[target=blocks.1.2.m.1.net.0](args = (%cat_21,), kwargs = {})
    %blocks_1_2_m_2_net_0 : [#users=1] = call_module[target=blocks.1.2.m.2.net.0](args = (%cat_22,), kwargs = {})
    %blocks_1_2_m_0_net_1 : [#users=1] = call_module[target=blocks.1.2.m.0.net.1](args = (%blocks_1_2_m_0_net_0,), kwargs = {})
    %blocks_1_2_m_1_net_1 : [#users=1] = call_module[target=blocks.1.2.m.1.net.1](args = (%blocks_1_2_m_1_net_0,), kwargs = {})
    %blocks_1_2_m_2_net_1 : [#users=1] = call_module[target=blocks.1.2.m.2.net.1](args = (%blocks_1_2_m_2_net_0,), kwargs = {})
    %blocks_1_2_m_0_net_2 : [#users=2] = call_module[target=blocks.1.2.m.0.net.2](args = (%blocks_1_2_m_0_net_1,), kwargs = {})
    %blocks_1_2_m_1_net_2 : [#users=3] = call_module[target=blocks.1.2.m.1.net.2](args = (%blocks_1_2_m_1_net_1,), kwargs = {})
    %blocks_1_2_m_2_net_2 : [#users=2] = call_module[target=blocks.1.2.m.2.net.2](args = (%blocks_1_2_m_2_net_1,), kwargs = {})
    %blocks_1_3_layers_1_conv_down : [#users=1] = call_module[target=blocks.1.3.layers.1.conv_down](args = (%blocks_1_2_m_0_net_2,), kwargs = {})
    %blocks_1_3_layers_1_conv_normal : [#users=1] = call_module[target=blocks.1.3.layers.1.conv_normal](args = (%blocks_1_2_m_1_net_2,), kwargs = {})
    %blocks_1_3_layers_2_conv_down : [#users=1] = call_module[target=blocks.1.3.layers.2.conv_down](args = (%blocks_1_2_m_1_net_2,), kwargs = {})
    %blocks_1_3_layers_2_conv_normal : [#users=1] = call_module[target=blocks.1.3.layers.2.conv_normal](args = (%blocks_1_2_m_2_net_2,), kwargs = {})
    %cat_24 : [#users=2] = call_function[target=torch.cat](args = ([%blocks_1_2_m_1_net_2, %blocks_1_3_layers_1_conv_down, %blocks_1_3_layers_1_conv_normal],), kwargs = {dim: 1})
    %cat_25 : [#users=2] = call_function[target=torch.cat](args = ([%blocks_1_2_m_2_net_2, %blocks_1_3_layers_2_conv_down, %blocks_1_3_layers_2_conv_normal],), kwargs = {dim: 1})
    %blocks_1_4_layers_2_conv_down : [#users=1] = call_module[target=blocks.1.4.layers.2.conv_down](args = (%cat_24,), kwargs = {})
    %blocks_1_4_layers_2_conv_normal : [#users=1] = call_module[target=blocks.1.4.layers.2.conv_normal](args = (%cat_25,), kwargs = {})
    %cat_28 : [#users=3] = call_function[target=torch.cat](args = ([%cat_25, %blocks_1_4_layers_2_conv_down, %blocks_1_4_layers_2_conv_normal],), kwargs = {dim: 1})
    %classifier_1_m_0_net_0 : [#users=1] = call_module[target=classifier.1.m.0.net.0](args = (%cat_28,), kwargs = {})
    %zeros_like : [#users=1] = call_function[target=torch.zeros_like](args = (%cat_28,), kwargs = {})
    %classifier_1_m_0_net_1 : [#users=1] = call_module[target=classifier.1.m.0.net.1](args = (%classifier_1_m_0_net_0,), kwargs = {})
    %classifier_1_m_0_net_2 : [#users=1] = call_module[target=classifier.1.m.0.net.2](args = (%classifier_1_m_0_net_1,), kwargs = {})
    %classifier_1_m_1_net_0 : [#users=1] = call_module[target=classifier.1.m.1.net.0](args = (%classifier_1_m_0_net_2,), kwargs = {})
    %classifier_1_m_1_net_1 : [#users=1] = call_module[target=classifier.1.m.1.net.1](args = (%classifier_1_m_1_net_0,), kwargs = {})
    %classifier_1_m_1_net_2 : [#users=1] = call_module[target=classifier.1.m.1.net.2](args = (%classifier_1_m_1_net_1,), kwargs = {})
    %classifier_1_m_2 : [#users=2] = call_module[target=classifier.1.m.2](args = (%classifier_1_m_1_net_2,), kwargs = {})
    %size_1 : [#users=1] = call_method[target=size](args = (%classifier_1_m_2, 0), kwargs = {})
    %view_1 : [#users=1] = call_method[target=view](args = (%classifier_1_m_2, %size_1, 384), kwargs = {})
    %classifier_1_linear : [#users=1] = call_module[target=classifier.1.linear](args = (%view_1,), kwargs = {})
    %scatters_1 : [#users=4] = call_module[target=scatters.1](args = ([%blocks_1_2_m_0_net_2, %cat_24, %cat_28, %zeros_like, %classifier_1_linear], %classifier_1_linear), kwargs = {})
    %getitem_10 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_1, 0), kwargs = {})
    %getitem_11 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_1, 1), kwargs = {})
    %getitem_12 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_1, 2), kwargs = {})
    %getitem_14 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_1, 4), kwargs = {})
    %getitem_15 : [#users=2] = call_function[target=operator.getitem](args = (%getitem_10, 1), kwargs = {})
    %getitem_16 : [#users=2] = call_function[target=operator.getitem](args = (%getitem_11, 1), kwargs = {})
    %getitem_17 : [#users=2] = call_function[target=operator.getitem](args = (%getitem_12, 1), kwargs = {})
    %getitem_18 : [#users=1] = call_function[target=operator.getitem](args = (%getitem_14, 0), kwargs = {})
    %blocks_1_3_layers_0_conv_normal : [#users=1] = call_module[target=blocks.1.3.layers.0.conv_normal](args = (%getitem_15,), kwargs = {})
    %blocks_1_4_layers_1_conv_normal : [#users=1] = call_module[target=blocks.1.4.layers.1.conv_normal](args = (%getitem_16,), kwargs = {})
    %blocks_2_0_layers_2_conv_normal : [#users=1] = call_module[target=blocks.2.0.layers.2.conv_normal](args = (%getitem_17,), kwargs = {})
    %cat_23 : [#users=3] = call_function[target=torch.cat](args = ([%getitem_15, %blocks_1_3_layers_0_conv_normal],), kwargs = {dim: 1})
    %blocks_1_4_layers_0_conv_normal : [#users=1] = call_module[target=blocks.1.4.layers.0.conv_normal](args = (%cat_23,), kwargs = {})
    %blocks_1_4_layers_1_conv_down : [#users=1] = call_module[target=blocks.1.4.layers.1.conv_down](args = (%cat_23,), kwargs = {})
    %cat_26 : [#users=3] = call_function[target=torch.cat](args = ([%cat_23, %blocks_1_4_layers_0_conv_normal],), kwargs = {dim: 1})
    %cat_27 : [#users=3] = call_function[target=torch.cat](args = ([%getitem_16, %blocks_1_4_layers_1_conv_down, %blocks_1_4_layers_1_conv_normal],), kwargs = {dim: 1})
    %blocks_2_0_layers_0_conv_normal : [#users=1] = call_module[target=blocks.2.0.layers.0.conv_normal](args = (%cat_26,), kwargs = {})
    %blocks_2_0_layers_1_conv_down : [#users=1] = call_module[target=blocks.2.0.layers.1.conv_down](args = (%cat_26,), kwargs = {})
    %blocks_2_0_layers_1_conv_normal : [#users=1] = call_module[target=blocks.2.0.layers.1.conv_normal](args = (%cat_27,), kwargs = {})
    %blocks_2_0_layers_2_conv_down : [#users=1] = call_module[target=blocks.2.0.layers.2.conv_down](args = (%cat_27,), kwargs = {})
    %cat_29 : [#users=3] = call_function[target=torch.cat](args = ([%cat_26, %blocks_2_0_layers_0_conv_normal],), kwargs = {dim: 1})
    %cat_30 : [#users=3] = call_function[target=torch.cat](args = ([%cat_27, %blocks_2_0_layers_1_conv_down, %blocks_2_0_layers_1_conv_normal],), kwargs = {dim: 1})
    %cat_31 : [#users=2] = call_function[target=torch.cat](args = ([%getitem_17, %blocks_2_0_layers_2_conv_down, %blocks_2_0_layers_2_conv_normal],), kwargs = {dim: 1})
    %blocks_2_1_layers_0_conv_normal : [#users=1] = call_module[target=blocks.2.1.layers.0.conv_normal](args = (%cat_29,), kwargs = {})
    %blocks_2_1_layers_1_conv_down : [#users=1] = call_module[target=blocks.2.1.layers.1.conv_down](args = (%cat_29,), kwargs = {})
    %blocks_2_1_layers_1_conv_normal : [#users=1] = call_module[target=blocks.2.1.layers.1.conv_normal](args = (%cat_30,), kwargs = {})
    %blocks_2_1_layers_2_conv_down : [#users=1] = call_module[target=blocks.2.1.layers.2.conv_down](args = (%cat_30,), kwargs = {})
    %blocks_2_1_layers_2_conv_normal : [#users=1] = call_module[target=blocks.2.1.layers.2.conv_normal](args = (%cat_31,), kwargs = {})
    %cat_32 : [#users=1] = call_function[target=torch.cat](args = ([%cat_29, %blocks_2_1_layers_0_conv_normal],), kwargs = {dim: 1})
    %cat_33 : [#users=3] = call_function[target=torch.cat](args = ([%cat_30, %blocks_2_1_layers_1_conv_down, %blocks_2_1_layers_1_conv_normal],), kwargs = {dim: 1})
    %cat_34 : [#users=2] = call_function[target=torch.cat](args = ([%cat_31, %blocks_2_1_layers_2_conv_down, %blocks_2_1_layers_2_conv_normal],), kwargs = {dim: 1})
    %blocks_2_2_layers_0_conv_down : [#users=1] = call_module[target=blocks.2.2.layers.0.conv_down](args = (%cat_32,), kwargs = {})
    %blocks_2_2_layers_0_conv_normal : [#users=1] = call_module[target=blocks.2.2.layers.0.conv_normal](args = (%cat_33,), kwargs = {})
    %blocks_2_2_layers_1_conv_down : [#users=1] = call_module[target=blocks.2.2.layers.1.conv_down](args = (%cat_33,), kwargs = {})
    %blocks_2_2_layers_1_conv_normal : [#users=1] = call_module[target=blocks.2.2.layers.1.conv_normal](args = (%cat_34,), kwargs = {})
    %cat_35 : [#users=1] = call_function[target=torch.cat](args = ([%cat_33, %blocks_2_2_layers_0_conv_down, %blocks_2_2_layers_0_conv_normal],), kwargs = {dim: 1})
    %cat_36 : [#users=1] = call_function[target=torch.cat](args = ([%cat_34, %blocks_2_2_layers_1_conv_down, %blocks_2_2_layers_1_conv_normal],), kwargs = {dim: 1})
    %blocks_2_3_m_0_net_0 : [#users=1] = call_module[target=blocks.2.3.m.0.net.0](args = (%cat_35,), kwargs = {})
    %blocks_2_3_m_1_net_0 : [#users=1] = call_module[target=blocks.2.3.m.1.net.0](args = (%cat_36,), kwargs = {})
    %blocks_2_3_m_0_net_1 : [#users=1] = call_module[target=blocks.2.3.m.0.net.1](args = (%blocks_2_3_m_0_net_0,), kwargs = {})
    %blocks_2_3_m_1_net_1 : [#users=1] = call_module[target=blocks.2.3.m.1.net.1](args = (%blocks_2_3_m_1_net_0,), kwargs = {})
    %blocks_2_3_m_0_net_2 : [#users=2] = call_module[target=blocks.2.3.m.0.net.2](args = (%blocks_2_3_m_0_net_1,), kwargs = {})
    %blocks_2_3_m_1_net_2 : [#users=2] = call_module[target=blocks.2.3.m.1.net.2](args = (%blocks_2_3_m_1_net_1,), kwargs = {})
    %blocks_2_4_layers_1_conv_down : [#users=1] = call_module[target=blocks.2.4.layers.1.conv_down](args = (%blocks_2_3_m_0_net_2,), kwargs = {})
    %blocks_2_4_layers_1_conv_normal : [#users=1] = call_module[target=blocks.2.4.layers.1.conv_normal](args = (%blocks_2_3_m_1_net_2,), kwargs = {})
    %cat_38 : [#users=4] = call_function[target=torch.cat](args = ([%blocks_2_3_m_1_net_2, %blocks_2_4_layers_1_conv_down, %blocks_2_4_layers_1_conv_normal],), kwargs = {dim: 1})
    %classifier_2_m_0_net_0 : [#users=1] = call_module[target=classifier.2.m.0.net.0](args = (%cat_38,), kwargs = {})
    %zeros_like_1 : [#users=1] = call_function[target=torch.zeros_like](args = (%cat_38,), kwargs = {})
    %zeros_like_2 : [#users=1] = call_function[target=torch.zeros_like](args = (%cat_38,), kwargs = {})
    %classifier_2_m_0_net_1 : [#users=1] = call_module[target=classifier.2.m.0.net.1](args = (%classifier_2_m_0_net_0,), kwargs = {})
    %classifier_2_m_0_net_2 : [#users=1] = call_module[target=classifier.2.m.0.net.2](args = (%classifier_2_m_0_net_1,), kwargs = {})
    %classifier_2_m_1_net_0 : [#users=1] = call_module[target=classifier.2.m.1.net.0](args = (%classifier_2_m_0_net_2,), kwargs = {})
    %classifier_2_m_1_net_1 : [#users=1] = call_module[target=classifier.2.m.1.net.1](args = (%classifier_2_m_1_net_0,), kwargs = {})
    %classifier_2_m_1_net_2 : [#users=1] = call_module[target=classifier.2.m.1.net.2](args = (%classifier_2_m_1_net_1,), kwargs = {})
    %classifier_2_m_2 : [#users=2] = call_module[target=classifier.2.m.2](args = (%classifier_2_m_1_net_2,), kwargs = {})
    %size_2 : [#users=1] = call_method[target=size](args = (%classifier_2_m_2, 0), kwargs = {})
    %view_2 : [#users=1] = call_method[target=view](args = (%classifier_2_m_2, %size_2, 352), kwargs = {})
    %classifier_2_linear : [#users=1] = call_module[target=classifier.2.linear](args = (%view_2,), kwargs = {})
    %scatters_2 : [#users=3] = call_module[target=scatters.2](args = ([%blocks_2_3_m_0_net_2, %cat_38, %zeros_like_1, %zeros_like_2, %classifier_2_linear], %classifier_2_linear), kwargs = {})
    %getitem_19 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_2, 0), kwargs = {})
    %getitem_20 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_2, 1), kwargs = {})
    %getitem_23 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_2, 4), kwargs = {})
    %getitem_24 : [#users=2] = call_function[target=operator.getitem](args = (%getitem_19, 1), kwargs = {})
    %getitem_25 : [#users=2] = call_function[target=operator.getitem](args = (%getitem_20, 1), kwargs = {})
    %getitem_26 : [#users=1] = call_function[target=operator.getitem](args = (%getitem_23, 0), kwargs = {})
    %blocks_2_4_layers_0_conv_normal : [#users=1] = call_module[target=blocks.2.4.layers.0.conv_normal](args = (%getitem_24,), kwargs = {})
    %blocks_3_0_layers_1_conv_normal : [#users=1] = call_module[target=blocks.3.0.layers.1.conv_normal](args = (%getitem_25,), kwargs = {})
    %cat_37 : [#users=3] = call_function[target=torch.cat](args = ([%getitem_24, %blocks_2_4_layers_0_conv_normal],), kwargs = {dim: 1})
    %blocks_3_0_layers_0_conv_normal : [#users=1] = call_module[target=blocks.3.0.layers.0.conv_normal](args = (%cat_37,), kwargs = {})
    %blocks_3_0_layers_1_conv_down : [#users=1] = call_module[target=blocks.3.0.layers.1.conv_down](args = (%cat_37,), kwargs = {})
    %cat_39 : [#users=3] = call_function[target=torch.cat](args = ([%cat_37, %blocks_3_0_layers_0_conv_normal],), kwargs = {dim: 1})
    %cat_40 : [#users=2] = call_function[target=torch.cat](args = ([%getitem_25, %blocks_3_0_layers_1_conv_down, %blocks_3_0_layers_1_conv_normal],), kwargs = {dim: 1})
    %blocks_3_1_layers_0_conv_normal : [#users=1] = call_module[target=blocks.3.1.layers.0.conv_normal](args = (%cat_39,), kwargs = {})
    %blocks_3_1_layers_1_conv_down : [#users=1] = call_module[target=blocks.3.1.layers.1.conv_down](args = (%cat_39,), kwargs = {})
    %blocks_3_1_layers_1_conv_normal : [#users=1] = call_module[target=blocks.3.1.layers.1.conv_normal](args = (%cat_40,), kwargs = {})
    %cat_41 : [#users=3] = call_function[target=torch.cat](args = ([%cat_39, %blocks_3_1_layers_0_conv_normal],), kwargs = {dim: 1})
    %cat_42 : [#users=2] = call_function[target=torch.cat](args = ([%cat_40, %blocks_3_1_layers_1_conv_down, %blocks_3_1_layers_1_conv_normal],), kwargs = {dim: 1})
    %blocks_3_2_layers_0_conv_normal : [#users=1] = call_module[target=blocks.3.2.layers.0.conv_normal](args = (%cat_41,), kwargs = {})
    %blocks_3_2_layers_1_conv_down : [#users=1] = call_module[target=blocks.3.2.layers.1.conv_down](args = (%cat_41,), kwargs = {})
    %blocks_3_2_layers_1_conv_normal : [#users=1] = call_module[target=blocks.3.2.layers.1.conv_normal](args = (%cat_42,), kwargs = {})
    %cat_43 : [#users=1] = call_function[target=torch.cat](args = ([%cat_41, %blocks_3_2_layers_0_conv_normal],), kwargs = {dim: 1})
    %cat_44 : [#users=2] = call_function[target=torch.cat](args = ([%cat_42, %blocks_3_2_layers_1_conv_down, %blocks_3_2_layers_1_conv_normal],), kwargs = {dim: 1})
    %blocks_3_3_layers_0_conv_down : [#users=1] = call_module[target=blocks.3.3.layers.0.conv_down](args = (%cat_43,), kwargs = {})
    %blocks_3_3_layers_0_conv_normal : [#users=1] = call_module[target=blocks.3.3.layers.0.conv_normal](args = (%cat_44,), kwargs = {})
    %cat_45 : [#users=1] = call_function[target=torch.cat](args = ([%cat_44, %blocks_3_3_layers_0_conv_down, %blocks_3_3_layers_0_conv_normal],), kwargs = {dim: 1})
    %blocks_3_4_m_0_net_0 : [#users=1] = call_module[target=blocks.3.4.m.0.net.0](args = (%cat_45,), kwargs = {})
    %blocks_3_4_m_0_net_1 : [#users=1] = call_module[target=blocks.3.4.m.0.net.1](args = (%blocks_3_4_m_0_net_0,), kwargs = {})
    %blocks_3_4_m_0_net_2 : [#users=5] = call_module[target=blocks.3.4.m.0.net.2](args = (%blocks_3_4_m_0_net_1,), kwargs = {})
    %classifier_3_m_0_net_0 : [#users=1] = call_module[target=classifier.3.m.0.net.0](args = (%blocks_3_4_m_0_net_2,), kwargs = {})
    %zeros_like_3 : [#users=1] = call_function[target=torch.zeros_like](args = (%blocks_3_4_m_0_net_2,), kwargs = {})
    %zeros_like_4 : [#users=1] = call_function[target=torch.zeros_like](args = (%blocks_3_4_m_0_net_2,), kwargs = {})
    %zeros_like_5 : [#users=1] = call_function[target=torch.zeros_like](args = (%blocks_3_4_m_0_net_2,), kwargs = {})
    %classifier_3_m_0_net_1 : [#users=1] = call_module[target=classifier.3.m.0.net.1](args = (%classifier_3_m_0_net_0,), kwargs = {})
    %classifier_3_m_0_net_2 : [#users=1] = call_module[target=classifier.3.m.0.net.2](args = (%classifier_3_m_0_net_1,), kwargs = {})
    %classifier_3_m_1_net_0 : [#users=1] = call_module[target=classifier.3.m.1.net.0](args = (%classifier_3_m_0_net_2,), kwargs = {})
    %classifier_3_m_1_net_1 : [#users=1] = call_module[target=classifier.3.m.1.net.1](args = (%classifier_3_m_1_net_0,), kwargs = {})
    %classifier_3_m_1_net_2 : [#users=1] = call_module[target=classifier.3.m.1.net.2](args = (%classifier_3_m_1_net_1,), kwargs = {})
    %classifier_3_m_2 : [#users=2] = call_module[target=classifier.3.m.2](args = (%classifier_3_m_1_net_2,), kwargs = {})
    %size_3 : [#users=1] = call_method[target=size](args = (%classifier_3_m_2, 0), kwargs = {})
    %view_3 : [#users=1] = call_method[target=view](args = (%classifier_3_m_2, %size_3, 304), kwargs = {})
    %classifier_3_linear : [#users=1] = call_module[target=classifier.3.linear](args = (%view_3,), kwargs = {})
    %scatters_3 : [#users=2] = call_module[target=scatters.3](args = ([%blocks_3_4_m_0_net_2, %zeros_like_3, %zeros_like_4, %zeros_like_5, %classifier_3_linear], %classifier_3_linear), kwargs = {})
    %getitem_27 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_3, 0), kwargs = {})
    %getitem_31 : [#users=1] = call_function[target=operator.getitem](args = (%scatters_3, 4), kwargs = {})
    %getitem_32 : [#users=2] = call_function[target=operator.getitem](args = (%getitem_27, 1), kwargs = {})
    %getitem_33 : [#users=1] = call_function[target=operator.getitem](args = (%getitem_31, 0), kwargs = {})
    %blocks_4_0_layers_0_conv_normal : [#users=1] = call_module[target=blocks.4.0.layers.0.conv_normal](args = (%getitem_32,), kwargs = {})
    %cat_46 : [#users=2] = call_function[target=torch.cat](args = ([%getitem_32, %blocks_4_0_layers_0_conv_normal],), kwargs = {dim: 1})
    %blocks_4_1_layers_0_conv_normal : [#users=1] = call_module[target=blocks.4.1.layers.0.conv_normal](args = (%cat_46,), kwargs = {})
    %cat_47 : [#users=2] = call_function[target=torch.cat](args = ([%cat_46, %blocks_4_1_layers_0_conv_normal],), kwargs = {dim: 1})
    %blocks_4_2_layers_0_conv_normal : [#users=1] = call_module[target=blocks.4.2.layers.0.conv_normal](args = (%cat_47,), kwargs = {})
    %cat_48 : [#users=2] = call_function[target=torch.cat](args = ([%cat_47, %blocks_4_2_layers_0_conv_normal],), kwargs = {dim: 1})
    %blocks_4_3_layers_0_conv_normal : [#users=1] = call_module[target=blocks.4.3.layers.0.conv_normal](args = (%cat_48,), kwargs = {})
    %cat_49 : [#users=1] = call_function[target=torch.cat](args = ([%cat_48, %blocks_4_3_layers_0_conv_normal],), kwargs = {dim: 1})
    %classifier_4_m_0_net_0 : [#users=1] = call_module[target=classifier.4.m.0.net.0](args = (%cat_49,), kwargs = {})
    %classifier_4_m_0_net_1 : [#users=1] = call_module[target=classifier.4.m.0.net.1](args = (%classifier_4_m_0_net_0,), kwargs = {})
    %classifier_4_m_0_net_2 : [#users=1] = call_module[target=classifier.4.m.0.net.2](args = (%classifier_4_m_0_net_1,), kwargs = {})
    %classifier_4_m_1_net_0 : [#users=1] = call_module[target=classifier.4.m.1.net.0](args = (%classifier_4_m_0_net_2,), kwargs = {})
    %classifier_4_m_1_net_1 : [#users=1] = call_module[target=classifier.4.m.1.net.1](args = (%classifier_4_m_1_net_0,), kwargs = {})
    %classifier_4_m_1_net_2 : [#users=1] = call_module[target=classifier.4.m.1.net.2](args = (%classifier_4_m_1_net_1,), kwargs = {})
    %classifier_4_m_2 : [#users=2] = call_module[target=classifier.4.m.2](args = (%classifier_4_m_1_net_2,), kwargs = {})
    %size_4 : [#users=1] = call_method[target=size](args = (%classifier_4_m_2, 0), kwargs = {})
    %view_4 : [#users=1] = call_method[target=view](args = (%classifier_4_m_2, %size_4, 560), kwargs = {})
    %classifier_4_linear : [#users=1] = call_module[target=classifier.4.linear](args = (%view_4,), kwargs = {})
    %final_gather : [#users=1] = call_module[target=final_gather](args = ([%getitem_9, %getitem_18, %getitem_26, %getitem_33, %classifier_4_linear],), kwargs = {})
    return final_gather
