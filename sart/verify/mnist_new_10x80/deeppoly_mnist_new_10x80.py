import os
from copy import deepcopy
import numpy as np
import time

import sys
base_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path[0] = base_dir

from sart.utils.util import save_radius_result, save_number_result




# class Logger(object):
#     def __init__(self, filename='default.log', stream=sys.stdout):
#         self.terminal = stream
#         self.log = open(filename, 'a')
#
#     # def write(self, message):
#     #     self.terminal.write(message)
#     #     self.log.write(message)
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.terminal.flush()
#         self.log.write(message)
#         self.log.flush()
#
#     def flush(self):
#         pass
#
# # Get the filename of the current script,Remove the file extension from the filename
# script_filename = os.path.basename(__file__)
# script_name_without_extension = os.path.splitext(script_filename)[0]
#
# style_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
# sys.stdout = Logger("../../result/log/"+script_name_without_extension+"_log_" + str(style_time) + ".txt", sys.stdout)
# # sys.stderr = Logger("../log/a_err_" + str(style_time) + ".txt", sys.stderr)  # redirect std err, if necessary


class neuron(object):
    """
    Attributes:
        algebra_lower (numpy ndarray of float): neuron's algebra lower bound(coeffients of previous neurons and a constant)
        algebra_upper (numpy ndarray of float): neuron's algebra upper bound(coeffients of previous neurons and a constant)
        concrete_algebra_lower (numpy ndarray of float): neuron's algebra lower bound(coeffients of input neurons and a constant)
        concrete_algebra_upper (numpy ndarray of float): neuron's algebra upper bound(coeffients of input neurons and a constant)
        concrete_lower (float): neuron's concrete lower bound
        concrete_upper (float): neuron's concrete upper bound
        weight (numpy ndarray of float): neuron's weight
        bias (float): neuron's bias
        certain_flag (int): 0 uncertain 1 activated(>=0) 2 deactivated(<=0)
    """

    def __init__(self):
        self.algebra_lower = None
        self.algebra_upper = None
        self.concrete_algebra_lower = None
        self.concrete_algebra_upper = None
        self.concrete_lower = None
        self.concrete_upper = None

        self.weight = None
        self.bias = None
        self.certain_flag = 0

    def print(self):
        # print('algebra_lower:', self.algebra_lower)
        # print('algebra_upper:', self.algebra_upper)
        # print('concrete_algebra_lower:', self.concrete_algebra_lower)
        # print('concrete_algebra_upper:', self.concrete_algebra_upper)
        print('concrete_lower:', self.concrete_lower)
        print('concrete_upper:', self.concrete_upper)
        # print('weight:', self.weight)
        # print('bias:', self.bias)
        # print('certain_flag:', self.certain_flag)


class layer(object):
    """
    Attributes:
        neurons (list of neuron): Layer neurons
        size (int): Layer size
        layer_type (int) : Layer type 0 input 1 affine 2 relu
    """
    INPUT_LAYER = 0
    AFFINE_LAYER = 1
    RELU_LAYER = 2

    def __init__(self):
        self.size = None
        self.neurons = None
        self.layer_type = None

    def print(self):
        print('Layer size:', self.size)
        print('Layer type:', self.layer_type)
        print('Neurons:')
        for neu in self.neurons:
            neu.print()
            print('\n')


class network(object):
    """
    Attributes:
        numLayers (int): Number of weight matrices or bias vectors in neural network
        layerSizes (list of ints): Size of input layer, hidden layers, and output layer
        inputSize (int): Size of input
        outputSize (int): Size of output
        mins (list of floats): Minimum values of inputs
        maxes (list of floats): Maximum values of inputs
        means (list of floats): Means of inputs and mean of outputs
        ranges (list of floats): Ranges of inputs and range of outputs
        layers (list of layer): Network Layers
    """

    def __init__(self):
        self.numlayers = None
        self.layerSizes = None
        self.inputSize = None
        self.outputSize = None
        self.mins = None
        self.maxes = None
        self.ranges = None
        self.layers = None
        self.property_flag = None

    # def deeppoly(self):
    #
    #     def back_propagation(cur_neuron, i):
    #         if i == 0:
    #             cur_neuron.concrete_algebra_lower = deepcopy(cur_neuron.algebra_lower)
    #             cur_neuron.concrete_algebra_upper = deepcopy(cur_neuron.algebra_upper)
    #         lower_bound = deepcopy(cur_neuron.algebra_lower)
    #         upper_bound = deepcopy(cur_neuron.algebra_upper)
    
    #             tmp_lower = np.zeros(len(self.layers[k].neurons[0].algebra_lower))
    #             tmp_upper = np.zeros(len(self.layers[k].neurons[0].algebra_lower))
    #             for p in range(self.layers[k].size):
    #                 if lower_bound[p] >= 0:
    #                     tmp_lower += lower_bound[p] * self.layers[k].neurons[p].algebra_lower
    #                     # print("lower_bound[p]:", lower_bound[p])
    #                     # print("self.layers[k].neurons[p].algebra_lower", self.layers[k].neurons[p].algebra_lower)
    #                     # print("self.layers[k].neurons[p].algebra_lower[]", self.layers[k].neurons[p].algebra_lower[0])
    #                     # print("tmp_lower", tmp_lower)
    #                 else:
    #                     tmp_lower += lower_bound[p] * self.layers[k].neurons[p].algebra_upper
    #
    #                 if upper_bound[p] >= 0:
    #                     tmp_upper += upper_bound[p] * self.layers[k].neurons[p].algebra_upper
    #                     # print("upper_bound[p]", upper_bound[p])
    #                     # print("self.layers[k].neurons[p].algebra_upper", self.layers[k].neurons[p].algebra_upper)
    #                     # print("tmp_upper", tmp_upper)
    #                 else:
    #                     tmp_upper += upper_bound[p] * self.layers[k].neurons[p].algebra_lower
    #
    #             # print("===========")
    #             # print("lower_bound:shi",lower_bound)
    #             # print("lower_bound[-1]:", lower_bound[-1])
    #             # print("tmp_lower[-1]:", tmp_lower[-1])
    #             tmp_lower[-1] += lower_bound[-1]
    #             # print("tmp_lower[-1]:", tmp_lower[-1])
    #
    #             # print("=======2")
    #             # print("upper_bound", upper_bound)
    #
    #             # print("upper_bound[-1]:", upper_bound[-1])
    #             # print("tmp_upper[-1]:", tmp_upper[-1])
    #             tmp_upper[-1] += upper_bound[-1]
    #             # print("tmp_upper[-1]:", tmp_upper[-1])
    #
    
    #             # print("lower_bound:", lower_bound)
    #             upper_bound = deepcopy(tmp_upper)
    #             # print("upper_bound：", upper_bound)
    #
    #             if k == 1:
    #                 cur_neuron.concrete_algebra_lower = deepcopy(lower_bound)
    #                 cur_neuron.concrete_algebra_upper = deepcopy(upper_bound)
    #
    #         assert (len(lower_bound) == 1)
    #         assert (len(upper_bound) == 1)
    #         cur_neuron.concrete_lower = lower_bound[0]
    #         cur_neuron.concrete_upper = upper_bound[0]
    #
    #         # print("cur_neuron.concrete_lower:", cur_neuron.concrete_lower)
    #         # print("cur_neuron.concrete_upper:", cur_neuron.concrete_upper)
    #
    #     for i in range(len(self.layers) - 1):
    #         # print('i=',i)
    #         pre_layer = self.layers[i]
    #         cur_layer = self.layers[i + 1]
    #         pre_neuron_list = pre_layer.neurons
    #         cur_neuron_list = cur_layer.neurons
    #
    #         if cur_layer.layer_type == layer.AFFINE_LAYER:
    #             for j in range(cur_layer.size):
    #                 cur_neuron = cur_neuron_list[j]
    #                 cur_neuron.algebra_lower = np.append(cur_neuron.weight, [cur_neuron.bias])
    #                 cur_neuron.algebra_upper = np.append(cur_neuron.weight, [cur_neuron.bias])
    #                 # note: layer index of cur_neuron is i + 1, so pack propagate form i
    #                 back_propagation(cur_neuron, i)
    #
    #         elif cur_layer.layer_type == layer.RELU_LAYER:
    #             for j in range(cur_layer.size):
    #                 cur_neuron = cur_neuron_list[j]
    #                 pre_neuron = pre_neuron_list[j]
    #
    #                 if pre_neuron.concrete_lower >= 0:
    #                     cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
    #                     cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
    #                     cur_neuron.algebra_lower[j] = 1
    #                     cur_neuron.algebra_upper[j] = 1
    #                     cur_neuron.concrete_algebra_lower = deepcopy(pre_neuron.concrete_algebra_lower)
    #                     cur_neuron.concrete_algebra_upper = deepcopy(pre_neuron.concrete_algebra_upper)
    #                     cur_neuron.concrete_lower = pre_neuron.concrete_lower
    #                     cur_neuron.concrete_upper = pre_neuron.concrete_upper
    #                     cur_neuron.certain_flag = 1
    #
    #                 elif pre_neuron.concrete_upper <= 0:
    #                     cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
    #                     cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
    #                     # cur_neuron.algebra_lower[j] = 0
    
    #                     cur_neuron.concrete_algebra_lower = np.zeros(self.inputSize)
    #                     cur_neuron.concrete_algebra_upper = np.zeros(self.inputSize)
    #                     cur_neuron.concrete_lower = 0
    #                     cur_neuron.concrete_upper = 0
    #                     cur_neuron.certain_flag = 2
    #
    #                 elif pre_neuron.concrete_lower + pre_neuron.concrete_upper <= 0:
    #                     # print("---------11112220-------")
    #                     # print("pre_neuron.concrete_lower:", pre_neuron.concrete_lower)
    #                     # print("pre_neuron.concrete_upper:", pre_neuron.concrete_upper)
    
    #
    #                     cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
    #                     cur_neuron.algebra_lower[j] = 0
    #                     # print("cur_neuron.algebra_lower:", cur_neuron.algebra_lower)
    #                     alpha = pre_neuron.concrete_upper / (pre_neuron.concrete_upper - pre_neuron.concrete_lower)
    #                     # print("alpha:", alpha)
    #                     cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
    #                     # print("cur_neuron.algebra_upper,", cur_neuron.algebra_upper)
    #                     cur_neuron.algebra_upper[j] = alpha
    #                     # print("cur_neuron.algebra_upper[j]:", cur_neuron.algebra_upper[j])
    #                     # print("cur_neuron.algebra_upper:", cur_neuron.algebra_upper)
    #                     cur_neuron.algebra_upper[-1] = -alpha * pre_neuron.concrete_lower
    #                     # print("cur_neuron.algebra_upper[-1]:", cur_neuron.algebra_upper[-1])
    #                     # print("cur_neuron.algebra_upper:", cur_neuron.algebra_upper)
    #                     back_propagation(cur_neuron, i)
    #                     # print("---------11112223-------")
    #
    #                 else:
    
    #                     cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
    #                     cur_neuron.algebra_lower[j] = 1
    #                     alpha = pre_neuron.concrete_upper / (pre_neuron.concrete_upper - pre_neuron.concrete_lower)
    #                     cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
    #                     cur_neuron.algebra_upper[j] = alpha
    #                     cur_neuron.algebra_upper[-1] = -alpha * pre_neuron.concrete_lower
    #                     back_propagation(cur_neuron, i)
    #
    
    #                 # else:
    #                 #     cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
    #                 #     cur_neuron.algebra_lower[j] = 0
    #                 #     # print("cur_neuron.algebra_lower:", cur_neuron.algebra_lower)
    #                 #     alpha = pre_neuron.concrete_upper / (pre_neuron.concrete_upper - pre_neuron.concrete_lower)
    #                 #     # print("alpha:", alpha)
    #                 #     cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
    #                 #     # print("cur_neuron.algebra_upper,", cur_neuron.algebra_upper)
    #                 #     cur_neuron.algebra_upper[j] = alpha
    #                 #     # print("cur_neuron.algebra_upper[j]:", cur_neuron.algebra_upper[j])
    #                 #     # print("cur_neuron.algebra_upper:", cur_neuron.algebra_upper)
    #                 #     cur_neuron.algebra_upper[-1] = -alpha * pre_neuron.concrete_lower
    #                 #     # print("cur_neuron.algebra_upper[-1]:", cur_neuron.algebra_upper[-1])
    #                 #     # print("cur_neuron.algebra_upper:", cur_neuron.algebra_upper)
    #                 #     back_propagation(cur_neuron, i)
    #                 #     # print("---------11112223-------")


#     def deeppoly(self):
#
#         def back_propagation(cur_neuron, i):
#             """



#             """

#             if i == 0:
#                 cur_neuron.concrete_algebra_lower = np.array(cur_neuron.algebra_lower, copy=True)
#                 cur_neuron.concrete_algebra_upper = np.array(cur_neuron.algebra_upper, copy=True)
#

#             upper_bound = np.array(cur_neuron.algebra_upper, copy=True)
#

#             for k in range(i, -1, -1):



#




#                 pos = (a >= 0.0)  # (n_k,)
#                 neg = ~pos
#



#                     Lk = Lk[None, :]
#                     Uk = Uk[None, :]
#

#                 tmp_L = (pos.astype(float) * a)[:, None] * Lk
#                 tmp_U = (neg.astype(float) * a)[:, None] * Uk


#

#                 a = upper_bound[:-1]
#                 c = upper_bound[-1]
#                 pos = (a >= 0.0)
#                 neg = ~pos


#                 tmp_upper = tmp_L.sum(axis=0) + tmp_U.sum(axis=0)
#                 tmp_upper[-1] += c
#

#                 lower_bound = tmp_lower
#                 upper_bound = tmp_upper
#

#                 if k == 1:
#                     cur_neuron.concrete_algebra_lower = np.array(lower_bound, copy=True)
#                     cur_neuron.concrete_algebra_upper = np.array(upper_bound, copy=True)
#

#             assert lower_bound.shape[0] == 1 and upper_bound.shape[0] == 1, \
#                 f"Expect scalar after full backprop, got {lower_bound.shape}, {upper_bound.shape}"
#             cur_neuron.concrete_lower = float(lower_bound[0])
#             cur_neuron.concrete_upper = float(upper_bound[0])
#

#         for i in range(len(self.layers) - 1):
#             pre_layer = self.layers[i]
#             cur_layer = self.layers[i + 1]
#             pre_neuron_list = pre_layer.neurons
#             cur_neuron_list = cur_layer.neurons
#
#             if cur_layer.layer_type == layer.AFFINE_LAYER:
#                 for j in range(cur_layer.size):
#                     cur_neuron = cur_neuron_list[j]
#                     cur_neuron.algebra_lower = np.append(cur_neuron.weight, [cur_neuron.bias])
#                     cur_neuron.algebra_upper = np.append(cur_neuron.weight, [cur_neuron.bias])
#                     back_propagation(cur_neuron, i)
#
#             elif cur_layer.layer_type == layer.RELU_LAYER:
#                 for j in range(cur_layer.size):
#                     cur_neuron = cur_neuron_list[j]
#                     pre_neuron = pre_neuron_list[j]
#
#                     if pre_neuron.concrete_lower >= 0:
#                         cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
#                         cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
#                         cur_neuron.algebra_lower[j] = 1
#                         cur_neuron.algebra_upper[j] = 1
#                         cur_neuron.concrete_algebra_lower = np.array(pre_neuron.concrete_algebra_lower, copy=True)
#                         cur_neuron.concrete_algebra_upper = np.array(pre_neuron.concrete_algebra_upper, copy=True)
#                         cur_neuron.concrete_lower = pre_neuron.concrete_lower
#                         cur_neuron.concrete_upper = pre_neuron.concrete_upper
#                         cur_neuron.certain_flag = 1
#
#                     elif pre_neuron.concrete_upper <= 0:
#                         cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
#                         cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
#                         cur_neuron.concrete_algebra_lower = np.zeros(self.inputSize)
#                         cur_neuron.concrete_algebra_upper = np.zeros(self.inputSize)
#                         cur_neuron.concrete_lower = 0.0
#                         cur_neuron.concrete_upper = 0.0
#                         cur_neuron.certain_flag = 2
#
#                     elif pre_neuron.concrete_lower + pre_neuron.concrete_upper <= 0:
#                         cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)

#                         alpha = pre_neuron.concrete_upper / (pre_neuron.concrete_upper - pre_neuron.concrete_lower)
#                         cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
#                         cur_neuron.algebra_upper[j] = alpha
#                         cur_neuron.algebra_upper[-1] = -alpha * pre_neuron.concrete_lower
#                         back_propagation(cur_neuron, i)
#
#                     else:

#                         cur_neuron.algebra_lower = np.zeros(cur_layer.size + 1)
#                         cur_neuron.algebra_lower[j] = 1
#                         alpha = pre_neuron.concrete_upper / (pre_neuron.concrete_upper - pre_neuron.concrete_lower)
#                         cur_neuron.algebra_upper = np.zeros(cur_layer.size + 1)
#                         cur_neuron.algebra_upper[j] = alpha
#                         cur_neuron.algebra_upper[-1] = -alpha * pre_neuron.concrete_lower
#                         back_propagation(cur_neuron, i)
#             else:
#                 raise ValueError("Unknown layer type")

    # def deeppoly(self):
    #     """
    
    
    
    
    #     """
    #
    #     n_in = self.inputSize
    #
    #     # ===== helpers =====
    #     def stack_layer_forms(k):
    
    
    #         """
    #         Lk = np.vstack([neu.algebra_lower for neu in self.layers[k].neurons])
    #         Uk = np.vstack([neu.algebra_upper for neu in self.layers[k].neurons])
    #         return Lk, Uk
    #
    #     def back_propagation_all(A_low_rows, A_up_rows, i):
    #         """
    
    
    #             lows  (m,)  = concrete_lower
    #             ups   (m,)  = concrete_upper
    
    #         """
    
    #         lower_rows = A_low_rows.copy()
    #         upper_rows = A_up_rows.copy()
    #
    #         for k in range(i, -1, -1):
    #             Lk, Uk = stack_layer_forms(k)  # (n_k, n_{k-1}+1)
    #             # --- lower ---
    #             a = lower_rows[:, :-1]  # (m, n_k)
    #             c = lower_rows[:, -1]  # (m,)
    #             a_pos = np.maximum(a, 0.0)
    #             a_neg = np.minimum(a, 0.0)
    #             tmp_lower = a_pos @ Lk + a_neg @ Uk  # (m, n_{k-1}+1)
    #             tmp_lower[:, -1] += c
    #             # --- upper ---
    #             a = upper_rows[:, :-1]
    #             c = upper_rows[:, -1]
    #             a_pos = np.maximum(a, 0.0)
    #             a_neg = np.minimum(a, 0.0)
    
    #             tmp_upper = a_pos @ Uk + a_neg @ Lk
    #             tmp_upper[:, -1] += c
    #
    #             lower_rows = tmp_lower
    #             upper_rows = tmp_upper
    #
    
    #         lows = lower_rows[:, -1].astype(float)
    #         ups = upper_rows[:, -1].astype(float)
    #         return lows, ups
    #
    
    #     for i in range(len(self.layers) - 1):
    #         pre_layer = self.layers[i]
    #         cur_layer = self.layers[i + 1]
    #
    #         if cur_layer.layer_type == layer.AFFINE_LAYER:
    
    #             W = np.vstack([neu.weight for neu in cur_layer.neurons]).astype(float)  # (m, d)
    #             b = np.array([neu.bias for neu in cur_layer.neurons], dtype=float)  # (m,)
    #             A_low_rows = np.hstack([W, b[:, None]])  # (m, d+1)
    #             A_up_rows = A_low_rows.copy()
    #
    
    #             lows, ups = back_propagation_all(A_low_rows, A_up_rows, i)
    #
    
    #             for j, neu in enumerate(cur_layer.neurons):
    
    #                 neu.algebra_lower = A_low_rows[j]
    #                 neu.algebra_upper = A_up_rows[j]
    #                 neu.concrete_lower = float(lows[j])
    #                 neu.concrete_upper = float(ups[j])
    #                 neu.certain_flag = 0
    #
    #         elif cur_layer.layer_type == layer.RELU_LAYER:
    
    #             z_l = np.array([neu.concrete_lower for neu in pre_layer.neurons], dtype=float)
    #             z_u = np.array([neu.concrete_upper for neu in pre_layer.neurons], dtype=float)
    #
    
    #             pos = (z_l >= 0.0)
    #             neg = (z_u <= 0.0)
    #             mix = ~(pos | neg)
    
    
    #
    #             # alpha, beta
    #             denom = (z_u - z_l)
    #             safe = np.where(denom == 0.0, 1.0, denom)
    #             alpha = np.zeros_like(z_l)
    #             beta = np.zeros_like(z_l)
    #             alpha[mix] = z_u[mix] / safe[mix]
    #             beta[mix] = -alpha[mix] * z_l[mix]
    #             alpha[pos] = 1.0;
    #             beta[pos] = 0.0
    #             alpha[neg] = 0.0;
    #             beta[neg] = 0.0
    #
    #             m = cur_layer.size
    #             d = pre_layer.size
    #
    
    
    #             A_low_rows = np.zeros((m, d + 1), dtype=float)
    #             A_up_rows = np.zeros((m, d + 1), dtype=float)
    #
    
    #             idx_keep = np.where(pos | mix_b)[0]
    #             if idx_keep.size > 0:
    
    #
    
    #             A_up_rows[np.arange(m), np.arange(m)] = alpha
    #             A_up_rows[:, -1] = beta
    #
    
    #             lows, ups = back_propagation_all(A_low_rows, A_up_rows, i)
    #
    
    #             for j, neu in enumerate(cur_layer.neurons):
    #                 neu.algebra_lower = A_low_rows[j]
    #                 neu.algebra_upper = A_up_rows[j]
    #                 neu.concrete_lower = float(lows[j])
    #                 neu.concrete_upper = float(ups[j])
    #                 if pos[j]:
    #                     neu.certain_flag = 1
    #                 elif neg[j]:
    #                     neu.certain_flag = 2
    #                 else:
    #                     neu.certain_flag = 0
    #         else:
    #             raise ValueError("Unknown layer type")

    # def deeppoly(self):
    #     """
    
    
    
    
    #     """
    #
    #     # ---------- utils ----------
    #     def clear_all_caches():
    #         for lyr in self.layers:
    #             if hasattr(lyr, "_Lk_cached"):
    #                 delattr(lyr, "_Lk_cached")
    #             if hasattr(lyr, "_Uk_cached"):
    #                 delattr(lyr, "_Uk_cached")
    #
    #     def cache_layer_forms(k):
    
    #         lyr = self.layers[k]
    #         if getattr(lyr, "_Lk_cached", None) is None:
    #             Lk = np.vstack([neu.algebra_lower for neu in lyr.neurons]).astype(np.float64, copy=False)
    #             Uk = np.vstack([neu.algebra_upper for neu in lyr.neurons]).astype(np.float64, copy=False)
    #             lyr._Lk_cached = Lk
    #             lyr._Uk_cached = Uk
    #         return lyr._Lk_cached, lyr._Uk_cached
    #
    #     def back_propagation_all(A_low_rows, A_up_rows, i):
    #         """
    
    
    #         """
    #         lower_rows = A_low_rows.astype(np.float64, copy=True)
    #         upper_rows = A_up_rows.astype(np.float64, copy=True)
    #
    #         for k in range(i, -1, -1):
    #             Lk, Uk = cache_layer_forms(k)  # (n_k, n_{k-1}+1)
    #
    
    #             a = lower_rows[:, :-1]  # (m, n_k)
    #             c = lower_rows[:, -1]  # (m,)
    #             a_pos = np.maximum(a, 0.0)
    #             a_neg = np.minimum(a, 0.0)
    #             tmp_lower = a_pos @ Lk + a_neg @ Uk  # (m, n_{k-1}+1)
    #             tmp_lower[:, -1] += c
    #
    
    #             a2 = upper_rows[:, :-1]
    #             c2 = upper_rows[:, -1]
    #             a2_pos = np.maximum(a2, 0.0)
    #             a2_neg = np.minimum(a2, 0.0)
    #             tmp_upper = a2_pos @ Uk + a2_neg @ Lk
    #             tmp_upper[:, -1] += c2
    #
    #             lower_rows = tmp_lower
    #             upper_rows = tmp_upper
    #
    #         return lower_rows[:, -1].astype(float), upper_rows[:, -1].astype(float)
    #
    #     # ---------- main ----------
    
    #     clear_all_caches()
    #
    #     for i in range(len(self.layers) - 1):
    #         pre_layer = self.layers[i]
    #         cur_layer = self.layers[i + 1]
    #
    #         if cur_layer.layer_type == layer.AFFINE_LAYER:
    
    #             W = np.vstack([neu.weight for neu in cur_layer.neurons]).astype(np.float64)
    #             b = np.array([neu.bias for neu in cur_layer.neurons], dtype=np.float64)
    #             A_low_rows = np.hstack([W, b[:, None]])  # (m, d+1)
    #             A_up_rows = A_low_rows.copy()
    #
    #             lows, ups = back_propagation_all(A_low_rows, A_up_rows, i)
    #
    
    #             for j, neu in enumerate(cur_layer.neurons):
    #                 neu.algebra_lower = A_low_rows[j]
    #                 neu.algebra_upper = A_up_rows[j]
    #                 neu.concrete_lower = float(lows[j])
    #                 neu.concrete_upper = float(ups[j])
    #                 neu.certain_flag = 0
    #             cur_layer._Lk_cached = A_low_rows
    #             cur_layer._Uk_cached = A_up_rows
    #
    #         elif cur_layer.layer_type == layer.RELU_LAYER:
    
    #             z_l = np.array([neu.concrete_lower for neu in pre_layer.neurons], dtype=np.float64)
    #             z_u = np.array([neu.concrete_upper for neu in pre_layer.neurons], dtype=np.float64)
    #
    #             pos = (z_l >= 0.0)
    #             neg = (z_u <= 0.0)
    #             mix = ~(pos | neg)
    
    
    #
    #             denom = (z_u - z_l)
    #             safe = np.where(denom == 0.0, 1.0, denom)
    #             alpha = np.zeros_like(z_l)
    #             beta = np.zeros_like(z_l)
    #             alpha[mix] = z_u[mix] / safe[mix]
    #             beta[mix] = -alpha[mix] * z_l[mix]
    #             alpha[pos] = 1.0;
    #             beta[pos] = 0.0
    #             alpha[neg] = 0.0;
    #             beta[neg] = 0.0
    #
    #             m = cur_layer.size
    #             d = pre_layer.size
    #
    
    #             A_low_rows = np.zeros((m, d + 1), dtype=np.float64)
    #             A_up_rows = np.zeros((m, d + 1), dtype=np.float64)
    #
    
    #             idx_keep = np.where(pos | mix_b)[0]
    #             if idx_keep.size > 0:
    #                 A_low_rows[idx_keep, idx_keep] = 1.0
    #
    
    #             A_up_rows[np.arange(m), np.arange(m)] = alpha
    #             A_up_rows[:, -1] = beta
    #
    #             lows, ups = back_propagation_all(A_low_rows, A_up_rows, i)
    #
    #             for j, neu in enumerate(cur_layer.neurons):
    #                 neu.algebra_lower = A_low_rows[j]
    #                 neu.algebra_upper = A_up_rows[j]
    #                 neu.concrete_lower = float(lows[j])
    #                 neu.concrete_upper = float(ups[j])
    #                 if pos[j]:
    #                     neu.certain_flag = 1
    #                 elif neg[j]:
    #                     neu.certain_flag = 2
    #                 else:
    #                     neu.certain_flag = 0
    #
    
    #             cur_layer._Lk_cached = A_low_rows
    #             cur_layer._Uk_cached = A_up_rows
    #
    #         else:
    #             raise ValueError("Unknown layer type")

    # def deeppoly(self):
    #     """
    
    
    
    
    #     """
    #
    
    
    
    
    
    #
    #     # ---------- utils ----------
    #     def clear_all_caches():
    #         for lyr in self.layers:
    #             if hasattr(lyr, "_Lk_cached"): delattr(lyr, "_Lk_cached")
    #             if hasattr(lyr, "_Uk_cached"): delattr(lyr, "_Uk_cached")
    #
    #     def cache_layer_forms(k):
    
    #         lyr = self.layers[k]
    #         if getattr(lyr, "_Lk_cached", None) is None:
    #             Lk = np.vstack([neu.algebra_lower for neu in lyr.neurons]).astype(np.float32, copy=False)
    #             Uk = np.vstack([neu.algebra_upper for neu in lyr.neurons]).astype(np.float32, copy=False)
    #             lyr._Lk_cached = Lk
    #             lyr._Uk_cached = Uk
    #         return lyr._Lk_cached, lyr._Uk_cached
    #
    #     def back_propagation_all(A_low_rows, A_up_rows, i, dtype=np.float32):
    #         """
    
    #         A_*: (m, n_i+1)
    
    #         """
    #         lower_rows = A_low_rows.astype(dtype, copy=True)
    #         upper_rows = A_up_rows.astype(dtype, copy=True)
    #
    #         for k in range(i, -1, -1):
    
    #             Lk32, Uk32 = cache_layer_forms(k)  # (n_k, n_{k-1}+1) float32
    #             Lk = Lk32.astype(dtype, copy=False)
    #             Uk = Uk32.astype(dtype, copy=False)
    #
    
    #             a = lower_rows[:, :-1]  # (m, n_k)
    #             c = lower_rows[:, -1]  # (m,)
    #             a_pos = np.maximum(a, 0.0)
    #             a_neg = np.minimum(a, 0.0)
    #             tmp_lower = a_pos @ Lk + a_neg @ Uk  # (m, n_{k-1}+1)
    #             tmp_lower[:, -1] += c
    #
    
    #             a2 = upper_rows[:, :-1]
    #             c2 = upper_rows[:, -1]
    #             a2_pos = np.maximum(a2, 0.0)
    #             a2_neg = np.minimum(a2, 0.0)
    #             tmp_upper = a2_pos @ Uk + a2_neg @ Lk
    #             tmp_upper[:, -1] += c2
    #
    #             lower_rows = tmp_lower
    #             upper_rows = tmp_upper
    #
    #         return lower_rows[:, -1].astype(float), upper_rows[:, -1].astype(float)
    #
    
    #     clear_all_caches()
    #
    
    #     for i in range(len(self.layers) - 1):
    #         pre_layer = self.layers[i]
    #         cur_layer = self.layers[i + 1]
    #
    #         if cur_layer.layer_type == layer.AFFINE_LAYER:
    
    #             W = np.vstack([neu.weight for neu in cur_layer.neurons]).astype(np.float32)
    #             b = np.array([neu.bias for neu in cur_layer.neurons], dtype=np.float32)
    #             A_low_rows = np.hstack([W, b[:, None]])  # (m, d+1)
    #             A_up_rows = A_low_rows.copy()
    #
    #             lows, ups = back_propagation_all(A_low_rows, A_up_rows, i, dtype=np.float32)
    #
    #             for j, neu in enumerate(cur_layer.neurons):
    #                 neu.algebra_lower = A_low_rows[j]
    #                 neu.algebra_upper = A_up_rows[j]
    #                 neu.concrete_lower = float(lows[j])
    #                 neu.concrete_upper = float(ups[j])
    #                 neu.certain_flag = 0
    #
    
    #             cur_layer._Lk_cached = A_low_rows
    #             cur_layer._Uk_cached = A_up_rows
    #
    #         elif cur_layer.layer_type == layer.RELU_LAYER:
    
    #             z_l = np.array([neu.concrete_lower for neu in pre_layer.neurons], dtype=np.float32)
    #             z_u = np.array([neu.concrete_upper for neu in pre_layer.neurons], dtype=np.float32)
    #
    
    #             eps_zero = ATOL + RTOL * np.maximum(np.abs(z_l), np.abs(z_u))
    #
    #             pos = (z_l >= eps_zero)
    #             neg = (z_u <= -eps_zero)
    #             mix = ~(pos | neg)
    #
    
    
    #
    #             denom = (z_u - z_l)
    
    #             safe = np.where(np.abs(denom) < DEN_EPS, np.sign(denom) * DEN_EPS, denom)
    #             alpha = np.zeros_like(z_l, dtype=np.float32)
    #             beta = np.zeros_like(z_l, dtype=np.float32)
    #             mask = mix
    #             alpha[mask] = z_u[mask] / safe[mask]
    #             alpha = np.clip(alpha, 0.0, 1.0, out=alpha)
    #             beta[mask] = -alpha[mask] * z_l[mask]
    #
    #             alpha[pos] = 1.0;
    #             beta[pos] = 0.0
    #             alpha[neg] = 0.0;
    #             beta[neg] = 0.0
    #
    #             m = cur_layer.size
    #             d = pre_layer.size
    #
    #             A_low_rows = np.zeros((m, d + 1), dtype=np.float32)
    #             A_up_rows = np.zeros((m, d + 1), dtype=np.float32)
    #
    #             idx_keep = np.where(pos | mix_b)[0]
    #             if idx_keep.size > 0:
    #                 A_low_rows[idx_keep, idx_keep] = 1.0
    #
    #             A_up_rows[np.arange(m), np.arange(m)] = alpha
    #             A_up_rows[:, -1] = beta
    #
    #             lows, ups = back_propagation_all(A_low_rows, A_up_rows, i, dtype=np.float32)
    #
    #             for j, neu in enumerate(cur_layer.neurons):
    #                 neu.algebra_lower = A_low_rows[j]
    #                 neu.algebra_upper = A_up_rows[j]
    #                 neu.concrete_lower = float(lows[j])
    #                 neu.concrete_upper = float(ups[j])
    #                 if pos[j]:
    #                     neu.certain_flag = 1
    #                 elif neg[j]:
    #                     neu.certain_flag = 2
    #                 else:
    #                     neu.certain_flag = 0
    #
    #             cur_layer._Lk_cached = A_low_rows
    #             cur_layer._Uk_cached = A_up_rows
    #
    #         else:
    #             raise ValueError("Unknown layer type")
    #
    
    
    
    #     last_affine_idx = None
    #     for i in range(len(self.layers) - 1):
    #         if self.layers[i + 1].layer_type == layer.AFFINE_LAYER:
    #             last_affine_idx = i
    #
    #     if last_affine_idx is not None:
    
    #         cur_layer = self.layers[last_affine_idx + 1]
    #         W64 = np.vstack([neu.weight for neu in cur_layer.neurons]).astype(np.float64)
    #         b64 = np.array([neu.bias for neu in cur_layer.neurons], dtype=np.float64)
    #         A_low_rows64 = np.hstack([W64, b64[:, None]])
    #         A_up_rows64 = A_low_rows64.copy()
    #
    #         lows64, ups64 = back_propagation_all(A_low_rows64, A_up_rows64, last_affine_idx, dtype=np.float64)
    #
    
    #         need_fix = np.array([abs(neu.concrete_upper) < CHECK_EPS for neu in cur_layer.neurons])
    #         if need_fix.any():
    #             idxs = np.where(need_fix)[0]
    #             for j in idxs:
    #                 cur_layer.neurons[j].concrete_lower = float(lows64[j])
    #                 cur_layer.neurons[j].concrete_upper = float(ups64[j])

    # def deeppoly(self):
    #     """
    
    
    
    
    
    
    #     """
    #
    #     # -------- utils --------
    #     def clear_all_caches():
    #         for lyr in self.layers:
    #             for attr in ("_Lk_cached", "_Uk_cached"):
    #                 if hasattr(lyr, attr):
    #                     delattr(lyr, attr)
    #
    #     def cache_layer_forms(k):
    #         """
    
    
    #         """
    #         lyr = self.layers[k]
    #         if getattr(lyr, "_Lk_cached", None) is None:
    #             Lk = np.vstack([neu.algebra_lower for neu in lyr.neurons]).astype(np.float64, copy=False)
    #             Uk = np.vstack([neu.algebra_upper for neu in lyr.neurons]).astype(np.float64, copy=False)
    #             lyr._Lk_cached = Lk
    #             lyr._Uk_cached = Uk
    #         return lyr._Lk_cached, lyr._Uk_cached
    #
    #     def back_propagation_all(A_low_rows, A_up_rows, i):
    #         """
    
    
    #         """
    #         lower_rows = A_low_rows.astype(np.float64, copy=True)
    #         upper_rows = A_up_rows.astype(np.float64, copy=True)
    #
    #         for k in range(i, -1, -1):
    #             Lk, Uk = cache_layer_forms(k)  # (n_k, n_{k-1}+1), fp64
    #
    
    #             a = lower_rows[:, :-1]  # (m, n_k)
    #             c = lower_rows[:, -1]  # (m,)
    #             a_pos = np.maximum(a, 0.0)
    #             a_neg = np.minimum(a, 0.0)
    #             tmp_lower = a_pos @ Lk + a_neg @ Uk  # (m, n_{k-1}+1)
    #             tmp_lower[:, -1] += c
    #
    
    #             a2 = upper_rows[:, :-1]
    #             c2 = upper_rows[:, -1]
    #             a2_pos = np.maximum(a2, 0.0)
    #             a2_neg = np.minimum(a2, 0.0)
    #             tmp_upper = a2_pos @ Uk + a2_neg @ Lk
    #             tmp_upper[:, -1] += c2
    #
    #             lower_rows = tmp_lower
    #             upper_rows = tmp_upper
    #
    #         return lower_rows[:, -1].astype(float), upper_rows[:, -1].astype(float)
    #
    #     # -------- main --------
    #     clear_all_caches()
    #
    #     for i in range(len(self.layers) - 1):
    #         pre_layer = self.layers[i]
    #         cur_layer = self.layers[i + 1]
    #
    #         if cur_layer.layer_type == layer.AFFINE_LAYER:
    
    #             W = np.vstack([neu.weight for neu in cur_layer.neurons]).astype(np.float64, copy=False)  # (m, d)
    #             b = np.array([neu.bias for neu in cur_layer.neurons], dtype=np.float64)  # (m,)
    #             A_low_rows = np.hstack([W, b[:, None]])  # (m, d+1)
    #             A_up_rows = A_low_rows.copy()
    #
    
    #             lows, ups = back_propagation_all(A_low_rows, A_up_rows, i)
    #
    
    #             for j, neu in enumerate(cur_layer.neurons):
    #                 neu.algebra_lower = A_low_rows[j]
    #                 neu.algebra_upper = A_up_rows[j]
    #                 neu.concrete_lower = float(lows[j])
    #                 neu.concrete_upper = float(ups[j])
    #                 neu.certain_flag = 0
    #
    #             cur_layer._Lk_cached = A_low_rows
    #             cur_layer._Uk_cached = A_up_rows
    #
    #         elif cur_layer.layer_type == layer.RELU_LAYER:
    
    #             z_l = np.array([neu.concrete_lower for neu in pre_layer.neurons], dtype=np.float64)
    #             z_u = np.array([neu.concrete_upper for neu in pre_layer.neurons], dtype=np.float64)
    #
    
    
    
    #             mix = ~(pos | neg)
    
    
    #
    
    #             denom = (z_u - z_l)
    
    #             alpha = np.zeros_like(z_l)
    #             beta = np.zeros_like(z_l)
    #             alpha[mix] = z_u[mix] / safe[mix]
    #             beta[mix] = -alpha[mix] * z_l[mix]
    #             alpha[pos] = 1.0;
    #             beta[pos] = 0.0
    #             alpha[neg] = 0.0;
    #             beta[neg] = 0.0
    #
    #             m = cur_layer.size
    #             d = pre_layer.size
    #
    
    
    #             A_low_rows = np.zeros((m, d + 1), dtype=np.float64)
    #             idx_keep = np.where(pos | mix_b)[0]
    #             if idx_keep.size > 0:
    #                 A_low_rows[idx_keep, idx_keep] = 1.0
    #
    
    #             A_up_rows = np.zeros((m, d + 1), dtype=np.float64)
    #             A_up_rows[np.arange(m), np.arange(m)] = alpha
    #             A_up_rows[:, -1] = beta
    #
    
    #             y_l = np.zeros_like(z_l)
    #             y_u = np.zeros_like(z_u)
    
    #             y_l[pos] = z_l[pos]
    #             y_l[mix_b] = z_l[mix_b]
    
    #             y_u = alpha * z_u + beta
    #
    
    #             for j, neu in enumerate(cur_layer.neurons):
    #                 neu.algebra_lower = A_low_rows[j]
    #                 neu.algebra_upper = A_up_rows[j]
    #                 neu.concrete_lower = float(y_l[j])
    #                 neu.concrete_upper = float(y_u[j])
    #                 if pos[j]:
    #                     neu.certain_flag = 1
    #                 elif neg[j]:
    #                     neu.certain_flag = 2
    #                 else:
    #                     neu.certain_flag = 0
    #
    #             cur_layer._Lk_cached = A_low_rows
    #             cur_layer._Uk_cached = A_up_rows
    #
    #         else:
    #             raise ValueError("Unknown layer type")

    # def deeppoly(self):
    #     """
    
    
    
    
    
    #     """
    #
    
    
    
    
    #
    #     # ---------- utils ----------
    #     def clear_all_caches():
    #         for lyr in self.layers:
    #             for attr in ("_Lk32", "_Uk32", "_Lk64", "_Uk64"):
    #                 if hasattr(lyr, attr):
    #                     delattr(lyr, attr)
    #
    #     def ensure_layer_cache(k):
    #         """
    
    
    #         """
    #         lyr = self.layers[k]
    #         has64 = getattr(lyr, "_Lk64", None) is not None
    #         has32 = getattr(lyr, "_Lk32", None) is not None
    #         if not (has64 and has32):
    
    #             L64 = np.vstack([neu.algebra_lower for neu in lyr.neurons]).astype(np.float64, copy=False)
    #             U64 = np.vstack([neu.algebra_upper for neu in lyr.neurons]).astype(np.float64, copy=False)
    #             L32 = L64.astype(np.float32)
    #             U32 = U64.astype(np.float32)
    #             lyr._Lk64, lyr._Uk64 = L64, U64
    #             lyr._Lk32, lyr._Uk32 = L32, U32
    #         return lyr._Lk32, lyr._Uk32, lyr._Lk64, lyr._Uk64
    #
    #     def back_propagation_all(A_low_rows, A_up_rows, i, use64=False):
    #         """
    
    
    #         """
    #         dt = np.float64 if use64 else np.float32
    #         rows_l = np.asarray(A_low_rows, dtype=dt)
    #         rows_u = np.asarray(A_up_rows, dtype=dt)
    #
    #         for k in range(i, -1, -1):
    #             L32, U32, L64, U64 = ensure_layer_cache(k)
    #             Lk = L64 if use64 else L32
    #             Uk = U64 if use64 else U32
    #
    
    #             a = rows_l[:, :-1];
    #             c = rows_l[:, -1]
    #             a_pos = np.maximum(a, 0.0);
    #             a_neg = np.minimum(a, 0.0)
    #             tmp_l = a_pos @ Lk + a_neg @ Uk
    #             tmp_l[:, -1] += c
    #
    
    #             a2 = rows_u[:, :-1];
    #             c2 = rows_u[:, -1]
    #             a2_pos = np.maximum(a2, 0.0);
    #             a2_neg = np.minimum(a2, 0.0)
    #             tmp_u = a2_pos @ Uk + a2_neg @ Lk
    #             tmp_u[:, -1] += c2
    #
    #             rows_l, rows_u = tmp_l, tmp_u
    #
    #         return rows_l[:, -1].astype(np.float64), rows_u[:, -1].astype(np.float64)
    #
    #     def near_zero_mask(l, u):
    
    #         eps0 = ATOL + RTOL * np.maximum(np.abs(l), np.abs(u))
    #         return (np.abs(l) < eps0) | (np.abs(u) < eps0)
    #
    #     # ---------- main ----------
    #     clear_all_caches()
    #
    #     for i in range(len(self.layers) - 1):
    #         pre_layer = self.layers[i]
    #         cur_layer = self.layers[i + 1]
    #
    #         if cur_layer.layer_type == layer.AFFINE_LAYER:
    
    #             W64 = np.vstack([neu.weight for neu in cur_layer.neurons]).astype(np.float64, copy=False)  # (m,d)
    #             b64 = np.array([neu.bias for neu in cur_layer.neurons], dtype=np.float64)  # (m,)
    #             A64 = np.hstack([W64, b64[:, None]])  # (m,d+1)
    #             A32 = A64.astype(np.float32)
    #
    
    #             l32, u32 = back_propagation_all(A32, A32, i, use64=False)
    #
    
    #             crit = near_zero_mask(l32, u32) | (np.abs(u32 - l32) < DEN_EPS)
    #             if np.any(crit):
    #                 l64c, u64c = back_propagation_all(A64[crit], A64[crit], i, use64=True)
    #                 z_l = l32.astype(np.float64);
    #                 z_u = u32.astype(np.float64)
    #                 z_l[crit] = l64c;
    #                 z_u[crit] = u64c
    #             else:
    #                 z_l, z_u = l32.astype(np.float64), u32.astype(np.float64)
    #
    
    #             for j, neu in enumerate(cur_layer.neurons):
    #                 neu.algebra_lower = A64[j]
    #                 neu.algebra_upper = A64[j]
    #                 neu.concrete_lower = float(z_l[j])
    #                 neu.concrete_upper = float(z_u[j])
    #                 neu.certain_flag = 0
    #
    
    #             cur_layer._Lk64 = A64;
    #             cur_layer._Uk64 = A64
    #             cur_layer._Lk32 = A32;
    #             cur_layer._Uk32 = A32
    #
    #         elif cur_layer.layer_type == layer.RELU_LAYER:
    
    #             z_l = np.array([neu.concrete_lower for neu in pre_layer.neurons], dtype=np.float64)
    #             z_u = np.array([neu.concrete_upper for neu in pre_layer.neurons], dtype=np.float64)
    #
    
    #             eps0 = ATOL + RTOL * np.maximum(np.abs(z_l), np.abs(z_u))
    #             pos = (z_l >= eps0)
    #             neg = (z_u <= -eps0)
    #             mix = ~(pos | neg)
    
    
    #
    
    #             denom = z_u - z_l
    #             safe = np.where(np.abs(denom) < DEN_EPS, np.sign(denom) * DEN_EPS, denom)
    #             alpha = np.zeros_like(z_l)
    #             beta = np.zeros_like(z_l)
    #             mask = mix
    #             alpha[mask] = z_u[mask] / safe[mask]
    #             alpha = np.clip(alpha, 0.0, 1.0, out=alpha)
    #             beta[mask] = -alpha[mask] * z_l[mask]
    #             alpha[pos] = 1.0;
    #             beta[pos] = 0.0
    #             alpha[neg] = 0.0;
    #             beta[neg] = 0.0
    #
    #             m = cur_layer.size
    #             d = pre_layer.size
    #
    
    #             A64_low = np.zeros((m, d + 1), dtype=np.float64)
    #             idx_keep = np.where(pos | mix_b)[0]
    #             if idx_keep.size > 0:
    #                 A64_low[idx_keep, idx_keep] = 1.0
    #
    #             A64_up = np.zeros((m, d + 1), dtype=np.float64)
    #             A64_up[np.arange(m), np.arange(m)] = alpha
    #             A64_up[:, -1] = beta
    #
    #             A32_low = A64_low.astype(np.float32)
    #             A32_up = A64_up.astype(np.float32)
    #
    
    #             y_l = np.zeros_like(z_l)
    #             y_l[pos] = z_l[pos]
    #             y_l[mix_b] = z_l[mix_b]
    #             y_u = alpha * z_u + beta
    #
    
    #             for j, neu in enumerate(cur_layer.neurons):
    #                 neu.algebra_lower = A64_low[j]
    #                 neu.algebra_upper = A64_up[j]
    #                 neu.concrete_lower = float(y_l[j])
    #                 neu.concrete_upper = float(y_u[j])
    #                 if pos[j]:
    #                     neu.certain_flag = 1
    #                 elif neg[j]:
    #                     neu.certain_flag = 2
    #                 else:
    #                     neu.certain_flag = 0
    #
    
    #             cur_layer._Lk64, cur_layer._Uk64 = A64_low, A64_up
    #             cur_layer._Lk32, cur_layer._Uk32 = A32_low, A32_up
    #
    #         else:
    #             raise ValueError("Unknown layer type")


    # def deeppoly(self, use_float32: bool = False):
    #     """
    
    
    
    
    
    #     """
    
    #     DT = np.float32 if use_float32 else np.float64
    
    #     RTOL = 1e-6 if use_float32 else 0.0
    
    #
    
    #     def clear_all_caches():
    #         for lyr in self.layers:
    #             for attr in ("_LkF", "_UkF"):
    #                 if hasattr(lyr, attr):
    #                     delattr(lyr, attr)
    #
    #     def cache_layer_forms_F(k: int):
    #         """
    
    
    #         """
    #         lyr = self.layers[k]
    #         if getattr(lyr, "_LkF", None) is None:
    #             Lk = np.vstack([neu.algebra_lower for neu in lyr.neurons]).astype(DT, copy=False, order='C')
    #             Uk = np.vstack([neu.algebra_upper for neu in lyr.neurons]).astype(DT, copy=False, order='C')
    
    #             lyr._LkF = np.asfortranarray(Lk)
    #             lyr._UkF = np.asfortranarray(Uk)
    #         return lyr._LkF, lyr._UkF
    #
    #     def back_propagation_all_np(A_low_rows: np.ndarray, A_up_rows: np.ndarray, i: int):
    #         """
    
    
    
    #         """
    #         rows_l = np.ascontiguousarray(A_low_rows, dtype=DT)
    #         rows_u = np.ascontiguousarray(A_up_rows, dtype=DT)
    #
    #         for k in range(i, -1, -1):
    
    #
    
    #             a = rows_l[:, :-1]  # (m, n_k)
    #             c = rows_l[:, -1]  # (m,)
    
    #             a_pos = np.empty_like(a)
    #             a_neg = np.empty_like(a)
    #             np.clip(a, 0, None, out=a_pos)  # max(a, 0)
    #             np.minimum(a, 0, out=a_neg)  # min(a, 0)
    #             tmp_l = a_pos @ LkF + a_neg @ UkF  # (m, n_{k-1}+1)
    #             tmp_l[:, -1] += c
    #
    
    #             a2 = rows_u[:, :-1]
    #             c2 = rows_u[:, -1]
    #             a2_pos = np.empty_like(a2)
    #             a2_neg = np.empty_like(a2)
    #             np.clip(a2, 0, None, out=a2_pos)
    #             np.minimum(a2, 0, out=a2_neg)
    #             tmp_u = a2_pos @ UkF + a2_neg @ LkF
    #             tmp_u[:, -1] += c2
    #
    
    #
    
    #         return rows_l[:, -1].astype(np.float64), rows_u[:, -1].astype(np.float64)
    #
    
    #     clear_all_caches()
    #
    #     for i in range(len(self.layers) - 1):
    #         pre_layer = self.layers[i]
    #         cur_layer = self.layers[i + 1]
    #
    #         if cur_layer.layer_type == layer.AFFINE_LAYER:
    
    #             W = np.vstack([neu.weight for neu in cur_layer.neurons]).astype(DT, copy=False, order='C')  # (m,d)
    #             b = np.array([neu.bias for neu in cur_layer.neurons], dtype=DT)  # (m,)
    #             A_rows = np.ascontiguousarray(np.hstack([W, b[:, None]]), dtype=DT)  # (m,d+1)
    #
    #             lows, ups = back_propagation_all_np(A_rows, A_rows, i)
    #
    
    #             for j, neu in enumerate(cur_layer.neurons):
    
    #                 neu.algebra_upper = A_rows[j].astype(np.float64, copy=False)
    #                 neu.concrete_lower = float(lows[j])
    #                 neu.concrete_upper = float(ups[j])
    #                 neu.certain_flag = 0
    #
    
    #             cur_layer._LkF = np.asfortranarray(A_rows.astype(DT, copy=False))
    
    #
    #         elif cur_layer.layer_type == layer.RELU_LAYER:
    
    #             z_l = np.array([neu.concrete_lower for neu in pre_layer.neurons], dtype=np.float64)
    #             z_u = np.array([neu.concrete_upper for neu in pre_layer.neurons], dtype=np.float64)
    #
    
    #             eps0 = ATOL + RTOL * np.maximum(np.abs(z_l), np.abs(z_u))
    #             pos = (z_l >= eps0)
    #             neg = (z_u <= -eps0)
    #             mix = ~(pos | neg)
    
    #
    
    #             denom = z_u - z_l
    #             denom_safe = np.where(np.abs(denom) < max(DEN_EPS, 0.0), np.sign(denom) * max(DEN_EPS, 1e-30), denom)
    #             alpha = np.zeros_like(z_l)
    #             beta = np.zeros_like(z_l)
    #             sel = mix
    #             alpha[sel] = z_u[sel] / denom_safe[sel]
    #             if use_float32:
    
    #                 np.clip(alpha, 0.0, 1.0, out=alpha)
    #             beta[sel] = -alpha[sel] * z_l[sel]
    #             alpha[pos] = 1.0;
    #             beta[pos] = 0.0
    #             alpha[neg] = 0.0;
    #             beta[neg] = 0.0
    #
    #             m = cur_layer.size
    #             d = pre_layer.size
    #
    
    #             A_low = np.zeros((m, d + 1), dtype=DT, order='C')
    #             idx_keep = np.where(pos | mix_b)[0]
    #             if idx_keep.size > 0:
    #                 A_low[idx_keep, idx_keep] = 1.0
    #             A_up = np.zeros((m, d + 1), dtype=DT, order='C')
    #             A_up[np.arange(m), np.arange(m)] = alpha.astype(DT, copy=False)
    #             A_up[:, -1] = beta.astype(DT, copy=False)
    #
    
    #             y_l = np.zeros_like(z_l)
    #             y_l[pos] = z_l[pos]
    #             y_l[mix_b] = z_l[mix_b]
    #             y_u = alpha * z_u + beta
    #
    
    #             for j, neu in enumerate(cur_layer.neurons):
    #                 neu.algebra_lower = A_low[j].astype(np.float64, copy=False)
    #                 neu.algebra_upper = A_up[j].astype(np.float64, copy=False)
    #                 neu.concrete_lower = float(y_l[j])
    #                 neu.concrete_upper = float(y_u[j])
    #                 if pos[j]:
    #                     neu.certain_flag = 1
    #                 elif neg[j]:
    #                     neu.certain_flag = 2
    #                 else:
    #                     neu.certain_flag = 0
    #
    
    #             cur_layer._LkF = np.asfortranarray(A_low)
    #             cur_layer._UkF = np.asfortranarray(A_up)
    #
    #         else:
    #             raise ValueError("Unknown layer type")

    
    def deeppoly(self, tight_pre_overrides=None, override_mode="override"):
        """
        Run the batched fp64 DeepPoly pass with closed-form ReLU handling and
        vectorized pre-activation bound overrides.
          - tight_pre_overrides: dict[int -> array_like(m, 2)] keyed by the
            affine-layer index inside ``self.layers``
          - override_mode: ``"override"`` or ``"intersect"``
        """

        # -------- utils --------
        def clear_all_caches():
            for lyr in self.layers:
                for attr in ("_Lk_cached", "_Uk_cached"):
                    if hasattr(lyr, attr):
                        delattr(lyr, attr)

        def cache_layer_forms(k):
            """
            Cache the fp64 stacked ``algebra_*`` matrices for layer ``k`` as
            the previous-layer forms.
            Shape: ``Lk``, ``Uk`` ~ ``(n_k, n_{k-1} + 1)``.
            """
            lyr = self.layers[k]
            if getattr(lyr, "_Lk_cached", None) is None:
                Lk = np.vstack([neu.algebra_lower for neu in lyr.neurons]).astype(np.float64, copy=False)
                Uk = np.vstack([neu.algebra_upper for neu in lyr.neurons]).astype(np.float64, copy=False)
                lyr._Lk_cached = Lk
                lyr._Uk_cached = Uk
            return lyr._Lk_cached, lyr._Uk_cached

        def back_propagation_all(A_low_rows, A_up_rows, i):
            """
            Back-propagate the full layer of lower and upper linear forms to
            the input in one pass.
            ``A_*`` has shape ``(m, n_i + 1)`` and returns ``lows``/``ups``
            with shape ``(m,)``.
            """
            rows_l = A_low_rows.astype(np.float64, copy=True)
            rows_u = A_up_rows.astype(np.float64, copy=True)

            for k in range(i, -1, -1):
                Lk, Uk = cache_layer_forms(k)  # (n_k, n_{k-1}+1), fp64

                
                a = rows_l[:, :-1]  # (m, n_k)
                c = rows_l[:, -1]  # (m,)
                a_pos = np.maximum(a, 0.0)
                a_neg = np.minimum(a, 0.0)
                tmp_lower = a_pos @ Lk + a_neg @ Uk  # (m, n_{k-1}+1)
                tmp_lower[:, -1] += c

                
                a2 = rows_u[:, :-1]
                c2 = rows_u[:, -1]
                a2_pos = np.maximum(a2, 0.0)
                a2_neg = np.minimum(a2, 0.0)
                tmp_upper = a2_pos @ Uk + a2_neg @ Lk
                tmp_upper[:, -1] += c2

                rows_l, rows_u = tmp_lower, tmp_upper

            return rows_l[:, -1].astype(float), rows_u[:, -1].astype(float)

        def apply_overrides_vec(z_l, z_u, pairs, mode):
            """
            Apply ``pairs`` with shape ``(m, 2)`` to ``z_l``/``z_u`` in a
            vectorized way and return the updated ``(z_l', z_u')`` pair.
            - ``mode == "override"`` replaces the bounds with ``pairs``
            - ``mode == "intersect"`` intersects the existing bounds
            The helper also guarantees ``lo <= hi`` and swaps endpoints when
            required.
            """
            P = np.asarray(pairs, dtype=np.float64)
            if P.ndim != 2 or P.shape[1] != 2 or P.shape[0] != z_l.shape[0]:
                raise AssertionError("tight_pre_overrides[idx] must be array-like with shape (m,2)")

            lo, hi = P[:, 0], P[:, 1]
            
            lo2 = np.minimum(lo, hi)
            hi2 = np.maximum(lo, hi)

            if mode == "intersect":
                z_l_new = np.maximum(z_l, lo2)
                z_u_new = np.minimum(z_u, hi2)
            else:  # "override"
                z_l_new = lo2.copy()
                z_u_new = hi2.copy()

            
            bad = z_l_new > z_u_new
            if np.any(bad):
                
                mid = 0.5 * (z_l_new[bad] + z_u_new[bad])
                z_l_new[bad] = mid
                z_u_new[bad] = mid

            return z_l_new, z_u_new

        # -------- main --------
        clear_all_caches()
        tight_pre_overrides = tight_pre_overrides or {}

        for i in range(len(self.layers) - 1):
            pre_layer = self.layers[i]
            cur_layer = self.layers[i + 1]

            if cur_layer.layer_type == layer.AFFINE_LAYER:
                
                W = np.vstack([neu.weight for neu in cur_layer.neurons]).astype(np.float64, copy=False)  # (m, d)
                b = np.array([neu.bias for neu in cur_layer.neurons], dtype=np.float64)  # (m,)
                A_rows = np.hstack([W, b[:, None]])  # (m, d+1)
                
                z_l, z_u = back_propagation_all(A_rows, A_rows, i)

                
                if i in tight_pre_overrides:
                    z_l, z_u = apply_overrides_vec(z_l, z_u, tight_pre_overrides[i], override_mode)
                    

                
                for j, neu in enumerate(cur_layer.neurons):
                    neu.algebra_lower = A_rows[j]
                    neu.algebra_upper = A_rows[j]
                    neu.concrete_lower = float(z_l[j])
                    neu.concrete_upper = float(z_u[j])
                    neu.certain_flag = 0

                cur_layer._Lk_cached = A_rows
                cur_layer._Uk_cached = A_rows

            elif cur_layer.layer_type == layer.RELU_LAYER:
                
                z_l = np.array([neu.concrete_lower for neu in pre_layer.neurons], dtype=np.float64)
                z_u = np.array([neu.concrete_upper for neu in pre_layer.neurons], dtype=np.float64)

                
                pos = (z_l >= 0.0)  
                neg = (z_u <= 0.0)  
                mix = ~(pos | neg)
                mix_b = mix & ((z_l + z_u) > 0.0)  

                
                denom = (z_u - z_l)
                safe = np.where(denom == 0.0, 1.0, denom)  
                alpha = np.zeros_like(z_l)
                beta = np.zeros_like(z_l)
                sel = mix
                alpha[sel] = z_u[sel] / safe[sel]
                beta[sel] = -alpha[sel] * z_l[sel]
                alpha[pos] = 1.0;
                beta[pos] = 0.0
                alpha[neg] = 0.0;
                beta[neg] = 0.0

                m = cur_layer.size
                d = pre_layer.size

                
                
                A_low_rows = np.zeros((m, d + 1), dtype=np.float64)
                idx_keep = np.where(pos | mix_b)[0]
                if idx_keep.size > 0:
                    A_low_rows[idx_keep, idx_keep] = 1.0

                
                A_up_rows = np.zeros((m, d + 1), dtype=np.float64)
                A_up_rows[np.arange(m), np.arange(m)] = alpha
                A_up_rows[:, -1] = beta

                
                y_l = np.zeros_like(z_l)
                y_l[pos] = z_l[pos]
                y_l[mix_b] = z_l[mix_b]
                y_u = alpha * z_u + beta

                
                for j, neu in enumerate(cur_layer.neurons):
                    neu.algebra_lower = A_low_rows[j]
                    neu.algebra_upper = A_up_rows[j]
                    neu.concrete_lower = float(y_l[j])
                    neu.concrete_upper = float(y_u[j])
                    if pos[j]:
                        neu.certain_flag = 1
                    elif neg[j]:
                        neu.certain_flag = 2
                    else:
                        neu.certain_flag = 0

                cur_layer._Lk_cached = A_low_rows
                cur_layer._Uk_cached = A_up_rows

            else:
                raise ValueError("Unknown layer type")

    def print(self):
        print('numlayers:', self.numLayers)
        print('layerSizes:', self.layerSizes)
        print("inputSize:", self.inputSize)
        print('outputSize:', self.outputSize)
        print('mins:', self.mins)
        print('maxes:', self.maxes)
        print('ranges:', self.ranges)
        print('Layers:')
        for l in self.layers:
            l.print()
            print('\n')

    def load_robustness(self, filename, delta, TRIM=False):
        if self.property_flag == True:
            self.layers.pop()
        self.property_flag = True
        with open(filename) as f:
            for i in range(self.layerSizes[0]):
                line = f.readline()
                linedata = [float(line.strip()) - delta, float(line.strip()) + delta]
                if TRIM:
                    if linedata[0] < 0: linedata[0] = 0
                    if linedata[1] > 1: linedata[1] = 1

                self.layers[0].neurons[i].concrete_lower = linedata[0]
                self.layers[0].neurons[i].concrete_upper = linedata[1]
                self.layers[0].neurons[i].concrete_algebra_lower = np.array([linedata[0]])
                self.layers[0].neurons[i].concrete_algebra_upper = np.array([linedata[1]])
                self.layers[0].neurons[i].algebra_lower = np.array([linedata[0]])
                self.layers[0].neurons[i].algebra_upper = np.array([linedata[1]])

            line = f.readline()
            verify_layer = layer()
            verify_layer.neurons = []
            while line:
                linedata = [float(x) for x in line.strip().split(' ')]
                assert (len(linedata) == self.outputSize + 1)
                verify_neuron = neuron()
                verify_neuron.weight = np.array(linedata[:-1])
                verify_neuron.bias = linedata[-1]
                verify_layer.neurons.append(verify_neuron)
                linedata = np.array(linedata)
                # print('linedata：', linedata)
                assert (len(linedata) == self.outputSize + 1)
                line = f.readline()
            verify_layer.size = len(verify_layer.neurons)
            verify_layer.layer_type = layer.AFFINE_LAYER
            if len(verify_layer.neurons) > 0:
                self.layers.append(verify_layer)

    def load_nnet(self, filename):
        with open(filename) as f:
            line = f.readline()
            cnt = 1
            while line[0:2] == "//":
                line = f.readline()
                cnt += 1
            # numLayers doesn't include the input layer!
            numLayers, inputSize, outputSize, _ = [int(x) for x in line.strip().split(",")[:-1]]
            line = f.readline()

            # input layer size, layer1size, layer2size...
            layerSizes = [int(x) for x in line.strip().split(",")[:-1]]

            line = f.readline()
            symmetric = int(line.strip().split(",")[0])
            line = f.readline()
            inputMinimums = [float(x) for x in line.strip().split(",")[:-1]]
            line = f.readline()
            inputMaximums = [float(x) for x in line.strip().split(",")[:-1]]
            line = f.readline()
            inputMeans = [float(x) for x in line.strip().split(",")[:-1]]
            line = f.readline()
            inputRanges = [float(x) for x in line.strip().split(",")[:-1]]

            # process the input layer
            self.layers = []
            new_layer = layer()
            new_layer.layer_type = layer.INPUT_LAYER
            new_layer.size = layerSizes[0]
            new_layer.neurons = []
            for i in range(layerSizes[0]):
                new_neuron = neuron()
                new_layer.neurons.append(new_neuron)
            self.layers.append(new_layer)

            for layernum in range(numLayers):
                previousLayerSize = layerSizes[layernum]
                currentLayerSize = layerSizes[layernum + 1]
                new_layer = layer()
                new_layer.size = currentLayerSize
                new_layer.layer_type = layer.AFFINE_LAYER
                new_layer.neurons = []

                # weights
                for i in range(currentLayerSize):
                    line = f.readline()
                    new_neuron = neuron()
                    weight = [float(x) for x in line.strip().split(",")[:-1]]
                    assert (len(weight) == previousLayerSize)
                    new_neuron.weight = np.array(weight)
                    new_layer.neurons.append(new_neuron)
                # biases
                for i in range(currentLayerSize):
                    line = f.readline()
                    x = float(line.strip().split(",")[0])
                    new_layer.neurons[i].bias = x

                self.layers.append(new_layer)

                # add relu layer
                if layernum + 1 == numLayers:
                    break
                new_layer = layer()
                new_layer.size = currentLayerSize
                new_layer.layer_type = layer.RELU_LAYER
                new_layer.neurons = []
                for i in range(currentLayerSize):
                    new_neuron = neuron()
                    new_layer.neurons.append(new_neuron)
                self.layers.append(new_layer)

            self.numLayers = numLayers
            self.layerSizes = layerSizes
            self.inputSize = inputSize
            self.outputSize = outputSize
            self.mins = inputMinimums
            self.maxes = inputMaximums
            self.means = inputMeans
            self.ranges = inputRanges

    def find_max_disturbance(self, PROPERTY, L=0, R=1000, TRIM=False):
        ans = 0
        while L <= R:
            # print(L,R)
            mid = int((L + R) / 2)
            self.load_robustness(PROPERTY, mid / 1000, TRIM=TRIM)
            self.deeppoly()
            flag = True
            for neuron_i in self.layers[-1].neurons:  
                # print("neuron_i.concrete_upper：", neuron_i.concrete_upper)
                if neuron_i.concrete_upper > 0:  
                    flag = False  
            if flag == True:  
                ans = mid / 1000
                L = mid + 1
            else:  
                R = mid - 1
        return ans

    def find_robustness_number(self, PROPERTY, t, TRIM=False):
        self.load_robustness(PROPERTY, delta=t, TRIM=TRIM)
        self.deeppoly()
        flag = True
        for neuron_i in self.layers[-1].neurons:  
            print("neuron_i.concrete_lower：", neuron_i.concrete_lower)
            print("neuron_i.concrete_upper：", neuron_i.concrete_upper)
            print("------------")
            if neuron_i.concrete_upper > 0:  
                flag = False  
        if flag == True:  
            # print('Verified')
            num = 1
        else:  
            # print('Unverified')
            num = 0
        return num



    def find_robustness_number_test(self, PROPERTY, t, TRIM=False):
        self.load_robustness(PROPERTY, delta=t, TRIM=TRIM)
        self.deeppoly()
        flag = True

        for neuron_i in self.layers[-1].neurons:  
            # print("neuron_i.concrete_lower：", neuron_i.concrete_lower)
            # print("neuron_i.concrete_upper：", neuron_i.concrete_upper)
            # print("------------")
            if neuron_i.concrete_upper > 0:  
                flag = False  

        save_deeppoly = []

        if flag == True:  
            # print('Verified')
            num = 1

            len_layerSizes = (len(self.layerSizes)-1)*2
            # print(len_layerSizes)

            # save_deeppoly = []

            for t in range(1, len_layerSizes+1):  
                # print(f"t:{t}")
                save_deeppoly_layer = []
                for neuron_i in self.layers[t].neurons:  
                    temp = []
                    # print("neuron_i.concrete_lower：", neuron_i.concrete_lower)
                    # print("neuron_i.concrete_upper：", neuron_i.concrete_upper)
                    temp.append(neuron_i.concrete_lower)
                    temp.append(neuron_i.concrete_upper)
                    save_deeppoly_layer.append(temp)
                    # print(f"save_deeppoly_layer:{save_deeppoly_layer}")
                save_deeppoly.append(save_deeppoly_layer)

        else:  
            # print('Unverified')
            num = 0

            # print(self.layerSizes)

            len_layerSizes = (len(self.layerSizes)-1)*2
            # print(len_layerSizes)

            # save_deeppoly = []

            for t in range(1, len_layerSizes+1):  
                # print(f"t:{t}")
                save_deeppoly_layer = []
                for neuron_i in self.layers[t].neurons:  
                    temp = []
                    # print("neuron_i.concrete_lower：", neuron_i.concrete_lower)
                    # print("neuron_i.concrete_upper：", neuron_i.concrete_upper)
                    temp.append(neuron_i.concrete_lower)
                    temp.append(neuron_i.concrete_upper)
                    save_deeppoly_layer.append(temp)
                    # print(f"save_deeppoly_layer:{save_deeppoly_layer}")
                save_deeppoly.append(save_deeppoly_layer)
                # print(f"save_deeppoly:{save_deeppoly}")

        return num, save_deeppoly



def mnist_robustness_radius():
    net = network()
    net.load_nnet("../../models/mnist_new_10x80/mnist_net_new_10x80.nnet")
    property_list = ["../../mnist_properties/mnist_properties_10x80/mnist_property_" + str(i) + ".txt" for i in range(100)]


    if not os.path.isdir('../../result/original_result'):
        os.makedirs('../../result/original_result')
    file = open("../../result/original_result/mnist_new_10x80_deeppoly_radius_result.txt", mode="w+", encoding="utf-8")

    for property_i in property_list:
        start_time = time.time()
        delta_base = net.find_max_disturbance(PROPERTY=property_i, TRIM=True)
        end_time = time.time()
        property_i = property_i[46:]
        property_i = property_i[:-4]
        # c_time = end_time - start_time
        print(f"{property_i} -- delta_base : {delta_base}")
        print(f"{property_i} -- time : {end_time - start_time}")

        save_radius_result(property_i, delta_base, end_time - start_time, file)
    file.close()



def acas_robustness_radius():
    for recur in range(1, 11):
        print("Recur:", recur)
        net = network()
        net.load_nnet("nnet/ACASXU_experimental_v2a_2_3.nnet")
        property_list = ["acas_properties/local_robustness_" + str(i) + ".txt" for i in range(2, 3)]
        for property_i in property_list:
            star_time = time.time()
            delta_base = net.find_max_disturbance(PROPERTY=property_i, TRIM=False)
            end_time = time.time()
            property_i = property_i[16:]
            property_i = property_i[:-4]
            print(f"{property_i} -- delta_base : {delta_base}")
            print(f"{property_i} -- time : {end_time - star_time}")


def cifar_robustness_radius():
    net = network()
    net.load_nnet("models/cifar_net.nnet")
    property_list = ["cifar_properties/cifar_properties_10x100/cifar_property_" + str(i) + ".txt" for i in range(50)]
    for property_i in property_list:
        star_time = time.time()
        delta_base = net.find_max_disturbance(PROPERTY=property_i, TRIM=True)
        end_time = time.time()
        property_i = property_i[41:-4]
        # property_i = property_i[:-4]
        print(f"{property_i} -- delta_base : {delta_base}")
        print(f"{property_i} -- time : {end_time - star_time}")



def test_example():
    net = network()
    net.load_nnet('paper_example/abstracmp_paper_illustration.nnet')
    net.load_robustness('paper_example/abstracmp_paper_illustration.txt', 1)
    net.deeppoly()
    net.print()


def test_acas():
    net = network()
    net.load_nnet('../../models/mnist_new_10x80/mnist_net_new_10x80.nnet')
    net.load_robustness('../../mnist_properties/mnist_new_10x80/mnist_property_19.txt', 0.001, TRIM=True)
    # print("-------111-----")
    net.deeppoly()
    # print("-------222-----")
    net.print()

    start_time = time.time()
    delta_base = net.find_max_disturbance('../../mnist_properties/mnist_property_19.txt', TRIM=True)
    end_time = time.time()
    print("delta_base:")
    print(delta_base)
    print(end_time - start_time)


def test_robustness_number0(t):
    net = network()
    net.load_nnet('../../models/mnist_new_10x80/mnist_net_new_10x80.nnet')
    property_list = ["../../mnist_properties/mnist_properties_10x80/mnist_property_" + str(i) + ".txt" for i in
                     range(100)]

    num = 0
    for property_i in property_list:
        net.load_robustness(property_i, delta = t, TRIM=True)
        start_time = time.time()
        net.deeppoly()
        flag = True
        for neuron_i in net.layers[-1].neurons:  
            # print("neuron_i.concrete_upper：", neuron_i.concrete_upper)
            if neuron_i.concrete_upper > 0:  
                flag = False  
        if flag == True:  
            print('Verified')
            num += 1
        else:  
            print('Unverified')

        end_time = time.time()
        print("time:", end_time - start_time)

    print("num:", num)


def test_robustness_number(d):
    net = network()
    net.load_nnet("../../models/mnist_new_10x80/mnist_net_new_10x80.nnet")
    property_list = ["../../mnist_properties/mnist_properties_10x80/mnist_property_" + str(i) + ".txt" for i in
                     range(0, 1)]

    if not os.path.isdir('../../result/original_result'):
        os.makedirs('../../result/original_result')
    file = open("../../result/original_result/mnist_new_10x80_deeppoly_number_result_delta_"+ str(d) +".txt", mode="w+", encoding="utf-8")

    num_ans = 0
    time_ans = 0
    for property_i in property_list:
        start_time = time.time()
        num_single = net.find_robustness_number(property_i, d, TRIM=True)
        property_i = property_i[46:]
        property_i = property_i[:-4]
        if num_single == 1:
            print(f"{property_i} -- Verified")
        else:
            print(f"{property_i} -- UnVerified")

        num_ans += num_single
        end_time = time.time()
        time_single = end_time - start_time
        print("time:", time_single)
        time_ans += time_single

        save_number_result(property_i, num_single, time_single, file)
    file.write("delta : " + str(d) + "\n")
    file.write("number_sum : " + str(num_ans) + "\n")
    file.write("time_sum : " + str(time_ans) + "\n")


    file.close()
    print("delta:", d)
    print("number_sum:", num_ans)
    print("time_sum:", time_ans)




def test_robustness_number_test(d):
    net = network()
    net.load_nnet("../../models/mnist_new_10x80/mnist_net_new_10x80.nnet")
    property_list = ["../../mnist_properties/mnist_properties_10x80/mnist_property_" + str(i) + ".txt" for i in
                     range(0, 100)]

    if not os.path.isdir('../../result/original_result'):
        os.makedirs('../../result/original_result')
    file = open("../../result/original_result/mnist_new_10x80_deeppoly_number_result_delta_"+ str(d) +".txt", mode="w+", encoding="utf-8")

    num_ans = 0
    time_ans = 0
    for property_i in property_list:
        start_time = time.time()
        num_single, save_deeppoly = net.find_robustness_number_test(property_i, d, TRIM=True)
        property_i = property_i[46:]
        property_i = property_i[:-4]
        if num_single == 1:
            print(f"{property_i} -- Verified")
        else:
            print(f"{property_i} -- UnVerified")


        num_ans += num_single
        end_time = time.time()
        time_single = end_time - start_time
        print("time:", time_single)
        time_ans += time_single

        save_number_result(property_i, num_single, time_single, file)
    file.write("delta : " + str(d) + "\n")
    file.write("number_sum : " + str(num_ans) + "\n")
    file.write("time_sum : " + str(time_ans) + "\n")


    file.close()
    print("delta:", d)
    print("number_sum:", num_ans)
    print("time_sum:", time_ans)



def test_robustness_number_test_return(d):
    net = network()
    net.load_nnet("../../models/mnist_new_10x80/mnist_net_new_10x80.nnet")
    property_list = ["../../mnist_properties/mnist_properties_10x80/mnist_property_" + str(i) + ".txt" for i in
                     range(0, 1)]

    if not os.path.isdir('../../result/original_result'):
        os.makedirs('../../result/original_result')
    file = open("../../result/original_result/mnist_new_10x80_deeppoly_number_result_delta_"+ str(d) +".txt", mode="w+", encoding="utf-8")

    num_ans = 0
    time_ans = 0
    for property_i in property_list:
        start_time = time.time()
        num_single = net.find_robustness_number_test(property_i, d, TRIM=True)
        property_i = property_i[46:]
        property_i = property_i[:-4]
        if num_single == 1:
            print(f"{property_i} -- Verified")
        else:
            print(f"{property_i} -- UnVerified")


        num_ans += num_single
        end_time = time.time()
        time_single = end_time - start_time
        print("time:", time_single)
        time_ans += time_single

        save_number_result(property_i, num_single, time_single, file)
    file.write("delta : " + str(d) + "\n")
    file.write("number_sum : " + str(num_ans) + "\n")
    file.write("time_sum : " + str(time_ans) + "\n")


    file.close()
    print("delta:", d)
    print("number_sum:", num_ans)
    print("time_sum:", time_ans)


if __name__ == "__main__":
    # test_acas()
    # test_example()
    # mnist_robustness_radius()
    # cifar_robustness_radius()
    # acas_robustness_radius()

    # test_robustness_number(0.015)
    test_robustness_number_test(0.015)
    # test_robustness_number(0.004)
    # test_robustness_number(0.006)
    # test_robustness_number(0.008)
    # test_robustness_number(0.010)
    # test_robustness_number(0.012)
    # test_robustness_number(0.014)
    # test_robustness_number(0.016)
    # test_robustness_number(0.018)
    # test_robustness_number(0.020)
    # test_robustness_number(0.022)
    # test_robustness_number(0.024)
    # test_robustness_number(0.026)
    # test_robustness_number(0.028)
    # test_robustness_number(0.030)
    # test_robustness_number(0.032)
    # test_robustness_number(0.034)
    # test_robustness_number(0.036)
    # test_robustness_number(0.038)
    # test_robustness_number(0.040)
    #
