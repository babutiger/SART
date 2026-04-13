import os

import torch
import torchvision
import time

import sys
base_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path[0] = base_dir

from sart.multipath_bp import BoundedModule, BoundedTensor
from sart.multipath_bp.perturbations import PerturbationLpNorm
from sart.utils.util import save_radius_result, save_number_result

from sart.models.mnist_new_5x50.mnist_net_5x50_model import *  # Import the created network model
from torch.utils.data import DataLoader

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    # def write(self, message):
    #     self.terminal.write(message)
    #     self.log.write(message)

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()


    def flush(self):
        pass



# Get the filename of the current script,Remove the file extension from the filename
script_filename = os.path.basename(__file__)
script_name_without_extension = os.path.splitext(script_filename)[0]

style_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
sys.stdout = Logger("../../result/log/"+script_name_without_extension+"_log_" + str(style_time) + ".txt", sys.stdout)
# sys.stderr = Logger("../log/a_err_" + str(style_time) + ".txt", sys.stderr)  # redirect std err, if necessary




BATCH_SIZE = 128


### Step 1: Define computational graph
# Models defined by nn.Sequential

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(784, 80),
#             nn.ReLU(),
#             nn.Linear(80, 80),
#             nn.ReLU(),
#             nn.Linear(80, 80),
#             nn.ReLU(),
#             nn.Linear(80, 80),
#             nn.ReLU(),
#             nn.Linear(80, 80),
#             nn.ReLU(),
#             nn.Linear(80, 80),
#             nn.ReLU(),
#             nn.Linear(80, 80),
#             nn.ReLU(),
#             nn.Linear(80, 80),
#             nn.ReLU(),
#             nn.Linear(80, 80),
#             nn.ReLU(),
#             nn.Linear(80, 80),
#             nn.ReLU(),
#             nn.Linear(80, 10)
#         )
#     def forward(self, x):
#         x = self.linear_relu_stack(x)
#         return x
def mnist_robustness_radius():
    model = NeuralNetwork()
    checkpoint = torch.load("../../models/mnist_new_5x50/mnist_net_new_5x50.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    ### Step 2: Prepare dataset as usual
    test_data = torchvision.datasets.MNIST("../../sources", train=False, download=True, transform=torchvision.transforms.ToTensor())
    n_classes = 10
    # for img_index in [38, 60, 67, 71, 82, 84, 97, 147, 150, 184, 186, 209, 212, 219, 222, 242, 243, 245, 250, 268]:
    # for img_index in [19]:

    if not os.path.isdir('../../result/original_result'):
        os.makedirs('../../result/original_result')
    style_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    file = open("../../result/original_result/mnist_new_5x50_alpha_crown_radius_result_" + str(style_time) + ".txt", mode="w+", encoding="utf-8")

    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
    dataiter = iter(test_dataloader)

    images, labels = dataiter.next()
    images = images.reshape(-1, 784)

    outputs_test = model(images)
    _, predicted = torch.max(outputs_test.data, 1)

    img_index, tmp = 0, 0
    while img_index < BATCH_SIZE and tmp < BATCH_SIZE :
        if predicted[tmp] == labels[tmp]:
            print("img_index:", img_index)

            # image = test_data.data[tmp].reshape(1, 784)
            # # # Convert to float between 0. and 1.
            # image = image.to(torch.float32) / 255.0
            # true_label = test_data.targets[tmp]

            image = images[tmp].reshape(1, 784)
            # Convert to float between 0. and 1.
            # image = image.to(torch.float32) / 255.0   The above has been converted to 0. and 1. Between, this step is repeated, re-turn, the value has changed, so can not want!! Here's the problem
            # true_label = test_data.targets[img_index]
            true_label = labels[tmp]

            # if torch.cuda.is_available():
            #     image = image.cuda()
            #     model = model.cuda()

            start_time = time.time()

            ### Step 3: wrap model with multipath_bp.
            # The second parameter is for constructing the trace of the computational graph, and its content is not important.
            lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
            # print('Running on', image.device)

            ### Step 4: Compute bounds using LiRPA given a perturbation
            left, right = 0, 1000
            delta_base = 0

            while left <= right:
                mid = int((left + right) / 2)
                norm = float("inf")
                ptb = PerturbationLpNorm(norm=norm, eps=mid / 1000)
                image = BoundedTensor(image, ptb)
                # Get model prediction as usual
                pred = lirpa_model(image)
                label = torch.argmax(pred, dim=1).cpu().detach().numpy()

                lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)

                C = torch.zeros(size=(1, n_classes - 1, n_classes), device=image.device)
                groundtruth = true_label.to(device=image.device).unsqueeze(0).unsqueeze(1).unsqueeze(2)
                C.scatter_(dim=2, index=groundtruth.repeat(1, n_classes - 1, 1), value=1.0)
                target_labels = torch.arange(1, 10, device=image.device).repeat(1, 1, 1).transpose(1, 2)
                target_labels = (target_labels + groundtruth) % n_classes
                C.scatter_(dim=2, index=target_labels, value=-1.0)
                # print('Computing bounds with a specification matrix:\n', C)

                method = 'backward'
                method = 'CROWN-Optimized'
                if 'Optimized' in method:
                    lirpa_model.set_bound_opts(
                        {'optimize_bound_args': {'ob_iteration': 20, 'ob_lr': 0.1, 'ob_verbose': 0}})

                lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0], C=C)
                # print("Image {} top-1 prediction {} ground-truth {}".format(i, label[i], true_label[i]))
                # print("lowest margin >= {l:10.5f}".format(l=torch.min(lb, dim=1)[0][i]))
                if torch.min(lb, dim=1)[0][0] >= 0:

                    delta_base = mid / 1000
                    left = mid + 1
                else:
                    right = mid - 1

            end_time = time.time()
            # print(delta_base)
            # print('Verified robust radius: {}, with time cost {}'.format(delta_base, end_time - start_time))

            print(f"mnist_property_{img_index} -- delta_base : {delta_base}")
            print(f"mnist_property_{img_index} -- time : {end_time - start_time}")

            img_str = "mnist_property_" + str(img_index)
            save_radius_result(img_str, delta_base, end_time - start_time, file)

            img_index += 1
            # 100 is enough
            if img_index == 100:
                break
            print("tmp:", tmp)
        tmp += 1
    print(tmp)
    file.close()




def test_robustness_number_acrown(d):
    
    amount = 100

    model = NeuralNetwork()
    checkpoint = torch.load("../../models/mnist_new_5x50/mnist_net_new_5x50.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    ### Step 2: Prepare dataset as usual
    test_data = torchvision.datasets.MNIST("../../sources", train=False, download=True, transform=torchvision.transforms.ToTensor())
    n_classes = 10
    # for img_index in [38, 60, 67, 71, 82, 84, 97, 147, 150, 184, 186, 209, 212, 219, 222, 242, 243, 245, 250, 268]:
    # for img_index in [19]:

    if not os.path.isdir('../../result/original_result'):
        os.makedirs('../../result/original_result')
    style_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    # file = open("../../result/original_result/mnist_new_5x50_alpha_crown_radius_result_" + str(style_time) + ".txt", mode="w+", encoding="utf-8")
    file = open("../../result/original_result/mnist_new_5x50_alpha_crown_number_result_delta_"+ str(d) +"_"+ str(style_time) + ".txt", mode="w+", encoding="utf-8")

    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
    dataiter = iter(test_dataloader)

    images, labels = dataiter.next()
    images = images.reshape(-1, 784)

    outputs_test = model(images)
    _, predicted = torch.max(outputs_test.data, 1)

    img_index, tmp = 0, 0
    number_sum = 0
    time_sum = 0
    time_max = 0
    # num = 0
    while img_index < BATCH_SIZE and tmp < BATCH_SIZE :
        if predicted[tmp] == labels[tmp]:
            print("img_index:", img_index)

            # image = test_data.data[tmp].reshape(1, 784)
            # # # Convert to float between 0. and 1.
            # image = image.to(torch.float32) / 255.0
            # true_label = test_data.targets[tmp]

            image = images[tmp].reshape(1, 784)
            # Convert to float between 0. and 1.
            # image = image.to(torch.float32) / 255.0   The above has been converted to 0. and 1. Between, this step is repeated, re-turn, the value has changed, so can not want!! Here's the problem
            # true_label = test_data.targets[img_index]
            true_label = labels[tmp]

            # if torch.cuda.is_available():
            #     image = image.cuda()
            #     model = model.cuda()

            start_time = time.time()

            ### Step 3: wrap model with multipath_bp.
            # The second parameter is for constructing the trace of the computational graph, and its content is not important.
            lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
            # print('Running on', image.device)

            ### Step 4: Compute bounds using LiRPA given a perturbation
            # left, right = 0, 1000
            delta_base = d

            # while left <= right:
            # mid = int((left + right) / 2)
            norm = float("inf")
            ptb = PerturbationLpNorm(norm=norm, eps=delta_base)
            image = BoundedTensor(image, ptb)
            # Get model prediction as usual
            pred = lirpa_model(image)
            label = torch.argmax(pred, dim=1).cpu().detach().numpy()

            lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)

            C = torch.zeros(size=(1, n_classes - 1, n_classes), device=image.device)
            groundtruth = true_label.to(device=image.device).unsqueeze(0).unsqueeze(1).unsqueeze(2)
            C.scatter_(dim=2, index=groundtruth.repeat(1, n_classes - 1, 1), value=1.0)
            target_labels = torch.arange(1, 10, device=image.device).repeat(1, 1, 1).transpose(1, 2)
            target_labels = (target_labels + groundtruth) % n_classes
            C.scatter_(dim=2, index=target_labels, value=-1.0)
            # print('Computing bounds with a specification matrix:\n', C)

            method = 'backward'
            method = 'CROWN-Optimized'
            if 'Optimized' in method:
                lirpa_model.set_bound_opts(
                    {'optimize_bound_args': {'ob_iteration': 20, 'ob_lr': 0.1, 'ob_verbose': 0}})

            lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0], C=C)
            # print("Image {} top-1 prediction {} ground-truth {}".format(i, label[i], true_label[i]))
            # print("lowest margin >= {l:10.5f}".format(l=torch.min(lb, dim=1)[0][i]))
            num = 0
            if torch.min(lb, dim=1)[0][0] >= 0:
                # delta_base = mid / 1000
                # left = mid + 1
                num = 1
                print(f"mnist_property_{img_index} -- Verified")
            else:
                # right = mid - 1
                num = 0
                print(f"mnist_property_{img_index} -- UnVerified")


            end_time = time.time()
            time_single = end_time - start_time
            # print(delta_base)
            # print('Verified robust radius: {}, with time cost {}'.format(delta_base, end_time - start_time))

            number_sum += num
            time_sum += time_single
            if time_single > time_max:
                time_max = time_single

            # num = 0


            print(f"mnist_property_{img_index} -- delta_base : {delta_base}")
            print(f"mnist_property_{img_index} -- time : {time_single}")

            img_str = "mnist_property_" + str(img_index)
            # save_radius_result(img_str, delta_base, time_single, file)
            save_number_result(img_str, num, time_single, file)


            img_index += 1
            # 100 is enough
            if img_index == amount:
                break
            # print("tmp:", tmp)
        tmp += 1
    # print(tmp)

    file.write("delta : " + str(d) + "\n")
    file.write("number_sum : " + str(number_sum) + "\n")
    file.write("time_sum : " + str(time_sum) + "\n")
    file.write("time_average : " + str(time_sum/amount) + "\n")
    file.write("time_max : " + str(time_max) + "\n")
    file.close()

    print("delta:", d)
    print("number_sum:", number_sum)
    print("time_sum:", time_sum)
    print("time_average:", time_sum/amount)
    print("time_max:", time_max)








if __name__ == "__main__":

    # mnist_robustness_radius()
    test_robustness_number_acrown(0.014)




