import torch
import torch.nn as nn
import torchvision
import time

import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path[0] = base_dir
from sart.utils.util import save_radius_result, save_number_result
from torch.utils.data import DataLoader
from sart.multipath_bp import BoundedModule, BoundedTensor
from sart.multipath_bp.perturbations import PerturbationLpNorm

from sart.models.cifar_new_10x100.cifar_net_10x100_model import *  

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

BATCH_SIZE = 512

def cifar_robustness_radius():

    model = NeuralNetwork()
    checkpoint = torch.load("../../models/cifar_new_10x100/cifar_net_new_10x100.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)


    ### Step 2: Prepare dataset as usual
    test_data = torchvision.datasets.CIFAR10("../../sources", train=False, download=True, transform=torchvision.transforms.ToTensor())
    n_classes = 10
    # for img_index in [38, 60, 67, 71, 82, 84, 97, 147, 150, 184, 186, 209, 212, 219, 222, 242, 243, 245, 250, 268]:
    # for img_index in [19]:

    if not os.path.isdir('../../result/original_result'):
        os.makedirs('../../result/original_result')
    file = open("../../result/original_result/cifar_new_10x100_alpha_crown_radius_result.txt", mode="w+", encoding="utf-8")

    test_dataloader = DataLoader(test_data, BATCH_SIZE)
    dataiter = iter(test_dataloader)
    images, labels = dataiter.next()
    images = images.reshape(-1, 3072)

    outputs_test = model(images)
    _, predicted = torch.max(outputs_test.data, 1)

    img_index, tmp = 0, 0
    while img_index < BATCH_SIZE and tmp < BATCH_SIZE:
        if predicted[tmp] == labels[tmp]:
            print("img_index:", img_index)
        # for img_index in range(100):
            # Adjust to model input shape!!!
            # print("img_index:", img_index)
            # image = test_data.data[img_index].reshape(1, 3072)
            # image = test_data[img_index][0].reshape(-1, 3072)
            image = images[tmp].reshape(-1, 3072)
            # Convert to float between 0. and 1.
            
            # true_label = test_data.targets[img_index]
            true_label = labels[tmp]
            # true_label = torch.tensor(true_label)

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
                ptb = PerturbationLpNorm(norm=norm, eps=mid/1000)
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
                    lirpa_model.set_bound_opts({'optimize_bound_args': {'ob_iteration': 20, 'ob_lr': 0.1, 'ob_verbose': 0}})

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

            print(f"cifar_property_{img_index} -- delta_base : {delta_base}")
            print(f"cifar_property_{img_index} -- time : {end_time - start_time}")

            img_str = "cifar_property_" + str(img_index)
            save_radius_result(img_str, delta_base, end_time - start_time, file)

            print("img_index：", img_index)
            img_index += 1
            if img_index == 100:
                break
        tmp += 1
        print("tmp: ", tmp)
    # print(tmp)

    file.close()



def test_robustness_number(d):
    model = NeuralNetwork()
    checkpoint = torch.load("../../models/cifar_new_10x100/cifar_net_new_10x100.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    ### Step 2: Prepare dataset as usual
    test_data = torchvision.datasets.CIFAR10("../../sources", train=False, download=True, transform=torchvision.transforms.ToTensor())
    n_classes = 10



    # for img_index in [38, 60, 67, 71, 82, 84, 97, 147, 150, 184, 186, 209, 212, 219, 222, 242, 243, 245, 250, 268]:
    # for img_index in [19]:

    if not os.path.isdir('../../result/original_result'):
        os.makedirs('../../result/original_result')
    file = open("../../result/original_result/cifar_new_10x100_alpha_crown_number_result_delta_"+ str(d) +".txt", mode="w+", encoding="utf-8")

    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
    dataiter = iter(test_dataloader)

    images, labels = dataiter.next()
    images = images.reshape(-1, 3072)

    outputs_test = model(images)
    _, predicted = torch.max(outputs_test.data, 1)

    num_ans = 0
    time_ans = 0

    img_index, tmp = 0, 0
    while img_index < BATCH_SIZE and tmp < BATCH_SIZE :
        if predicted[tmp] == labels[tmp]:
            # print("img_index:", img_index)

            # image = test_data.data[tmp].reshape(1, 784)
            # # # Convert to float between 0. and 1.
            # image = image.to(torch.float32) / 255.0
            # true_label = test_data.targets[tmp]

            # image = images[tmp].reshape(1, 784)
            image = images[tmp].reshape(-1, 3072)
            # Convert to float between 0. and 1.
            
            # true_label = test_data.targets[img_index]
            true_label = labels[tmp]

            # Adjust to model input shape!!!
            # image = test_data.data[img_index].reshape(1, 784)
            # Convert to float between 0. and 1.
            # image = image.to(torch.float32) / 255.0
            # true_label = test_data.targets[img_index]

            # torch.cuda.empty_cache()
            # if torch.cuda.is_available():
            #     image = image.cuda()
            #     model = model.cuda()

            start_time = time.time()

            ### Step 3: wrap model with MultipathBP.
            # The second parameter is for constructing the trace of the computational graph, and its content is not important.
            lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
            # print('Running on', image.device)

            ### Step 4: Compute bounds using LiRPA given a perturbation
            # d = 0.0014
            norm = float("inf")
            ptb = PerturbationLpNorm(norm=norm, eps=d)
            image = BoundedTensor(image, ptb)
            # Get model prediction as usual
            pred = lirpa_model(image)
            label = torch.argmax(pred, dim=1).cpu().detach().numpy()

            # if label.item() != true_label.item():
            #     continue

            C = torch.zeros(size=(1, n_classes - 1, n_classes), device=image.device)
            groundtruth = true_label.to(device=image.device).unsqueeze(0).unsqueeze(1).unsqueeze(2)
            C.scatter_(dim=2, index=groundtruth.repeat(1, n_classes - 1, 1), value=1.0)
            target_labels = torch.arange(1, 10, device=image.device).repeat(1, 1, 1).transpose(1, 2)
            target_labels = (target_labels + groundtruth) % n_classes
            C.scatter_(dim=2, index=target_labels, value=-1.0)
            # print('Computing bounds with a specification matrix:\n', C)

            method = 'CROWN-Optimized'
            if 'Optimized' in method:
                lirpa_model.set_bound_opts({'optimize_bound_args': {'ob_iteration': 20, 'ob_lr': 0.1, 'ob_verbose': 0}})
            lb, ub = lirpa_model.compute_bounds(x=(image,), method=method.split()[0], C=C)
            if torch.min(lb, dim=1)[0][0] >= 0:
                print(f"cifar_property_{img_index} -- Verified")
                num_single = 1
            else:
                print(f"cifar_property_{img_index} -- UnVerified")
                num_single = 0

            end_time = time.time()
            time_single = end_time - start_time

            num_ans += num_single
            time_ans += time_single

            print("time:", time_single)

            img_str = "cifar_property_" + str(img_index)
            save_number_result(img_str, num_single, time_single, file)

            img_index += 1
            
            if img_index == 100:
                break

        tmp += 1

    file.write("delta : " + str(d) + "\n")
    file.write("number_sum : " + str(num_ans) + "\n")
    file.write("time_sum : " + str(time_ans) + "\n")
    file.close()

    print("tmp:", tmp)
    print("delta:", d)
    print("number_sum:", num_ans)
    print("time_sum:", time_ans)






if __name__ == "__main__":

    # cifar_robustness_radius()

    test_robustness_number(0.0005)
    # test_robustness_number(0.0010)
    # test_robustness_number(0.0015)
    # test_robustness_number(0.0020)
    # test_robustness_number(0.0025)
    # test_robustness_number(0.0030)
    # test_robustness_number(0.0035)
    # test_robustness_number(0.0040)
    # test_robustness_number(0.0045)
    # test_robustness_number(0.0050)
