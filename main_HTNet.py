from os import path
import os
import sys
import importlib

import numpy as np
import cv2
import time
from datetime import timedelta

import pandas
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse

# migration advice https://peps.python.org/pep-0632/#migration-advice
# from distutils.util import strtobool
import torch
from Model import *
import numpy as np
from facenet_pytorch import MTCNN


CSV_PATH = './label/combined_3_class2_for_optical_flow.csv'
BASE_DATA_DIR = './datasets/combined_datasets_whole'
WHOLE_OPTICAL_FLOW_PATH = './datasets/STSNet_whole_norm_u_v_os'
MAIN_PATH = './datasets/three_norm_u_v_os'
WEIGHT_DIR = 'weights'

# copy from setuptools _distutil strtobool
def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError(f"invalid truth value {val!r}")

# Some of the codes are adapted from STSNet
def reset_weights(m):  # Reset the weights for network to avoid weight leakage
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def confusionMatrix(gt, pred, show=False):
    TN, FP, FN, TP = confusion_matrix(gt, pred, labels=[0,1]).ravel()
    # TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples
    return f1_score, average_recall

def recognition_evaluation(final_gt, final_pred, show=False):
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}
    # Display recognition result
    f1_list = []
    ar_list = []
    
    for emotion, emotion_index in label_dict.items():
        gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
        pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
        f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
        f1_list.append(f1_recog)
        ar_list.append(ar_recog)
    UF1 = np.mean(f1_list)
    UAR = np.mean(ar_list)
    return UF1, UAR

# def recognition_evaluation(final_gt, final_pred, show=False):
#     label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}
#     # Display recognition result
#     f1_list = []
#     ar_list = []
#     try:
#         for emotion, emotion_index in label_dict.items():
#             gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
#             pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
#             try:
#                 f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
#                 f1_list.append(f1_recog)
#                 ar_list.append(ar_recog)
#             except Exception as e:
#                 print("recognition evaluation error!")
#                 sys.exit()
#         UF1 = np.mean(f1_list)
#         UAR = np.mean(ar_list)
#         return UF1, UAR
#     except:
#         return '', ''

# 1. get the whole face block coordinates
def whole_face_block_coordinates():
    df = pandas.read_csv(CSV_PATH)
    m, n = df.shape
    base_data_src = BASE_DATA_DIR
    total_emotion = 0
    image_size_u_v = 28
    # get the block center coordinates
    face_block_coordinates = {}

    # for i in range(0, m):
    for i in range(0, m):
        image_name = str(df['sub'][i]) + '_' + str(df['filename_o'][i]) + ' .png'
        # print(image_name)
        img_path_apex = base_data_src + '/' + df['imagename'][i]
        train_face_image_apex = cv2.imread(img_path_apex) # (444, 533, 3)
        face_apex = cv2.resize(train_face_image_apex, (28,28), interpolation=cv2.INTER_AREA)
        # get face and bounding box
        # 参数有设置CUDA device
        mtcnn = MTCNN(margin=0, image_size=image_size_u_v, select_largest=True, post_process=False, device='cuda:0')
        batch_boxes, _, batch_landmarks = mtcnn.detect(face_apex, landmarks=True)
        # print(img_path_apex,batch_landmarks)
        # if not detecting face
        if batch_landmarks is None:
            # print( df['imagename'][i])
            batch_landmarks = np.array([[[9.528073, 11.062551]
                                            , [21.396168, 10.919773]
                                            , [15.380184, 17.380562]
                                            , [10.255435, 22.121233]
                                            , [20.583706, 22.25584]]])
            # print(img_path_apex)
        row_n, col_n = np.shape(batch_landmarks[0])
        # print(batch_landmarks[0])
        for i in range(0, row_n):
            for j in range(0, col_n):
                if batch_landmarks[0][i][j] < 7:
                    batch_landmarks[0][i][j] = 7
                if batch_landmarks[0][i][j] > 21:
                    batch_landmarks[0][i][j] = 21
        batch_landmarks = batch_landmarks.astype(int)
        # print(batch_landmarks[0])
        # get the block center coordinates
        face_block_coordinates[image_name] = batch_landmarks[0]
    # print(len(face_block_coordinates))
    return face_block_coordinates

# 2. crop the 28*28-> 14*14 according to i5 image centers
def crop_optical_flow_block():
    face_block_coordinates_dict = whole_face_block_coordinates()
    # Get train dataset
    whole_optical_flow_path = WHOLE_OPTICAL_FLOW_PATH
    whole_optical_flow_imgs = os.listdir(whole_optical_flow_path)
    four_parts_optical_flow_imgs = {}

    for n_img in whole_optical_flow_imgs:
        four_parts_optical_flow_imgs[n_img]=[]
        flow_image = cv2.imread(whole_optical_flow_path + '/' + n_img)
        four_part_coordinates = face_block_coordinates_dict[n_img]
        l_eye = flow_image[four_part_coordinates[0][0]-7:four_part_coordinates[0][0]+7,
                four_part_coordinates[0][1]-7: four_part_coordinates[0][1]+7]
        l_lips = flow_image[four_part_coordinates[1][0] - 7:four_part_coordinates[1][0] + 7,
                four_part_coordinates[1][1] - 7: four_part_coordinates[1][1] + 7]
        nose = flow_image[four_part_coordinates[2][0] - 7:four_part_coordinates[2][0] + 7,
                four_part_coordinates[2][1] - 7: four_part_coordinates[2][1] + 7]
        r_eye = flow_image[four_part_coordinates[3][0] - 7:four_part_coordinates[3][0] + 7,
                four_part_coordinates[3][1] - 7: four_part_coordinates[3][1] + 7]
        r_lips = flow_image[four_part_coordinates[4][0] - 7:four_part_coordinates[4][0] + 7,
                four_part_coordinates[4][1] - 7: four_part_coordinates[4][1] + 7]
        four_parts_optical_flow_imgs[n_img].append(l_eye)
        four_parts_optical_flow_imgs[n_img].append(l_lips)
        four_parts_optical_flow_imgs[n_img].append(nose)
        four_parts_optical_flow_imgs[n_img].append(r_eye)
        four_parts_optical_flow_imgs[n_img].append(r_lips)
        # print(np.shape(l_eye))
    # print((four_parts_optical_flow_imgs['spNO.189_f_150.png'][0]))->(14,14,3)
    print(len(four_parts_optical_flow_imgs))
    return four_parts_optical_flow_imgs

# never used
class Fusionmodel(nn.Module):
  def __init__(self):
    #  extend from original
    super(Fusionmodel,self).__init__()
    self.fc1 = nn.Linear(15, 3) # 15->3
    self.bn1 = nn.BatchNorm1d(3)
    self.d1 = nn.Dropout(p=0.5)
    # Linear 256 to 26
    self.fc_2 = nn.Linear(6, 2) # 6->3
    # self.fc_cont = nn.Linear(256, 3)
    self.relu = nn.ReLU()

    # forward layers is to use these layers above
  def forward(self, whole_feature, l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature):
    fuse_five_features = torch.cat((l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature), 0)
    # nn.linear - fc
    fuse_out = self.fc1(fuse_five_features)
    # fuse_out = self.bn1(fuse_out)
    fuse_out = self.relu(fuse_out)
    fuse_out = self.d1(fuse_out) # drop out
    fuse_whole_five_parts = torch.cat(
        (whole_feature,fuse_out), 0)
    # fuse_whole_five_parts = self.bn1(fuse_whole_five_parts)
    fuse_whole_five_parts = self.relu(fuse_whole_five_parts)
    fuse_whole_five_parts = self.d1(fuse_whole_five_parts)  # drop out
    out = self.fc_2(fuse_whole_five_parts)
    return out

def main(config):

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    loss_fn = nn.CrossEntropyLoss()
    if (config.train):
        if not path.exists(WEIGHT_DIR):
            os.mkdir(WEIGHT_DIR)
    
    # print(f"Models: {config.model_type}\n")
    # print('lr=%f, epochs=%d, device=%s\n' % (config.learning_rate, config.epochs, device))
    print("Configuration Parameters:")
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 20)

    all_accuracy_dict = {}
    Module = importlib.import_module("Model")
    model_type = getattr(Module, config.model_type)
    total_gt = []
    total_pred = []
    best_total_pred = []
    # model_name_printed = False

    t = time.time()

    main_path = MAIN_PATH
    subName = os.listdir(main_path)
    all_five_parts_optical_flow = crop_optical_flow_block()
    print(subName)
    ## LOSO，共68个
    for n_subName in subName:
        print('Subject:', n_subName)
        y_train = []
        y_test = []
        four_parts_train = []
        four_parts_test = []
        # Get train dataset
        expression = os.listdir(main_path + '/' + n_subName + '/u_train')
        for n_expression in expression:
            img = os.listdir(main_path + '/' + n_subName + '/u_train/' + n_expression)

            for n_img in img:
                y_train.append(int(n_expression))
                l_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][0], all_five_parts_optical_flow[n_img][1]])
                r_eye_lips  =  cv2.hconcat([all_five_parts_optical_flow[n_img][3], all_five_parts_optical_flow[n_img][4]])
                lr_eye_lips = cv2.vconcat([l_eye_lips, r_eye_lips])
                four_parts_train.append(lr_eye_lips)


        # Get test dataset
        expression = os.listdir(main_path + '/' + n_subName + '/u_test')
        for n_expression in expression:
            img = os.listdir(main_path + '/' + n_subName + '/u_test/' + n_expression)

            for n_img in img:
                y_test.append(int(n_expression))
                l_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][0], all_five_parts_optical_flow[n_img][1]])
                r_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][3], all_five_parts_optical_flow[n_img][4]])
                lr_eye_lips = cv2.vconcat([l_eye_lips, r_eye_lips])
                four_parts_test.append(lr_eye_lips)
        weight_path = WEIGHT_DIR + '/' + n_subName + '.pth'

        # Reset or load model weigts
        paramDict = vars(config)
        import inspect
        model_signature = inspect.signature(model_type.__init__)
        valid_params = model_signature.parameters.keys()
        filtered_paramDict = {k: v for k, v in paramDict.items() if k in valid_params}

        model = model_type(**filtered_paramDict)
        # model = model_type(
        #     image_size=config.image_size,
        #     patch_size=config.patch_size,
        #     dim=config.dim,  # 256,--96, 56-, 192
        #     heads=config.heads,  # 3 ---- , 6-
        #     num_hierarchies=config.num_hierarchies,  # 3----number of hierarchies
        #     block_repeats=config.block_repeats,#(2, 2, 8),------# the number of transformer blocks at each heirarchy, starting from the bottom(2,2,10) -
        #     num_classes=config.num_classes,
        #     gb_tf_channels = config.gb_tf_channels,
        #     gb_heads = config.gb_heads,
        #     gb_n_windows = config.gb_n_windows
        # )
        
        # if not model_name_printed:
        #     print("="*20)
        #     print(f"Model: {model.__class__.__name__}")
        #     print("="*20)
        #     model_name_printed = True
        
        model = model.to(device)

        if(config.train):
            # model.apply(reset_weights)
            print('train')
        else:
            model.load_state_dict(torch.load(weight_path))
        optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
        y_train = torch.Tensor(y_train).to(dtype=torch.long)
        four_parts_train =  torch.Tensor(np.array(four_parts_train)).permute(0, 3, 1, 2)
        dataset_train = TensorDataset(four_parts_train, y_train)
        train_dl = DataLoader(dataset_train, batch_size=config.batch_size)
        
        y_test = torch.Tensor(y_test).to(dtype=torch.long)
        four_parts_test = torch.Tensor(np.array(four_parts_test)).permute(0, 3, 1, 2)
        dataset_test = TensorDataset(four_parts_test, y_test)
        test_dl = DataLoader(dataset_test, batch_size=config.batch_size)
        # store best results
        best_accuracy_for_each_subject = 0
        best_each_subject_pred = []

        time_one_sub = time.time()
        for epoch in range(1, config.epochs + 1):
            if (config.train):
                # Training
                model.train()
                train_loss = 0.0
                num_train_correct = 0
                num_train_examples = 0

                for batch in train_dl:
                    optimizer.zero_grad()
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    yhat = model(x)
                    loss = loss_fn(yhat, y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.data.item() * x.size(0)
                    num_train_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                    num_train_examples += x.shape[0]

                train_acc = num_train_correct / num_train_examples
                train_loss = train_loss / len(train_dl.dataset)

            # Testing
            model.eval()
            val_loss = 0.0
            num_val_correct = 0
            num_val_examples = 0
            for batch in test_dl:
                x = batch[0].to(device)
                y = batch[1].to(device)
                yhat = model(x)
                loss = loss_fn(yhat, y)
                val_loss += loss.data.item() * x.size(0)
                num_val_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                num_val_examples += y.shape[0]

            val_acc = num_val_correct / num_val_examples
            val_loss = val_loss / len(test_dl.dataset)
            #### best result
            temp_best_each_subject_pred = []
            if best_accuracy_for_each_subject <= val_acc:
                best_accuracy_for_each_subject = val_acc
                temp_best_each_subject_pred.extend(torch.max(yhat, 1)[1].tolist())
                best_each_subject_pred = temp_best_each_subject_pred
                # Save Weights
                if (config.train):
                    torch.save(model.state_dict(), weight_path)

        # For UF1 and UAR computation
        print('Best Predicted    :', best_each_subject_pred)
        accuracydict = {}
        accuracydict['pred'] = best_each_subject_pred
        accuracydict['truth'] = y.tolist()
        all_accuracy_dict[n_subName] = accuracydict

        print('Ground Truth :', y.tolist())
        print('Evaluation until this subject: ')
        total_pred.extend(torch.max(yhat, 1)[1].tolist())   # 这里存的是最后一个epoch的预测结果
        total_gt.extend(y.tolist())
        best_total_pred.extend(best_each_subject_pred)      # 这里存的是最好的预测结果
        # print(total_gt)
        # print(total_pred)
        # print(best_total_pred)
        UF1, UAR = recognition_evaluation(total_gt, total_pred, show=True)
        print('Evalution with last prediction\nUF1:', round(UF1, 4), '| UAR:', round(UAR, 4))
        best_UF1, best_UAR = recognition_evaluation(total_gt, best_total_pred, show=True)
        print('best UF1:', round(best_UF1, 4), '| best UAR:', round(best_UAR, 4))
        
        time_delta = timedelta(seconds=time.time()-time_one_sub)
        print("total time used for one subject: ", str(time_delta))
        print("="*20)

    print('Final Evaluation: ')
    UF1, UAR = recognition_evaluation(total_gt, total_pred) # 这里的UF1和UAR是
    print('Evalution with last prediction\nUF1:', round(UF1, 4), '| UAR:', round(UAR, 4))
    print(np.shape(total_gt))
    total_time_delta = timedelta(seconds=time.time()-t)
    print('Total Time Taken:', str(total_time_delta))
    print(all_accuracy_dict)


if __name__ == '__main__':
    # get_whole_u_v_os()
    # create_norm_u_v_os_train_test()

    # input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=strtobool, default=False)  # Train or use pre-trained weight for prediction
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=800, help='Number of epochs for training')
    parser.add_argument('--model_type', type=str, default="HTNet", help="decide which model to use")
    
    # HTNet-specific parameters
    parser.add_argument('--image_size', type=int, default=28, help='Input image size (e.g., 28 for 28x28 images)')
    parser.add_argument('--patch_size', type=int, default=7, help='Patch size for dividing the image')
    parser.add_argument('--dim', type=int, default=256, help='Base dimension for the model')
    parser.add_argument('--heads', type=int, default=3, help='Number of attention heads')
    parser.add_argument('--num_hierarchies', type=int, default=3, help='Number of hierarchies in the model')
    parser.add_argument('--block_repeats', type=str, default="2,2,10", help='Comma-separated list of block repeats for each hierarchy')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of output classes')

    # HTNet_Enhanced_v5 specific parameters
    parser.add_argument('--gb_tf_channels', type=int, default=64, help='Number of channels for global transformer')
    parser.add_argument('--gb_heads', type=int, default=2, help='Number of attention heads for global transformer')
    parser.add_argument('--gb_n_windows', type=int, default=7, help='Number of windows for global transformer')
    parser.add_argument('--gb_out_channel', type=int, default=256, help='Number of channels for gb branch output')

    config = parser.parse_args()
    # Convert block_repeats from string to tuple of integers
    config.block_repeats = tuple(map(int, config.block_repeats.split(',')))
    main(config)
