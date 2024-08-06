import pandas as pd
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, Subset
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import random
from imblearn.over_sampling import SMOTE
from PIL import Image
import os
from statsmodels.stats.contingency_tables import mcnemar


print(torch.cuda.is_available())


SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
#
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
#
fig_size=(7,6)
title_font = {'size':BIGGER_SIZE, 'color':'black', 'weight':'normal'} # Bottom vertical alignment for more space



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def accuracy(target, pred):
    return metrics.accuracy_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())


def compute_confusion_matrix(target, pred, labels, normalize=None):
    return metrics.confusion_matrix(
        target.detach().cpu().numpy(),
        pred.detach().cpu().numpy(),
        labels=labels,
        normalize=normalize
    )


# set up a standard CNN model
class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# set up a complex CNN model
class VGGNet(nn.Module):
    def __init__(self, num_classes):
        super(VGGNet, self).__init__()
        # define convolutional layers
        self.features = nn.Sequential(
            # first convolutional block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),

            # second convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2),

            # third convolutional block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),

            # fourth convolutional block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),

            # fifth convolutional block
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3)
        )
        # define fully connected layers
        if args.basetype == 'TLmix' or args.basetype == 'TL_B':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 4 * 4, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 7)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 4 * 4, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.4),
                nn.Linear(256, num_classes),
            )

    def forward(self, x):
        # convolutional layers
        x = self.features(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # fully connected layers
        x = self.classifier(x)
        return x


def train(model, optimizer, epochs, device):
    model.train()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        train_accuracies_batches = []
        train_losses_batches = []

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            # Compute accuracy and loss
            predictions = outputs.max(1)[1]
            train_accuracies_batches.append(accuracy(targets, predictions))
            train_losses_batches.append(loss.item())

        # Append average training accuracy to list
        train_accuracies.append(np.mean(train_accuracies_batches))
        train_losses.append(np.mean(train_losses_batches))

        # Compute accuracies and losses on validation set
        valid_accuracies_batches = []
        valid_losses_batches = []
        with torch.no_grad():
            model.eval()
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                loss = loss_fn(output, targets)

                predictions = output.max(1)[1]
                valid_accuracies_batches.append(accuracy(targets, predictions) * len(inputs))
                valid_losses_batches.append(loss.item() * len(inputs))

            model.train()

        # Append average validation accuracy and loss to list
        val_accuracies.append(np.sum(valid_accuracies_batches) / len(val_dataset))
        val_losses.append(np.sum(valid_losses_batches) / len(val_dataset))

        # print(f"Epoch {epoch + 1:<5}   training accuracy: {train_accuracies[-1]}")
        # print(f"              validation accuracy: {val_accuracies[-1]}")

    return train_accuracies, train_losses, val_accuracies, val_losses


# get confusion matrices and test accuracy
def test_model_performance():
    # Initialize the confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))
    labels = list(range(num_classes))

    with torch.no_grad():
        model.eval()
        test_accuracies = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            predictions = output.max(1)[1]

            test_accuracies.append(accuracy(targets, predictions) * len(inputs))

            batch_confusion_matrix = compute_confusion_matrix(targets, predictions, labels=labels)
            confusion_matrix += batch_confusion_matrix
            confusion_matrix = confusion_matrix.astype(int)

        test_accuracy = np.sum(test_accuracies) / len(test_dataset)

    return confusion_matrix, test_accuracy

# calculate the mean of confusion matrices and test accuracy for 30 baseline and VGG models
def load_cm(model_name):
    # load confusion matrix
    confusion_matrices = np.load(f'algorithms/{model_name}_confusion_matrices_{args.batch_size}_{args.epochs}_{args.lr}.npy', allow_pickle=True)
    mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
    mean_confusion_matrix = np.round(mean_confusion_matrix).astype(int)

    total_test_accuracy = np.load(f'algorithms/{model_name}_test_accuracy_{args.batch_size}_{args.epochs}_{args.lr}.npy', allow_pickle=True)
    avg_test_accuracy = np.mean(total_test_accuracy)
    print(f'The average test accuracy of {model_name} optimizer is : {avg_test_accuracy:.3f}')

    return mean_confusion_matrix

# normalize the confusion matrix
def normalize(matrix, axis):
    axis = {'true': 1, 'pred': 0}[axis]
    normalized_matrix = matrix / matrix.sum(axis=axis, keepdims=True)

    return normalized_matrix

# plot the confusion matrix for different models
def plot_cm(model_name, n=None):
    if model_name == 'Adam':
        mean_cm = load_cm(model_name)
        print(mean_cm)
        title = 'standard CNN model'
    elif model_name == 'VGG':
        mean_cm = load_cm(model_name)
        print(mean_cm)
        title = 'VGG model'
    # for 'SMOTE' and 'TL' methods
    # just use 'mean_cm' to replace the 'cm' for the next step and draw heatmap
    elif model_name == 'Smote':
        mean_cm, test_accuracy = test_model_performance()
        np.save(f'algorithms/{model_name}_confusion_matrices_{args.batch_size}_{args.epochs}_{args.lr}.npy', mean_cm)
        print(mean_cm)
        print(f'The test accuracy of VGG+SMOTE model is: {test_accuracy:.3f}')
        title = 'VGG+SMOTE model'
    elif model_name == 'TL':
        mean_cm, test_accuracy = test_model_performance()
        np.save(f'algorithms/TL_{args.datatype}/{model_name}_confusion_matrices_{args.epochs}_{args.lr}_{n}.npy', mean_cm)
        print('The number of samples is:', n*3)
        print(mean_cm)
        print(f'The test accuracy of VGG+TL_A model is: {test_accuracy:.3f}')
        title = 'VGG+TL_A1 model'
    elif model_name == 'TL2':
        mean_cm, test_accuracy = test_model_performance()
        np.save(f'algorithms/TL_{args.datatype}/{model_name}_confusion_matrices_{args.epochs}_{args.lr}.npy', mean_cm)
        print(mean_cm)
        print(f'The test accuracy of VGG+TL_A2 model is: {test_accuracy:.3f}')
        title = 'VGG+TL_A2 model'
    elif model_name == 'TL_B':
        mean_cm, test_accuracy = test_model_performance()
        np.save(f'algorithms/TL_{args.datatype}/{model_name}_confusion_matrices_{args.epochs}_{args.lr}.npy', mean_cm)
        print(mean_cm)
        print(f'The test accuracy of VGG+TL_B model is: {test_accuracy:.3f}')
        title = 'VGG+TL_B model'

    x_labels = [classes_str[i] for i in classes_str]
    y_labels = x_labels
    plt.figure(figsize=(7, 6))

    normalized_cm = normalize(mean_cm, 'true')
    row_sums = mean_cm.sum(axis=1)
    annot_matrix = np.empty(mean_cm.shape, dtype=object)

    for i in range(mean_cm.shape[0]):
        for j in range(mean_cm.shape[1]):
            percentage = f"{normalized_cm[i, j] * 100:.1f}"
            if i == j:
                annot_matrix[i, j] = f"{percentage}\n{mean_cm[i, j]}/{row_sums[i]}"
            else:
                annot_matrix[i, j] = f"{percentage}\n{mean_cm[i, j]}"

    sns.heatmap(
        ax=plt.gca(),
        data=normalized_cm,
        annot=annot_matrix,
        linewidths=0.5,
        cmap="Blues",
        cbar=False,
        fmt="",
        xticklabels=x_labels,
        yticklabels=y_labels,
    )

    plt.title(f'Confusion matrix for {title}', fontdict=title_font)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.ylabel("True class")
    plt.xlabel("Predicted class")
    plt.tight_layout()
    if model_name == 'TL' or model_name == 'TL2' or model_name == 'TL_B':
        plt.savefig(f'algorithms/TL_{args.datatype}/{model_name}_confusion_matrix_{args.epochs}_{args.lr}.pdf')
    else:
        plt.savefig(f'results_figure/{model_name}_confusion_matrix_{args.epochs}_{args.lr}.pdf')
    plt.show()

    return normalized_cm

# calculate the fairness score for different demographic groups
def calculate_probabilities(data, attribute, labels):
    C = 7
    probabilities = {label: [] for label in labels}

    for c in range(1, C + 1):
        for attr_value, label in zip(range(len(labels)), labels):
            if args.fairnesstype == 'equal':
                # equal opportunity
                title = 'Equal opportunity'
                data_c_attr = data[(data['Emotion'] == c) & (data[attribute] == attr_value)]
                if not data_c_attr.empty:
                    prob = len(data_c_attr[data_c_attr['Predicted'] == c]) / len(data_c_attr)
                    if prob == 0:
                        print(f"{emotion_map[c]} emotion: The number of {label} samples is {len(data_c_attr)}")
                else:
                    prob = 0
                    print(f"{emotion_map[c]} emotion: No samples of {label}")
                probabilities[label].append(prob)

            elif args.fairnesstype == 'predictive':
                # predictive parity
                title = 'Predictive parity'
                data_c_attr = data[(data['Predicted'] == c) & (data[attribute] == attr_value)]
                if not data_c_attr.empty:
                    prob = len(data_c_attr[data_c_attr['Emotion'] == c]) / len(data_c_attr)
                    if prob == 0:
                        print(
                            f'{emotion_map[c]} emotion: The number of {label} predicted samples is {len(data_c_attr)}')
                else:
                    prob = 0
                    print(f'{emotion_map[c]} emotion: No predicted samples of {label}')
                probabilities[label].append(prob)

            elif args.fairnesstype == 'demographic':
                # demographic parity
                title = 'Demographic parity'
                data_attr = data[data[attribute] == attr_value]
                if not data_attr.empty:
                    prob = len(data_attr[data_attr['Predicted'] == c]) / len(data_attr)
                    if prob == 0:
                        print(
                            f"The number of {label} samples is {len(data_attr)}, but no predicted samples for {emotion_map[c]} emotion")
                else:
                    prob = 0
                    print(f'No samples of {label}')
                probabilities[label].append(prob)

    mean_class_accuracies = np.zeros(len(labels))
    F_measure = []
    for idx, label in enumerate(labels):
        mean_class_accuracy = np.mean(probabilities[label])
        mean_class_accuracies[idx] = mean_class_accuracy
        # print(f'The mean class-wise accuracy of {label} is : {mean_class_accuracy * 100:.1f}%')
        measure_value = np.sum(probabilities[label])
        F_measure.append(measure_value)

    if attribute == 'gender':
        ratio_1 = F_measure[0] / F_measure[1]
        ratio_2 = F_measure[1] / F_measure[0]
        F = min(ratio_1, ratio_2)
    else:
        d_value = max(F_measure)
        F_values = F_measure / d_value
        F = min(F_values)

    # print(f'The fairness of {args.fairnesstype} between different groups is: {F:.3f}')
    # print('********************************')

    # # v1: plot the fairness between subgroups at the same attribute (like: compared male with female)
    # bar_width = 0.8 / len(labels)
    # index = np.arange(len(emotion_labels))
    #
    # fig, ax = plt.subplots(figsize=(10, 9))
    #
    # for i, label in enumerate(labels):
    #     ax.bar(index + i * bar_width, probabilities[label], bar_width, label=label)
    #
    # ax.set_ylabel('Fairness score')
    # ax.set_title(f'{title} for {attribute} using {args.basetype} model', fontdict=title_font)
    # ax.set_xticks(index + bar_width * (len(labels) - 1) / 2)
    # ax.set_xticklabels(emotion_labels, rotation=45)
    # ax.legend()
    # plt.savefig(f'fairness_figure/{args.basetype}_{args.fairnesstype}_fairness_{attribute}.pdf')
    # plt.show()

    return probabilities, mean_class_accuracies, F

# get the samples from RAF-DB training data of other emotions except for fear, disgust, anger
class CustomImageDataset(Dataset):
    def __init__(self, root, folder_indices, num_samples, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform

        for folder_index in folder_indices:
            folder_path = os.path.join(root, str(folder_index))
            files = os.listdir(folder_path)[:num_samples]

            for file in files:
                img_path = os.path.join(folder_path, file)
                img = Image.open(img_path).convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
                self.data.append(img)
                self.labels.append(folder_index-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# get the filtered FER2013 dataset to retrain in TL
class FER2013Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pixels = self.data.iloc[idx, 1]
        label = self.data.iloc[idx, 0]
        pixels = np.array([int(pixel) for pixel in pixels.split()]).reshape(48, 48).astype('uint8')

        # converting greyscale images to RGB
        pixels = np.stack((pixels,) * 3, axis=-1)
        pixels = Image.fromarray(pixels)

        if self.transform:
            pixels = self.transform(pixels)

        return pixels, label

# the function for retraining step in TL
def train_model(model, data_loader, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        batch_itter = 0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item()
            batch_itter += 1

            loss.backward()
            optimizer.step()

        print(
            f"\nEpoch {epoch + 1}/{epochs} loss: {loss.item():.4f} avg. loss: {running_loss / batch_itter:.4f}")


if __name__ == "__main__":
    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='basetrain', choices=['basetrain', 'VGGtrain', 'plotcm', 'SMOTE', 'TL', 'TL2', 'TL_B', 'prediction', 'fairness', 'bias', 'twodataset', 'McNemar'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--basetype', type=str, default='Adam', choices=['Adam', 'VGG', 'VGG+SMOTE', 'TL', 'TL2', 'TL_B', 'TLmix'], help='model type (default: %(default)s)')
    parser.add_argument('--fairnesstype', type=str, default='equal', choices=['equal', 'predictive', 'demographic'], help='fairness measure (default: %(default)s)')
    parser.add_argument('--datatype', type=str, default='FER', choices=['FER', 'SFEW'], help='dataset type for TL (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device
    # rafdb_data = pd.read_csv('rafdb_aligned.csv')
    # img_size = 100  # original size of the image
    # targetx = 128
    # targety = 128

    # training data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # test data transform
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    set_seed(42)

    if args.mode == 'SMOTE':
        train_dataset = datasets.ImageFolder(root='aligned/resampled/train_resampled', transform=train_transform)
    else:
        train_dataset = datasets.ImageFolder(root='aligned/train', transform=train_transform)

    test_dataset = datasets.ImageFolder(root='aligned/test', transform=test_transform)

    # get validation dataset and test dataset
    num_test = len(test_dataset)
    val_size = int(0.5 * num_test)
    test_size = num_test - val_size
    val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    # print("Number of Training data:", len(train_dataset))   # 11519
    # x, y = next(iter(train_loader))
    # print("Batch dimension (B x C x H x W):", x.shape)    # torch.Size([32, 3, 128, 128])
    # print(f"Number of distinct labels: {len(set(train_dataset.targets))} (unique labels: {set(train_dataset.targets)})")
    #                                                                    # 7 (unique labels: {0, 1, 2, 3, 4, 5, 6})
    #
    # print(len(val_dataset))   # 1434
    # print(len(test_dataset))  # 1435

    num_classes = 7
    loss_fn = nn.CrossEntropyLoss()

    # Map from class index to class name
    emotion_map = {
        1: 'Surprise',
        2: 'Fear',
        3: 'Disgust',
        4: 'Happiness',
        5: 'Sadness',
        6: 'Anger',
        7: 'Neutral'
    }

    classes = {index: name for name, index in train_dataset.class_to_idx.items()}
    # {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7'}
    classes_str = {index: emotion_map[int(emotion_id)] for index, emotion_id in classes.items()}
    # {0: 'Surprise', 1: 'Fear', 2: 'Disgust', 3: 'Happiness', 4: 'Sadness', 5: 'Anger', 6: 'Neutral'}
    emotion_labels = [emotion_map[i] for i in range(1, 8)]

    if args.mode == 'basetrain':
        # Initialize a list to store confusion matrices
        all_confusion_matrices = np.zeros((30, num_classes, num_classes))
        total_test_accuracy = []

        for i in range(30):
            set_seed(i)

            model = Model(num_classes).to(device)

            # train model with Adam optimizer
            optimizer = optim.Adam(model.parameters(), lr=args.lr)  # lr=0.0001
            train_accuracies, train_losses, val_accuracies, val_losses = train(model, optimizer, args.epochs, device)
            torch.save(model.state_dict(), f'algorithms/baseline/{args.basetype}_model {i}_{args.batch_size}_{args.epochs}_{args.lr}.pt')

            confusion_matrix, test_accuracy = test_model_performance()

            # Save the confusion matrix and test accuracy for this loop
            all_confusion_matrices[i] = confusion_matrix
            total_test_accuracy.append(test_accuracy)

        # Save the list of confusion matrices to a file
        np.save(f'algorithms/{args.basetype}_confusion_matrices_{args.batch_size}_{args.epochs}_{args.lr}.npy', all_confusion_matrices)
        np.save(f'algorithms/{args.basetype}_test_accuracy_{args.batch_size}_{args.epochs}_{args.lr}.npy', total_test_accuracy)

    elif args.mode == 'VGGtrain':
        # Initialize a list to store confusion matrices
        all_confusion_matrices = np.zeros((30, num_classes, num_classes))
        total_test_accuracy = []

        for i in range(30):
            set_seed(i)

            # complex CNN model using VGG
            model = VGGNet(num_classes).to(device)

            # train model with Adam optimizer
            optimizer = optim.Adam(model.parameters(), lr=args.lr)  # lr=0.0001
            train_accuracies, train_losses, val_accuracies, val_losses = train(model, optimizer, args.epochs, device)
            torch.save(model.state_dict(), f'algorithms/VGG/{args.basetype}_model {i}_{args.batch_size}_{args.epochs}_{args.lr}.pt')

            confusion_matrix, test_accuracy = test_model_performance()

            # Save the confusion matrix and test accuracy for this loop
            all_confusion_matrices[i] = confusion_matrix
            total_test_accuracy.append(test_accuracy)

        # Save the list of confusion matrices to a file
        np.save(f'algorithms/{args.basetype}_confusion_matrices_{args.batch_size}_{args.epochs}_{args.lr}.npy', all_confusion_matrices)
        np.save(f'algorithms/{args.basetype}_test_accuracy_{args.batch_size}_{args.epochs}_{args.lr}.npy', total_test_accuracy)

    elif args.mode == 'plotcm':
        model_name_list = ['Adam', 'VGG']
        diagonal_data = {}

        for model_name in model_name_list:
            normalized_cm = plot_cm(model_name)
            diagonal_data[model_name] = np.diag(normalized_cm)

        labels = [classes_str[i] for i in classes_str]
        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=fig_size)

        model_names = list(diagonal_data.keys())
        model1 = model_names[0]
        model2 = model_names[1]

        rects1 = ax.bar(x - width / 2, diagonal_data[model1], width, label='Standard')
        rects2 = ax.bar(x + width / 2, diagonal_data[model2], width, label=model2)

        ax.set_ylabel('Accuracy score')
        ax.set_title('Accuracy comparison by emotion class', fontdict=title_font)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'results_figure/comparison_{model1}_vs_{model2}.pdf')
        plt.show()

    elif args.mode == 'SMOTE':
        '''
        # apply SMOTE to increase the number of imbalanced dataset subgroups
        df = pd.read_csv('rafdb_train.csv')

        # Function to apply SMOTE for a specific emotion and attribute
        def apply_smote_for_condition(df, emotion, attribute, target_value):
            condition_df = df[df['Emotion'] == emotion]
            attribute_counts = condition_df[attribute].value_counts()
            other_count = attribute_counts.drop(target_value).max()

            X = condition_df.drop(columns=['Image', 'Emotion'])
            y = condition_df[attribute]

            smote = SMOTE(sampling_strategy={target_value: other_count})
            X_resampled, y_resampled = smote.fit_resample(X, y)

            resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
            resampled_df['Emotion'] = emotion
            resampled_df[attribute] = y_resampled

            return resampled_df

        # Function to add Image column from original data by randomly assigning an Image from matching rows
        def add_random_image_column(df_resampled, df_original):
            df_resampled['Image'] = None

            for index, row in df_resampled.iterrows():
                matching_rows = df_original[
                    (df_original['Emotion'] == row['Emotion']) &
                    (df_original['gender'] == row['gender']) &
                    (df_original['race'] == row['race']) &
                    (df_original['age'] == row['age'])
                    ]

                if not matching_rows.empty:
                    random_image = matching_rows.sample(n=1)['Image'].values[0]
                    df_resampled.at[index, 'Image'] = random_image

            # Drop rows where Image is NaN (i.e., no matching row found in original data)
            df_resampled = df_resampled.dropna(subset=['Image'])
            return df_resampled

        # Initialize a list to hold resampled dataframes
        resampled_dfs = []

        # Apply SMOTE for race=1 in emotion=2
        resampled_dfs.append(apply_smote_for_condition(df, emotion=2, attribute='race', target_value=1))
        # Apply SMOTE for age=0 in emotion=2, 3, 6
        for emotion in [2, 3, 6]:
            resampled_dfs.append(apply_smote_for_condition(df, emotion=emotion, attribute='age', target_value=0))
        # Apply SMOTE for age=4 in emotion=2, 6
        for emotion in [2, 6]:
            resampled_dfs.append(apply_smote_for_condition(df, emotion=emotion, attribute='age', target_value=4))

        # Combine all resampled dataframes
        combined_resampled_df = pd.concat(resampled_dfs, ignore_index=True)
        # Add Image column and filter out rows without matching Image
        filtered_resampled_df = add_random_image_column(combined_resampled_df, df)
        # Combine the original dataframe with the filtered resampled dataframe
        final_df = pd.concat([df, filtered_resampled_df], ignore_index=True)
        final_df.to_csv('rafdb_train_resampled.csv', index=False)
        '''

        # train model
        model = VGGNet(num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)  # lr=0.0001
        train_accuracies, train_losses, val_accuracies, val_losses = train(model, optimizer, args.epochs, device)
        torch.save(model.state_dict(), 'algorithms/VGG+SMOTE_model.pt')

        # # Load the saved model file
        # model = VGGNet(num_classes)
        # model.load_state_dict(torch.load('algorithms/VGG+SMOTE_model.pt', map_location=torch.device(args.device)))
        # model.to(device)

        normalized_cm = plot_cm('Smote')

    elif args.mode == 'TL':
        # choose the best performance VGG model
        total_test_accuracy = np.load('algorithms/VGG_test_accuracy_32_30_0.0001.npy', allow_pickle=True)
        max_test_accuracy = max(total_test_accuracy)
        max_index = list(total_test_accuracy).index(max_test_accuracy)
        print('The index of best performance VGG model is: ', max_index)  # 4

        # Load the saved model file to test
        model = VGGNet(num_classes)
        model.load_state_dict(
            torch.load(f'algorithms/VGG/VGG_model {max_index}_32_30_0.0001.pt', map_location=torch.device(args.device)))
        model.to(device)

        # implement TL method
        # freeze the parameters of convolutional layers
        for param in model.features.parameters():
            param.requires_grad = False

        # define the optimizer
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)

        # choose the same number of emotion samples (i)
        i = 50
        if args.datatype == 'SFEW':
            # get the same number of samples of fear, disgust, anger emotions from SFEW training dataset
            root = 'aligned/Train_Aligned_Faces'
            folder_indices = [2, 3, 6]  # here is the index of folders, not the true label which should minus 1

            num_samples = i   # i=50
            SFEW_train_dataset = CustomImageDataset(root, folder_indices, num_samples, transform=train_transform)
            SFEW_train_loader = DataLoader(SFEW_train_dataset, batch_size=16, shuffle=True)

            # retrain using the SFEW data in last layers (epoch=15/20)
            train_model(model, SFEW_train_loader, args.epochs)

        elif args.datatype == 'FER':
            # load the filtered FER2013 data (only including three emotions fear, disgust, anger)
            FERdata = pd.read_csv('fer_sampled.csv')
            fear_data = FERdata[FERdata['emotion'] == 1]
            disgust_data = FERdata[FERdata['emotion'] == 2]
            anger_data = FERdata[FERdata['emotion'] == 5]

            # get the samples of fear, disgust, anger emotions from FER2013 dataset (i=50)
            sampled_fear = fear_data.sample(n=i, random_state=42)
            sampled_anger = anger_data.sample(n=i, random_state=42)
            sampled_disgust = disgust_data.sample(n=i, random_state=42)
            combined_sample = pd.concat([sampled_fear, sampled_anger, sampled_disgust], ignore_index=True)
            fer_train_dataset = FER2013Dataset(combined_sample, transform=train_transform)
            fer_train_loader = DataLoader(fer_train_dataset, batch_size=16, shuffle=True)

            # retrain using the filtered FER2013 data in last layers
            train_model(model, fer_train_loader, args.epochs)

        torch.save(model.state_dict(), f'algorithms/TL_{args.datatype}/TL_model_{args.epochs}_{i}.pt')
        # evaluate the model performance
        normalized_cm = plot_cm('TL', i)

    elif args.mode == 'TL2':
        # choose the best performance VGG model
        total_test_accuracy = np.load('algorithms/VGG_test_accuracy_32_30_0.0001.npy', allow_pickle=True)
        max_test_accuracy = max(total_test_accuracy)
        max_index = list(total_test_accuracy).index(max_test_accuracy)
        print('The index of best performance VGG model is: ', max_index)  # 4

        # Load the saved model file to test
        model = VGGNet(num_classes)
        model.load_state_dict(
            torch.load(f'algorithms/VGG/VGG_model {max_index}_32_30_0.0001.pt', map_location=torch.device(args.device)))
        model.to(device)

        # implement TL method
        # freeze the parameters of convolutional layers
        for param in model.features.parameters():
            param.requires_grad = False

        # define the optimizer
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)

        # make all emotions to have the same number of samples (i)
        i = 50
        if args.datatype == 'SFEW':
            # load the SFEW training dataset(all emotions)
            root = 'aligned/Train_Aligned_Faces'
            folder_indices = range(1, 8)  # here is the index of folders, not the true label which should minus 1
            num_samples = i
            SFEW_train_dataset = CustomImageDataset(root, folder_indices, num_samples, transform=train_transform)
            SFEW_train_loader = DataLoader(SFEW_train_dataset, batch_size=16, shuffle=True)

            # retrain using the SFEW data in last layers (epoch=10)
            train_model(model, SFEW_train_loader, args.epochs)

        elif args.datatype == 'FER':
            # load the FER2013 data(all emotions)
            FERdata = pd.read_csv('fer2013_modified.csv')
            train_FERdata = FERdata[FERdata['Usage'] == 'Training']
            filtered_train_data = []
            for emo in range(6):
                emo_data = train_FERdata[train_FERdata['emotion'] == emo]
                sampled_data = emo_data.sample(n=i, random_state=42)
                filtered_train_data.append(sampled_data)

            combined_sample = pd.concat(filtered_train_data, ignore_index=True)
            fer_train_dataset = FER2013Dataset(combined_sample, transform=train_transform)
            fer_train_loader = DataLoader(fer_train_dataset, batch_size=16, shuffle=True)

            # retrain using the FER2013 data in last layers (epoch=10)
            train_model(model, fer_train_loader, args.epochs)

        torch.save(model.state_dict(), f'algorithms/TL_{args.datatype}/TL2_model_{args.epochs}.pt')
        # evaluate the model performance
        normalized_cm = plot_cm('TL2')

    elif args.mode == 'TL_B':
        # choose the best performance VGG model
        total_test_accuracy = np.load('algorithms/VGG_test_accuracy_32_30_0.0001.npy', allow_pickle=True)
        max_test_accuracy = max(total_test_accuracy)
        max_index = list(total_test_accuracy).index(max_test_accuracy)
        print('The index of best performance VGG model is: ', max_index)  # 4

        # Load the saved model file to test
        model = VGGNet(num_classes)
        model.load_state_dict(
            torch.load(f'algorithms/VGG/VGG_model {max_index}_32_30_0.0001.pt', map_location=torch.device(args.device)))
        model.to(device)

        # freeze the parameters of convolutional layers
        for param in model.features.parameters():
            param.requires_grad = False

        # the classifier of model
        model.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 7)
        )
        model.to(device)
        # define the optimizer and train model
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)

        if args.datatype == 'SFEW':
            # pre-trained VGG model (retrain the filtered SFEW first and then retrain RAF-DB data)
            # get the same number of samples of fear, disgust, anger emotions from SFEW training dataset
            root = 'aligned/Train_Aligned_Faces'
            folder_indices = [2, 3, 6]  # here is the index of folders, not the true label which should be minus 1
            i = 75
            num_samples = i
            SFEW_train_dataset = CustomImageDataset(root, folder_indices, num_samples, transform=train_transform)
            SFEW_train_loader = DataLoader(SFEW_train_dataset, batch_size=16, shuffle=True)

            # retrain the filtered SFEW dataset first (epoch = 15)
            train_model(model, SFEW_train_loader, args.epochs)
            # retrain the RAF-DB training dataset then
            train_model(model, train_loader, 10)

        elif args.datatype == 'FER':
            # pre-trained VGG model (retrain the filtered FER2013 first and then retrain RAF-DB data)
            # filtered FER2013 data includes three emotions fear, disgust, anger
            # each emotion has N(14388)/7 = 2055 samples
            FERdata = pd.read_csv('fer_sampled_v1.csv')
            fer_train_dataset = FER2013Dataset(FERdata, transform=train_transform)
            fer_train_loader = DataLoader(fer_train_dataset, batch_size=32, shuffle=True)

            # retrain the filtered FER2013 data first (epoch = 15)
            train_model(model, fer_train_loader, args.epochs)
            # retrain the RAF-DB training dataset then
            train_model(model, train_loader, 10)

        torch.save(model.state_dict(), f'algorithms/TL_{args.datatype}/TL_B_model_{args.epochs}.pt')
        # evaluate the model performance
        normalized_cm = plot_cm('TL_B')

        '''
        # implement experiment for TL method, use different number of emotions from FER2013
        # for example, fear_number = 200, 400, ..., 2000, other emotions are same
        # still use combined dataset (RAF-DB + FER2013)
        FERdata = pd.read_csv('fer_sampled_v1.csv')
        fear_data = FERdata[FERdata['emotion'] == 1]
        anger_data = FERdata[FERdata['emotion'] == 5]
        disgust_data = FERdata[FERdata['emotion'] == 2]

        all_confusion_matrices = np.zeros((10, num_classes, num_classes))
        counter = 0
        total_test_accuracy = []

        for i in range(200, 2001, 200):
            sampled_fear = fear_data.sample(n=i, random_state=42)
            sampled_anger = anger_data.sample(n=i, random_state=42)
            sampled_disgust = disgust_data.sample(n=i, random_state=42)
            combined_sample = pd.concat([sampled_fear, sampled_anger, sampled_disgust], ignore_index=True)

            # combined dataset
            if args.basetype == 'TLmix':
                fer_train_dataset = FER2013Dataset(combined_sample, transform=train_transform)
                fer_train_loader = DataLoader(fer_train_dataset, batch_size=32, shuffle=True)

                train_model(model, fer_train_loader, args.epochs)
                train_model(model, train_loader, args.epochs)

                # # combine FER2013 and RAF-DB dataset
                # combined_train_dataset = ConcatDataset([fer_train_dataset, train_dataset])
                # train_loader = DataLoader(combined_train_dataset, batch_size=32, shuffle=True)
                # train(model, optimizer, args.epochs, device)

                torch.save(model.state_dict(), f'algorithms/TLmix/VGG+TLmix_{args.epochs}_{i}.pt')
                cm, test_accuracy = test_model_performance()

                all_confusion_matrices[counter] = cm
                counter += 1
                total_test_accuracy.append(test_accuracy)

        np.save(f'algorithms/{args.basetype}/{args.basetype}_confusion_matrices_{args.epochs}.npy', all_confusion_matrices)
        np.save(f'algorithms/{args.basetype}/{args.basetype}_test_accuracy_{args.epochs}.npy', total_test_accuracy)

        print('Finished')
        '''

    elif args.mode == 'prediction':
        # # get the file name of test_dataset
        # test_image_paths = [test_dataset.dataset.imgs[idx][0] for idx in test_dataset.indices]
        # test_image_names = [os.path.basename(path).split('/')[-1].split('\\')[-1] for path in test_image_paths]
        # # test_image_info = rafdb_data[rafdb_data['Image'].isin(test_image_names)]
        # # save new file for test images
        # # test_image_info.to_csv('rafdb_test.csv', index=False)
        # data = pd.read_csv('rafdb_test.csv')

        # use the total dataset of RAF-DB
        data = pd.read_csv('rafdb_aligned.csv')

        total_dataset = datasets.ImageFolder(root='aligned/aligned_total', transform=test_transform)
        total_loader = DataLoader(total_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
        image_names = [os.path.basename(total_dataset.imgs[i][0]) for i in range(len(total_dataset))]

        if args.basetype == 'Adam':
            for i in range(30):
                # Load the saved model file to test
                model = Model(num_classes)
                model.load_state_dict(torch.load(f'algorithms/baseline/Adam_model {i}_32_30_0.0001.pt',
                                                 map_location=torch.device(args.device)))
                model.to(device)

                # Evaluate test set
                all_predictions = []
                with torch.no_grad():
                    model.eval()
                    for inputs, targets in total_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        output = model(inputs)
                        predictions = output.max(1)[1]
                        all_predictions.extend(predictions.cpu().numpy())

                all_predictions = [classes[pred] for pred in all_predictions]
                predictions_dict = dict(zip(image_names, all_predictions))
                data['Predicted'] = data['Image'].map(predictions_dict)
                data['Predicted'] = data['Predicted'].astype(int)
                data.to_csv(f'prediction_files/baseline_csv/rafdb_total_pred_base {i}.csv', index=False)
                np.save(f'prediction_files/baseline/rafdb_total_pred_base {i}.npy', data)

        elif args.basetype == 'VGG':
            for i in range(30):
                # Load the saved model file to test
                model = VGGNet(num_classes)
                model.load_state_dict(torch.load(f'algorithms/VGG/VGG_model {i}_32_30_0.0001.pt',
                                                 map_location=torch.device(args.device)))
                model.to(device)

                # Evaluate test set
                all_predictions = []
                with torch.no_grad():
                    model.eval()
                    for inputs, targets in total_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        output = model(inputs)
                        predictions = output.max(1)[1]
                        all_predictions.extend(predictions.cpu().numpy())

                all_predictions = [classes[pred] for pred in all_predictions]
                predictions_dict = dict(zip(image_names, all_predictions))
                data['Predicted'] = data['Image'].map(predictions_dict)
                data['Predicted'] = data['Predicted'].astype(int)
                data.to_csv(f'prediction_files/VGG_csv/rafdb_total_pred_VGG {i}.csv', index=False)
                np.save(f'prediction_files/VGG/rafdb_total_pred_VGG {i}.npy', data)

        elif args.basetype == 'VGG+SMOTE':
            # Load the saved model file to test
            model = VGGNet(num_classes)
            model.load_state_dict(torch.load(f'algorithms/VGG+SMOTE_model.pt', map_location=torch.device(args.device)))
            model.to(device)

            # Evaluate test set
            all_predictions = []
            with torch.no_grad():
                model.eval()
                for inputs, targets in total_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    output = model(inputs)
                    predictions = output.max(1)[1]
                    all_predictions.extend(predictions.cpu().numpy())

            all_predictions = [classes[pred] for pred in all_predictions]
            predictions_dict = dict(zip(image_names, all_predictions))
            data['Predicted'] = data['Image'].map(predictions_dict)
            data['Predicted'] = data['Predicted'].astype(int)
            data.to_csv(f'prediction_files/rafdb_total_pred_{args.basetype}.csv', index=False)

        elif args.basetype == 'TL':
            # Load the saved model file to test
            model = VGGNet(num_classes)
            if args.datatype == 'FER':
                model.load_state_dict(torch.load(f'algorithms/TL_FER/TL_model_20_50.pt',
                                                 map_location=torch.device(args.device)))
            elif args.datatype == 'SFEW':
                model.load_state_dict(torch.load(f'algorithms/TL_SFEW/TL_model_15_50.pt',
                                                 map_location=torch.device(args.device)))
            model.to(device)

            # Evaluate test set
            all_predictions = []
            with torch.no_grad():
                model.eval()
                for inputs, targets in total_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    output = model(inputs)
                    predictions = output.max(1)[1]
                    all_predictions.extend(predictions.cpu().numpy())

            all_predictions = [classes[pred] for pred in all_predictions]
            predictions_dict = dict(zip(image_names, all_predictions))
            data['Predicted'] = data['Image'].map(predictions_dict)
            data['Predicted'] = data['Predicted'].astype(int)
            data.to_csv(f'prediction_files/rafdb_total_pred_VGG+{args.basetype}_{args.datatype}.csv', index=False)

        elif args.basetype == 'TL2':
            # Load the saved model file to test
            model = VGGNet(num_classes)
            model.load_state_dict(torch.load(f'algorithms/TL_{args.datatype}/TL2_model_10.pt',
                                             map_location=torch.device(args.device)))
            model.to(device)

            # Evaluate test set
            all_predictions = []
            with torch.no_grad():
                model.eval()
                for inputs, targets in total_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    output = model(inputs)
                    predictions = output.max(1)[1]
                    all_predictions.extend(predictions.cpu().numpy())

            all_predictions = [classes[pred] for pred in all_predictions]
            predictions_dict = dict(zip(image_names, all_predictions))
            data['Predicted'] = data['Image'].map(predictions_dict)
            data['Predicted'] = data['Predicted'].astype(int)
            data.to_csv(f'prediction_files/rafdb_total_pred_VGG+{args.basetype}_{args.datatype}.csv', index=False)

        elif args.basetype == 'TL_B':
            # Load the saved model file to test
            model = VGGNet(num_classes)
            model.load_state_dict(torch.load(f'algorithms/TL_{args.datatype}/TL_B_model_15.pt',
                                             map_location=torch.device(args.device)))
            model.to(device)

            # Evaluate test set
            all_predictions = []
            with torch.no_grad():
                model.eval()
                for inputs, targets in total_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    output = model(inputs)
                    predictions = output.max(1)[1]
                    all_predictions.extend(predictions.cpu().numpy())

            all_predictions = [classes[pred] for pred in all_predictions]
            predictions_dict = dict(zip(image_names, all_predictions))
            data['Predicted'] = data['Image'].map(predictions_dict)
            data['Predicted'] = data['Predicted'].astype(int)
            data.to_csv(f'prediction_files/rafdb_total_pred_VGG+{args.basetype}_{args.datatype}.csv', index=False)

        print('Finished')

    elif args.mode == 'fairness':
        gender_labels = ['Male', 'Female']
        race_labels = ['Cau', 'A-A', 'Asian']
        age_labels = ['0-3', '4-19', '20-39', '40-69', '70+']

        # calculate the fairness score using baseline model and VGG model
        # get the mean of 30 models and confidence interval
        def fairness_score_normal(model_name, name, attribute, labels):
            total_prob = []
            total_mean_class_accuracy = np.zeros((30, len(labels)))   # shape:(30, 2) / (30, 3) / (30, 5)
            total_F = []
            for i in range(30):
                # using whole dataset
                data = pd.read_csv(f'prediction_files/{model_name}_csv/rafdb_total_pred_{name} {i}.csv')
                # # using test dataset
                # test_data = data[data['Image'].str.contains('test')]
                probabilities, mean_class_accuracies, F = calculate_probabilities(data, attribute, labels)

                total_prob.append(probabilities)
                total_mean_class_accuracy[i, :] = mean_class_accuracies
                total_F.append(F)

            # the mean of fairness scores for each label (male/female/Cau/A-A...)
            mean_probabilities = {label: [] for label in labels}
            ci_probabilities = {label: [] for label in labels}

            for label in labels:
                emotion_probs = np.array([prob[label] for prob in total_prob])
                mean_probs = np.mean(emotion_probs, axis=0)
                std_probs = np.std(emotion_probs, axis=0, ddof=1)

                n = len(emotion_probs)
                z = 1.96
                ci_lower = mean_probs - z * (std_probs / np.sqrt(n))
                ci_upper = mean_probs + z * (std_probs / np.sqrt(n))

                mean_probabilities[label] = mean_probs
                ci_probabilities[label] = (ci_lower, ci_upper)

            # get the average value of 30 mean emotion class-wise accuracy for each label
            mean_class_accuracy_per_label = np.round(np.mean(total_mean_class_accuracy, axis=0), 3)

            # get the average value of 30 fairness measure
            mean_F = np.mean(total_F)

            return mean_probabilities, ci_probabilities, mean_class_accuracy_per_label, mean_F

        # calculate the fairness score using SMOTE and TL model
        def fairness_score_advanced(model_name, attribute, labels, datatype=None):
            # using whole dataset
            if model_name == 'VGG+SMOTE':
                data = pd.read_csv(f'prediction_files/rafdb_total_pred_{model_name}.csv')
            else:
                data = pd.read_csv(f'prediction_files/rafdb_total_pred_VGG+{model_name}_{datatype}.csv')
            # # using test dataset
            # test_data = data[data['Image'].str.contains('test')]
            probabilities, mean_class_accuracies, F = calculate_probabilities(data, attribute, labels)
            mean_class_accuracies = np.round(mean_class_accuracies, 3)

            return probabilities, mean_class_accuracies, F

        # plot the comparison of subgroups for models
        def plot_fairness_scores(mean_prob, ci_prob, emotion_labels, attribute, label):
            x = np.arange(len(emotion_labels))  # the label locations
            width = 0.2  # the width of the bars

            fig, ax = plt.subplots(figsize=(10, 6))

            rects1 = ax.bar(x - 1.5 * width, mean_prob['Baseline'][label], width,
                            yerr=np.abs(ci_prob['Baseline'][label] - mean_prob['Baseline'][label]), label='Standard',
                            capsize=5)
            rects2 = ax.bar(x - 0.5 * width, mean_prob['VGG'][label], width,
                            yerr=np.abs(ci_prob['VGG'][label] - mean_prob['VGG'][label]), label='VGG', capsize=5)
            rects3 = ax.bar(x + 0.5 * width, mean_prob['VGG+SMOTE'][label], width, label='VGG+SMOTE')
            # rects4 = ax.bar(x + width, mean_prob['TL-B with FER'][label], width, label='TL-B with FER', alpha=0.7, color='gold')
            rects5 = ax.bar(x + 1.5 * width, mean_prob['TL-B with SFEW'][label], width, label='TL-B with SFEW', alpha=0.7, color='lightskyblue')

            ax.set_ylabel('Fairness score')
            ax.set_title(f'Equal opportunity comparison by models for {attribute}({label})', fontdict=title_font)
            ax.set_xticks(x)
            ax.set_xticklabels(emotion_labels)
            ax.legend(loc='lower right')
            if label in gender_labels:
                ax.set_ylim(0.75, 1.0)
            elif label in race_labels:
                ax.set_ylim(0.75, 1.03)
            elif label in age_labels:
                ax.set_ylim(0.65, 1.03)
            fig.tight_layout()
            plt.savefig(f'fairness_figure/{args.fairnesstype}_comparison_models_{label}.pdf')
            plt.show()

        def fairness_diff_label(attribute, labels):
            mean_prob_baseline, ci_baseline, mean_class_accuracy_per_label_base, mean_F_base = fairness_score_normal('baseline', 'base', attribute, labels)
            mean_prob_VGG, ci_VGG, mean_class_accuracy_per_label_VGG, mean_F_VGG = fairness_score_normal('VGG', 'VGG', attribute, labels)
            prob_smote, mean_class_accuracies_smote, F_smote = fairness_score_advanced('VGG+SMOTE', attribute, labels)
            # prob_TLB_FER, mean_class_accuracies_TLB_FER, F_TLB_FER = fairness_score_advanced('TL_B', attribute, labels, 'FER')
            prob_TLB_SFEW, mean_class_accuracies_TLB_SFEW, F_TLB_SFEW = fairness_score_advanced('TL_B', attribute, labels, 'SFEW')

            print(f'Sensitive attribute is: {attribute}')
            print(f'The standard CNN model: mean emotion class-wise accuracy: {mean_class_accuracy_per_label_base}, fairness measure F: {mean_F_base:.3f}')
            print(f'VGG model: mean emotion class-wise accuracy: {mean_class_accuracy_per_label_VGG}, fairness measure F: {mean_F_VGG:.3f}')
            print(f'VGG+SMOTE model: mean emotion class-wise accuracy: {mean_class_accuracies_smote}, fairness measure F: {F_smote:.3f}')
            # print(f'VGG+TL-B with FER model: mean emotion class-wise accuracy: {mean_class_accuracies_TLB_FER}, fairness measure F: {F_TLB_FER:.3f}')
            print(f'VGG+TL-B with SFEW model: mean emotion class-wise accuracy: {mean_class_accuracies_TLB_SFEW}, fairness measure F: {F_TLB_SFEW:.3f}')

            # Prepare the data structures to hold mean and CI for different models
            mean_probabilities = {
                'Baseline': mean_prob_baseline,
                'VGG': mean_prob_VGG,
                'VGG+SMOTE': prob_smote,
                # 'TL-B with FER': prob_TLB_FER,
                'TL-B with SFEW': prob_TLB_SFEW
            }

            ci_probabilities = {
                'Baseline': ci_baseline,
                'VGG': ci_VGG
            }

            # Call the plotting function for different subgroups (male/female, cau/A-A/Asian, age ranges)
            for label in labels:
                plot_fairness_scores(mean_probabilities, ci_probabilities, emotion_labels, attribute, label)

        # fairness_diff_label('gender', gender_labels)
        fairness_diff_label('race', race_labels)
        fairness_diff_label('age', age_labels)

    elif args.mode == 'bias':
        '''
        # plot the total test accuracy
        def total_test_acc(model_name):
            total_test_acc = np.load(f'algorithms/{model_name}_test_accuracy_32_30_0.0001.npy', allow_pickle=True)
            mean_test_acc = np.mean(total_test_acc)
            std_test_acc = np.std(total_test_acc)

            n = len(total_test_acc)
            z = 1.96
            ci_lower = mean_test_acc - z * (std_test_acc / np.sqrt(n))
            ci_upper = mean_test_acc + z * (std_test_acc / np.sqrt(n))
            ci = (ci_lower, ci_upper)

            return mean_test_acc, ci

        mean_test_acc_baseline, ci_baseline = total_test_acc('Adam')
        mean_test_acc_VGG, ci_VGG = total_test_acc('VGG')
        test_acc_smote = 0.822
        test_acc_TL_FER = 0.767
        test_acc_TL2_FER = 0.783
        test_acc_TL_B_FER = 0.847

        labels = ['Standard', 'VGG', 'VGG+SMOTE', 'TL-A1', 'TL-A2', 'TL-B']
        mean_test_accs = [mean_test_acc_baseline, mean_test_acc_VGG, test_acc_smote, test_acc_TL_FER, test_acc_TL2_FER,
                          test_acc_TL_B_FER]
        ci_VGG_lower = ci_VGG[0]
        ci_VGG_upper = ci_VGG[1]  # 0.8231479926354925
        # ci_errors = [ci_baseline, ci_VGG, (0, 0), (0, 0)]
        # lower_errors = [mean - ci[0] for mean, ci in zip(mean_test_accs[:2], ci_errors[:2])]
        # upper_errors = [ci[1] - mean for mean, ci in zip(mean_test_accs[:2], ci_errors[:2])]
        # asymmetric_errors = [lower_errors, upper_errors]

        # plot the total test accuracy for the six models (four methods)
        def plot_testacc_four():
            fig, ax = plt.subplots(figsize=fig_size)
            # ax.errorbar(labels[:2], mean_test_accs[:2], yerr=asymmetric_errors, fmt='o', capsize=5,
            #             label='Confidence interval', color='blue')
            ax.bar(labels, mean_test_accs, alpha=0.7, label='Test accuracy')
            # ax.plot(labels, mean_test_accs, marker='o', linestyle='--', color='b', label='Test accuracy')
            ax.axhline(y=ci_VGG_lower, color='r', linestyle=':')
            ax.axhline(y=ci_VGG_upper, color='r', linestyle=':', label='95%-CI of VGG')
            ax.set_ylabel('Test accuracy')
            ax.set_title('Test accuracy for models', fontdict=title_font)
            ax.legend()
            ax.set_ylim(0.65, 0.9)
            plt.tight_layout()
            plt.savefig('results_figure/comparison_total_test_acc.pdf')
            plt.show()

        plot_testacc_four()
        '''

        # '''
        # calculate the test accuracy per emotion for the six models (four methods)
        def acc_per_emotion_normal(model_name):
            # Load and process the confusion matrices
            mean_cm = load_cm(model_name)
            # Get the diagonal elements of the normalized confusion matrix (accuracy per emotion)
            mean_diag = np.diag(normalize(mean_cm, 'true'))

            # Calculate standard deviation for diagonal elements
            confusion_matrices = np.load(f'algorithms/{model_name}_confusion_matrices_32_30_0.0001.npy', allow_pickle=True)
            diags = [np.diag(normalize(cm, 'true')) for cm in confusion_matrices]
            diags = np.array(diags)
            std_diag = np.std(diags, axis=0)

            # the number of trails
            n = 30
            # computing the 95% confidence intervals
            z = 1.96
            ci_lower = mean_diag - z * (std_diag / np.sqrt(n))
            ci_upper = mean_diag + z * (std_diag / np.sqrt(n))
            ci = (ci_lower, ci_upper)

            return mean_diag, ci

        def acc_per_emotion_advanced(model_name, datatype=None):
            if model_name == 'SMOTE':
                cm = np.load(f'algorithms/Smote_confusion_matrices_32_30_0.0001.npy')
            elif model_name in {'TL', 'TL2', 'TL_B'}:
                cm = np.load(f'algorithms/TL_{datatype}/{model_name}_confusion_matrices.npy')

            normalized_cm = normalize(cm, 'true')
            acc_peremo = np.diag(normalized_cm)

            return acc_peremo

        mean_acc_peremo_baseline, ci_baseline = acc_per_emotion_normal('Adam')
        mean_acc_peremo_VGG, ci_VGG = acc_per_emotion_normal('VGG')
        acc_peremo_smote = acc_per_emotion_advanced('SMOTE')
        acc_peremo_TL_FER = acc_per_emotion_advanced('TL', 'FER')
        acc_peremo_TL2_FER = acc_per_emotion_advanced('TL2', 'FER')
        acc_peremo_TL_B_FER = acc_per_emotion_advanced('TL_B', 'FER')

        # plot the accuracy per emotion using FER2013 dataset(in TL method)
        def plot_accuracy_per_emotion(mean_acc_peremo_baseline, ci_baseline, mean_acc_peremo_VGG, ci_VGG,
                                      acc_peremo_smote, acc_peremo_TL_FER, acc_peremo_TL2_FER, acc_peremo_TL_B_FER,
                                      labels):
            x = np.arange(len(labels))  # the label locations
            width = 0.13  # the width of the bars

            fig, ax = plt.subplots(figsize=(10, 6))

            rects1 = ax.bar(x - 2.5 * width, mean_acc_peremo_baseline, width,
                            yerr=np.abs(ci_baseline[0] - mean_acc_peremo_baseline), label='Standard', capsize=5)
            rects2 = ax.bar(x - 1.5 * width, mean_acc_peremo_VGG, width,
                            yerr=np.abs(ci_VGG[0] - mean_acc_peremo_VGG), label='VGG', capsize=5)
            rects3 = ax.bar(x - 0.5 * width, acc_peremo_smote, width, label='VGG+SMOTE')
            rects4 = ax.bar(x + 0.5 * width, acc_peremo_TL_FER, width, label='TL-A1')
            rects5 = ax.bar(x + 1.5 * width, acc_peremo_TL2_FER, width, label='TL-A2')
            rects6 = ax.bar(x + 2.5 * width, acc_peremo_TL_B_FER, width, label='TL-B')

            ax.set_ylabel('Accuracy score')
            ax.set_title('Accuracy per emotion for models', fontdict=title_font)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend(loc='lower right')
            ax.set_ylim(0.4, 1.0)
            fig.tight_layout()
            plt.savefig(f'results_figure/comparison_acc_per_emotion.pdf')
            plt.show()

        # plot_accuracy_per_emotion(mean_acc_peremo_baseline, ci_baseline, mean_acc_peremo_VGG, ci_VGG,
        #                           acc_peremo_smote, acc_peremo_TL_FER, acc_peremo_TL2_FER, acc_peremo_TL_B_FER,
        #                           emotion_labels)

        # plot the accuracy of VGG+TL-A1 and VGG+TL-A2 model per emotion using FER2013 dataset
        def plot_cata_forgetting(labels):
            x = np.arange(len(labels))  # the label locations
            width = 0.2  # the width of the bars
            fig, ax = plt.subplots(figsize=fig_size)
            colors = ['#ff7f0e', '#d62728', '#9467bd']

            rects1 = ax.bar(x - width, mean_acc_peremo_VGG,  width, yerr=np.abs(ci_VGG[0] - mean_acc_peremo_VGG),
                            label='VGG', capsize=5, color=colors[0])
            rects2 = ax.bar(x, acc_peremo_TL_FER, width, label='TL-A1', color=colors[1])
            rects3 = ax.bar(x + width, acc_peremo_TL2_FER, width, label='TL-A2', color=colors[2])

            ax.set_ylabel('Accuracy score')
            ax.set_title('Accuracy per emotion for models', fontdict=title_font)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend(loc='lower right')
            ax.set_ylim(0.4, 1.0)
            fig.tight_layout()
            plt.savefig('results_figure/comparison_cata_forgetting.pdf')
            plt.show()

        plot_cata_forgetting(emotion_labels)
        # '''

        '''
        # calculate the F1-scores for models
        # load the files of the standard CNN model or VGG model
        def f1_normal(model_name, name):
            total_f1 = []
            emotion_classes = range(1, 8)
            total_f1_peremo = {emotion: [] for emotion in emotion_classes}
            for i in range(30):
                data = pd.read_csv(f'prediction_files/{model_name}_csv/rafdb_total_pred_{name} {i}.csv')
                # only using test dataset
                test_data = data[data['Image'].str.contains('test')]
                true_labels = test_data['Emotion']
                predictions = test_data['Predicted']
                # calculate weighted average F1-score
                f1 = metrics.f1_score(true_labels, predictions, average='weighted')
                total_f1.append(f1)
                # calculate the f1-score per emotion
                report = metrics.classification_report(true_labels, predictions, output_dict=True,
                                                       labels=emotion_classes)
                for emotion in emotion_classes:
                    total_f1_peremo[emotion].append(report[str(emotion)]['f1-score'])

            n = len(total_f1)
            z = 1.96

            # get the mean and confidence interval of weighted average f1-score
            mean_f1 = np.mean(total_f1)
            std_f1 = np.std(total_f1)
            ci_lower = mean_f1 - z * (std_f1 / np.sqrt(n))
            ci_upper = mean_f1 + z * (std_f1 / np.sqrt(n))
            ci = (ci_lower, ci_upper)

            # get the mean and confidence interval of f1-scores per emotion
            mean_f1_peremo = {emotion: np.mean(total_f1_peremo[emotion]) for emotion in emotion_classes}
            std_f1_peremo = {emotion: np.std(total_f1_peremo[emotion]) for emotion in emotion_classes}
            ci_peremo = {}
            for emotion in emotion_classes:
                ci_lower = mean_f1_peremo[emotion] - z * (std_f1_peremo[emotion] / np.sqrt(n))
                ci_upper = mean_f1_peremo[emotion] + z * (std_f1_peremo[emotion] / np.sqrt(n))
                ci_peremo[emotion] = (ci_lower, ci_upper)

            return mean_f1, ci, mean_f1_peremo, ci_peremo

        # load the file of the VGG+SMOTE model or VGG+TL model
        def f1_advanced(model_name, datatype=None):
            emotion_classes = range(1, 8)
            if model_name == 'VGG+SMOTE':
                data = pd.read_csv(f'prediction_files/rafdb_total_pred_{model_name}.csv')
            else:
                data = pd.read_csv(f'prediction_files/rafdb_total_pred_VGG+{model_name}_{datatype}.csv')
            test_data = data[data['Image'].str.contains('test')]
            true_labels = test_data['Emotion']
            predictions = test_data['Predicted']

            # calculate the weighted average f1-score
            f1 = metrics.f1_score(true_labels, predictions, average='weighted')

            # calculate the f1-score per emotion
            report = metrics.classification_report(true_labels, predictions, output_dict=True, labels=emotion_classes)
            f1_peremo = {emotion: report[str(emotion)]['f1-score'] for emotion in emotion_classes}

            return f1, f1_peremo

        mean_f1_baseline, ci_baseline, mean_f1_baseline_classwise, ci_baseline_classwise = f1_normal('baseline', 'base')
        mean_f1_VGG, ci_VGG, mean_f1_VGG_classwise, ci_VGG_classwise = f1_normal('VGG', 'VGG')
        f1_smote, f1_smote_classwise = f1_advanced('VGG+SMOTE')
        f1_TL_FER, f1_TL_classwise_FER = f1_advanced('TL', 'FER')
        f1_TL2_FER, f1_TL2_classwise_FER = f1_advanced('TL2', 'FER')
        f1_TL_B_FER, f1_TL_B_classwise_FER = f1_advanced('TL_B', 'FER')

        ci_VGG_upper = ci_VGG[1]
        ci_VGG_lower = ci_VGG[0]

        # plot the weighted average f1-score for models
        def plot_f1():
            labels = ['Standard', 'VGG', 'VGG+SMOTE', 'TL-A1', 'TL-A2', 'TL-B']
            mean_f1_scores = [mean_f1_baseline, mean_f1_VGG, f1_smote, f1_TL_FER, f1_TL2_FER, f1_TL_B_FER]
            fig, ax = plt.subplots(figsize=fig_size)
            # ax.plot(labels, mean_f1_scores, marker='o', linestyle='--', color='b', label='F1-score')
            ax.bar(labels, mean_f1_scores, alpha=0.7, label='F1-score')
            ax.axhline(y=ci_VGG_lower, color='r', linestyle=':')
            ax.axhline(y=ci_VGG_upper, color='r', linestyle=':', label='95%-CI of VGG')
            ax.set_ylabel('F1-score')
            ax.set_title('Weighted average F1-score comparison of models', fontdict=title_font)
            ax.legend()
            ax.set_ylim(0.65, 0.9)
            fig.tight_layout()
            plt.savefig('results_figure/comparison_f1score.pdf')
            plt.show()

        # plot the f1-score per emotion for models
        def plot_f1_peremo():
            x = np.arange(7)
            width = 0.13

            fig, ax = plt.subplots(figsize=(10, 6))
            baseline_means = list(mean_f1_baseline_classwise.values())
            baseline_errors = [(mean - ci[0], ci[1] - mean) for mean, ci in
                               zip(baseline_means, ci_baseline_classwise.values())]
            baseline_errors = np.array(baseline_errors).T

            VGG_means = list(mean_f1_VGG_classwise.values())
            VGG_errors = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(VGG_means, ci_VGG_classwise.values())]
            VGG_errors = np.array(VGG_errors).T

            smote_means = list(f1_smote_classwise.values())
            TL_means = list(f1_TL_classwise_FER.values())
            TL2_means = list(f1_TL2_classwise_FER.values())
            TLB_means = list(f1_TL_B_classwise_FER.values())

            rects1 = ax.bar(x - 2.5 * width, baseline_means, width, label='Standard', yerr=baseline_errors, capsize=5)
            rects2 = ax.bar(x - 1.5 * width, VGG_means, width, label='VGG', yerr=VGG_errors, capsize=5)
            rects3 = ax.bar(x - 0.5 * width, smote_means, width, label='VGG+SMOTE')
            rects4 = ax.bar(x + 0.5 * width, TL_means, width, label='TL-A1')
            rects5 = ax.bar(x + 1.5 * width, TL2_means, width, label='TL-A2')
            rects6 = ax.bar(x + 2.5 * width, TLB_means, width, label='TL-B')

            ax.set_ylabel('F1-score')
            ax.set_title('F1-score per emotion for models', fontdict=title_font)
            ax.set_xticks(x)
            ax.set_xticklabels(emotion_labels)
            ax.legend(loc='lower right')
            ax.set_ylim(0.48, 0.95)
            fig.tight_layout()
            plt.savefig('results_figure/comparison_f1score_per_emotion.pdf')
            plt.show()

        plot_f1()
        plot_f1_peremo()
        '''

    elif args.mode == 'McNemar':
        '''
        # get the predicted file of VGG majority vote by 30 repetitions
        test_df = pd.read_csv('rafdb_test.csv')
        for i in range(30):
            pred_df = pd.read_csv(f'prediction_files/VGG_csv/rafdb_total_pred_VGG {i}.csv')
            test_df[f'predicted{i}'] = test_df['Image'].map(pred_df.set_index('Image')['Predicted'])

        predicted_columns = [f'predicted{i}' for i in range(30)]
        test_df['majpredicted'] = (test_df[predicted_columns].mode(axis=1)[0]).astype(int)
        test_df.to_csv('prediction_files/rafdb_test_pred_VGG_majority.csv', index=False)
        
        VGG_maj_df = pd.read_csv('prediction_files/rafdb_test_pred_VGG_majority.csv')
        test_image_names = VGG_maj_df['Image']
        VGG_maj_pre = VGG_maj_df['majpredicted']
        true_labels = VGG_maj_df['Emotion']
        '''
        # # choose the best-performing VGG model
        # total_test_accuracy = np.load('algorithms/VGG_test_accuracy_32_30_0.0001.npy', allow_pickle=True)
        # print(total_test_accuracy)
        # max_test_accuracy = max(total_test_accuracy)
        # max_index = list(total_test_accuracy).index(max_test_accuracy)
        # print('The index of best-performing VGG model is: ', max_index)  # 4
        # # load the prediction results of different models for RAF-DB dataset
        # VGG_df = pd.read_csv('prediction_files/VGG_csv/rafdb_total_pred_VGG 12.csv')
        # # get the predicted label of the best-performing VGG model
        # VGG_df = VGG_df.set_index('Image').loc[test_image_names].reset_index()
        # VGG_pre = VGG_df['Predicted']

        # get the names of test data and their true emotion labels
        test_data = pd.read_csv('rafdb_test.csv')
        test_image_names = test_data['Image']
        true_labels = test_data['Emotion']

        # load the prediction results of different models for RAF-DB dataset
        SMOTE_df = pd.read_csv('prediction_files/rafdb_total_pred_VGG+SMOTE.csv')
        TLB_FER_df = pd.read_csv('prediction_files/rafdb_total_pred_VGG+TL_B_FER.csv')
        TLB_SFEW_df = pd.read_csv('prediction_files/rafdb_total_pred_VGG+TL_B_SFEW.csv')

        # get the predicted label of SMOTE method
        SMOTE_df = SMOTE_df.set_index('Image').loc[test_image_names].reset_index()
        SMOTE_pre = SMOTE_df['Predicted']

        # get the predicted lable of TL-B method using FER2013 data
        TLB_FER_df = TLB_FER_df.set_index('Image').loc[test_image_names].reset_index()
        TLB_FER_pre = TLB_FER_df['Predicted']

        # get the predicted lable of TL-B method using SFEW data
        TLB_SFEW_df = TLB_SFEW_df.set_index('Image').loc[test_image_names].reset_index()
        TLB_SFEW_pre = TLB_SFEW_df['Predicted']

        # plot the 2*2 table for McNemar'test between VGG and VGG+SMOTE, VGG and VGG+TL-B
        # the implementation of VGG+TL-B model using two different dataset (FER2013 and SFEW)
        def McNemar_test(model1_pre, model2_pre, model1_name, model2_name):
            # create 2*2 confusion matrix
            n11 = np.sum((model1_pre == true_labels) & (model2_pre == true_labels))
            n01 = np.sum((model1_pre != true_labels) & (model2_pre == true_labels))
            n10 = np.sum((model1_pre == true_labels) & (model2_pre != true_labels))
            n00 = np.sum((model1_pre != true_labels) & (model2_pre != true_labels))

            conf_matrix = np.array([[n11, n10], [n01, n00]])

            # implement McNemar's test
            result = mcnemar(conf_matrix, exact=False, correction=True)
            print(f'{model1_name} vs {model2_name}')
            print(result)

            labels = np.array([["n11\n\n{}".format(n11), "n10\n\n{}".format(n10)],
                               ["n01\n\n{}".format(n01), "n00\n\n{}".format(n00)]])

            fig, ax = plt.subplots(figsize=fig_size)
            sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues', cbar=False, annot_kws={"size": 16})

            ax.set_xlabel(model2_name)
            ax.set_ylabel(model1_name)
            ax.xaxis.set_ticklabels(['Correct', 'Wrong'])
            ax.yaxis.set_ticklabels(['Correct', 'Wrong'])
            plt.title("McNemar's test table", fontdict=title_font)
            plt.savefig(f'results_figure/comparison_testcm_{model1_name} and {model2_name}.pdf')
            plt.show()

        McNemar_test(SMOTE_pre, TLB_FER_pre, 'VGG+SMOTE', 'VGG+TL-B with FER2013')
        McNemar_test(SMOTE_pre, TLB_SFEW_pre, 'VGG+SMOTE', 'VGG+TL-B with SFEW')
        McNemar_test(TLB_FER_pre, TLB_SFEW_pre, 'VGG+TL-B with FER2013', 'VGG+TL-B with SFEW')

    elif args.mode == 'twodataset':
        # plot the test accuracy
        def total_test_acc(model_name):
            total_test_acc = np.load(f'algorithms/{model_name}_test_accuracy_32_30_0.0001.npy', allow_pickle=True)
            mean_test_acc = np.mean(total_test_acc)
            std_test_acc = np.std(total_test_acc)

            n = len(total_test_acc)
            z = 1.96
            ci_lower = mean_test_acc - z * (std_test_acc / np.sqrt(n))
            ci_upper = mean_test_acc + z * (std_test_acc / np.sqrt(n))
            ci = (ci_lower, ci_upper)

            return mean_test_acc, ci
        mean_test_acc_VGG, ci_VGG = total_test_acc('VGG')
        ci_VGG_lower = ci_VGG[0]
        ci_VGG_upper = ci_VGG[1]

        # test accuracy of different models(VGG+TL) by using different dataset(FER and SFEW)
        test_acc_TL_FER = 0.767
        test_acc_TL2_FER = 0.783
        test_acc_TL_B_FER = 0.847
        test_acc_TL_SFEW = 0.771
        test_acc_TL2_SFEW = 0.794
        test_acc_TL_B_SFEW = 0.844

        labels = ['TL-A1', 'TL-A2', 'TL-B']

        # plot the comparison of test accuracy
        def compared_twodata():
            fer_data = [test_acc_TL_FER, test_acc_TL2_FER, test_acc_TL_B_FER]
            sfew_data = [test_acc_TL_SFEW, test_acc_TL2_SFEW, test_acc_TL_B_SFEW]

            x = np.arange(len(labels))  # the label locations
            width = 0.35  # the width of the bars

            fig, ax = plt.subplots(figsize=fig_size)

            rects1 = ax.bar(x - width / 2, fer_data, width, alpha=0.7, label='FER', color='gold')
            rects2 = ax.bar(x + width / 2, sfew_data, width, alpha=0.7, label='SFEW', color='lightskyblue')
            ax.axhline(y=mean_test_acc_VGG, color='k', linestyle='--', label='mean of VGG')
            ax.axhline(y=ci_VGG_lower, color='r', linestyle=':')
            ax.axhline(y=ci_VGG_upper, color='r', linestyle=':', label='95%-CI of VGG')

            ax.set_ylabel('Test accuracy')
            ax.set_title('Test accuracy by models and dataset', fontdict=title_font)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()
            ax.set_ylim(0.7, 0.9)
            fig.tight_layout()
            plt.savefig('results_figure/comparison_test_acc_twodata.pdf')
            plt.show()

        compared_twodata()

        # get the confusion matrix of VGG, VGG+TL-A1, VGG+TL-A2, VGG+TL-B under different dataset(FER and SFEW)
        def acc_per_emotion_normal(model_name):
            # Load and process the confusion matrices
            mean_cm = load_cm(model_name)
            # Get the diagonal elements of the normalized confusion matrix (accuracy per emotion)
            mean_diag = np.diag(normalize(mean_cm, 'true'))

            # Calculate standard deviation for diagonal elements
            confusion_matrices = np.load(f'algorithms/{model_name}_confusion_matrices_32_30_0.0001.npy',
                                         allow_pickle=True)
            diags = [np.diag(normalize(cm, 'true')) for cm in confusion_matrices]
            diags = np.array(diags)
            std_diag = np.std(diags, axis=0)

            # the number of trails
            n = 30
            # computing the 95% confidence intervals
            z = 1.96
            ci_lower = mean_diag - z * (std_diag / np.sqrt(n))
            ci_upper = mean_diag + z * (std_diag / np.sqrt(n))
            ci = (ci_lower, ci_upper)

            return mean_diag, ci
        def acc_per_emotion_advanced(model_name, datatype):
            cm = np.load(f'algorithms/TL_{datatype}/{model_name}_confusion_matrices.npy')

            normalized_cm = normalize(cm, 'true')
            acc_peremo = np.diag(normalized_cm)

            return acc_peremo

        mean_acc_peremo_VGG, ci_VGG = acc_per_emotion_normal('VGG')
        acc_peremo_TL_FER = acc_per_emotion_advanced('TL', 'FER')
        acc_peremo_TL2_FER = acc_per_emotion_advanced('TL2', 'FER')
        acc_peremo_TL_B_FER = acc_per_emotion_advanced('TL_B', 'FER')
        acc_peremo_TL_SFEW = acc_per_emotion_advanced('TL', 'SFEW')
        acc_peremo_TL2_SFEW = acc_per_emotion_advanced('TL2', 'SFEW')
        acc_peremo_TL_B_SFEW = acc_per_emotion_advanced('TL_B', 'SFEW')

        # plot the comparison of accuracy per emotion between different models and dataset
        def plot_accuracy_per_emotion(model_name, vgg_data, model_fer_data, model_sfew_data, emotion_labels, title):
            x = np.arange(len(emotion_labels))  # the label locations
            width = 0.2  # the width of the bars

            fig, ax = plt.subplots(figsize=(10, 6))

            rects1 = ax.bar(x - width, vgg_data, width, yerr=np.abs(ci_VGG[0] - vgg_data),
                            label='VGG model', capsize=5, color='#ff7f0e')
            rects2 = ax.bar(x, model_fer_data, width, alpha=0.7, label=f'{title} with FER', color='gold')
            rects3 = ax.bar(x + width, model_sfew_data, width, alpha=0.7, label=f'{title} with SFEW', color='lightskyblue')

            ax.set_ylabel('Accuracy score')
            ax.set_title(f'Accuracy per emotion for {title} model using different dataset', fontdict=title_font)
            ax.set_xticks(x)
            ax.set_xticklabels(emotion_labels)
            ax.legend(loc='upper left')
            ax.set_ylim(0.4, 1.0)
            fig.tight_layout()
            plt.savefig(f'results_figure/comparison_acc_per_emotion_{model_name}_twodata.pdf')
            plt.show()

        plot_accuracy_per_emotion('TL', mean_acc_peremo_VGG, acc_peremo_TL_FER, acc_peremo_TL_SFEW, emotion_labels, title='VGG+TL-A1')
        plot_accuracy_per_emotion('TL2', mean_acc_peremo_VGG, acc_peremo_TL2_FER, acc_peremo_TL2_SFEW, emotion_labels, title='VGG+TL-A2')
        plot_accuracy_per_emotion('TL_B', mean_acc_peremo_VGG, acc_peremo_TL_B_FER, acc_peremo_TL_B_SFEW,
                                  emotion_labels, title='VGG+TL-B')




