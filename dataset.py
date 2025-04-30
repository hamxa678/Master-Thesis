import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
from torchvision import transforms
from glob import glob
import torch


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def init_datasets(ROOT_DIR, args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)), 
    ])
    
    training_dataset = BOSCHDATASET(root_dir=f'{ROOT_DIR}/Clean data', train=True, transform=transform, random_crop=False)
    testing_dataset = BOSCHDATASET(root_dir=f'{ROOT_DIR}/Clean data', train=False, transform=transform, random_crop=False)

    return training_dataset, testing_dataset


def init_dataset_loader(dataset, args, shuffle=True):
    return cycle(DataLoader(
        dataset,
        batch_size=args['Batch_Size'],
        shuffle=shuffle,
        num_workers= 2,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True
    ))

class BOSCHDATASET(Dataset):
    def __init__(self, root_dir, transform=None, img_size=(256, 256), train=True, random_crop=True, anomalous=False):
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.random_crop = random_crop
        self.anomalous = anomalous
        if train and not anomalous:
            root_dir = os.path.join(root_dir, 'Train')
        elif not train and not anomalous:    
            root_dir = os.path.join(root_dir, 'Test')
        elif anomalous:
            root_dir = os.path.join(root_dir, 'Anomolous data')

        self.filenames = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        
        image = Image.open(img_name).convert('L')

        if self.anomalous:
            mask_name = img_name[:-4] + '.png'
            mask = Image.open(mask_name).convert('L')

        if self.transform:
            image = self.transform(image)
            if self.anomalous:
                mask = self.transform(mask)
        if self.anomalous:
            sample = {'image': image, 'mask': mask, 'filename': img_name}
        else:
            sample = {'image': image, 'filename': img_name}

        return sample





class Dataset_DA_DET(Dataset):
    def __init__(self, root, config, is_train=True):
        self.image_transform = transforms.Compose(
            [   
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((config.data.image_size, config.data.image_size)),  
                transforms.ToTensor(),
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((config.data.image_size, config.data.image_size)),
                transforms.ToTensor(), # Scales data into [0,1] 
            ]
        )
        self.config = config
        self.is_train = is_train
        if is_train:
            self.image_files = glob(
                os.path.join(root,"Clean data", "Train", "*.jpg")
            )
        else:
            self.image_files = []
            test_images = glob(os.path.join(root, "Clean data", "Test", "*.jpg"))
            anom_images = glob(os.path.join(root, "Anomolous data", "*.jpg"))[:10]

            self.image_files.extend(test_images)
            self.image_files.extend(anom_images)
        

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        # image = self.image_transform(image)
        image = self.image_transform(image)
        
        if self.is_train:
            label = 'good'
            return image, label
        else:
            image_dir = os.path.dirname(image_file)
            if "Anomolous data" in image_dir:
                label = "defective"
                if self.config.data.mask:
                    mask_file = image_file.replace(".jpg", ".png")
                    if os.path.exists(mask_file):
                        mask = Image.open(mask_file)
                        target = self.mask_transform(mask)
            else:
                label = "good"
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            
            return image, target, label, image_file

    def __len__(self):
        return len(self.image_files)
