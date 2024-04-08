import json
import os
import random

import numpy as np
import torch
import torchvision


class AppDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root,
            data_name,
            images_subdir,
            max_image_stack_size=8,
            device=None,
            seed=None,
            verbose=False
        ):
        self.max_image_stack_size = max_image_stack_size

        seed = seed or random.randint(0, 2 ** 32 - 1)
        self.rnd_state = np.random.RandomState(seed)

        self.device = device
        self.verbose = verbose

        if self.verbose:
            print("Loading datafile...")

        self._load_datafile(root, data_name, images_subdir)

    def _load_datafile(self, root, data_name, images_subdir):
        data_json = json.load(open(os.path.join(root, data_name), "r", encoding="utf-8"))
        dataset_raw = {
            app["appId"]: {"image_paths": [], "genreId": app["genreId"]}
            for app in data_json
        }

        # Add image paths
        for image in os.listdir(os.path.join(root, images_subdir)):
            app_id = "_".join(image.split("_")[:-1])
            if app_id in dataset_raw:
                dataset_raw[app_id]["image_paths"].append(os.path.join(root, images_subdir, image))
            else:
                if self.verbose:
                    print("Not found", image)

        self.image_paths = []
        self.classes = []

        # Extract image paths and descriptions into separate lists
        for app_id, app in dataset_raw.items():
            if len(app["image_paths"]) > 0:
                self.image_paths.append(app["image_paths"])
                self.classes.append(app["genreId"])
            else:
                if self.verbose:
                    print("No images for", app_id)

        class_names = list(set(self.classes))

        self.label_map = {class_name: i for i, class_name in enumerate(class_names)}
        self.labels = [self.label_map[class_name] for class_name in self.classes]


    def __len__(self):
        return len(self.image_paths)

    def _generate_probabilities(self, count):
        """
        [1/2, 1/4, 1/8, ... 1/2^(count - 1), 1/2^(count - 1)]
        or [1]
        """

        probabilities = [1 / 2 ** (i + 1) for i in range(count)]
        if len(probabilities) > 1:
            probabilities[-1] = probabilities[-2]
        else:
            probabilities = [1]

        return probabilities

    def __getitem__(self, idx):
        probabilites = self._generate_probabilities(len(self.image_paths[idx]))
        generating_count = min(self.max_image_stack_size, len(self.image_paths[idx]))
        
        # Random sample from image paths based on probabilities
        # Choose maximum max_image_stack_size images
        # If there are less images, return only those
        image_paths = self.rnd_state.choice(
            self.image_paths[idx],
            size=generating_count,
            replace=False,
            p=probabilites
        )

        # Load images
        images = []
        for path in image_paths:
            image = torchvision.io.read_image(
                path,
                torchvision.io.ImageReadMode.RGB
            )
            images.append(image)

        label = self.labels[idx]

        return images, label


class AppDataLoader(torch.utils.data.DataLoader):
    def __init__(
            self,
            dataset,
            batch_size,
            shuffle=True,
            image_size=448,
            processor=None,
            device=None
        ):
        self.image_size = image_size
        self.processor = processor

        self.device = device

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch):
        images, labels = zip(*list(batch))

        if self.processor is not None:
            images = self._image_preprocess(images)
            images = images.to(self.device)

        labels = torch.tensor(labels)
        labels = labels.to(self.device)

        return images, labels

    def _image_preprocess(self, images):
        """Preprocess images using the processor.
        
        Receives a 2D list of images (5D list):
        [
            [image1, image2, image3, ...],
            [image1, image2, image3, ...],
            ...
        ]
        
        Returns a tensor of tensors of batched images (5D tensor):
        [
            [image0_0, image1_0, image2_0, ...],
            [image0_1, image1_1, image2_1, ...],
            ...
        ]
        """

        preprocessed_images = []

        for image_stack in images:
            preprocessed_stack = self.processor(
                image_stack,
                return_tensors="pt"
            ).pixel_values

            preprocessed_images.append(preprocessed_stack)

        images = torch.stack(preprocessed_images, dim=1)

        return images
