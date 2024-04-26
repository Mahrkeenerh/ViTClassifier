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
            embeds_name,
            image_stack_size=8,
            device=None,
            seed=None,
            verbose=False
        ):
        self.image_stack_size = image_stack_size

        seed = seed or random.randint(0, 2 ** 32 - 1)
        self.rnd_state = np.random.RandomState(seed)

        self.device = device
        self.verbose = verbose

        if self.verbose:
            print("Loading datafile...")

        self._load_datafile(root, data_name, images_subdir, embeds_name)

    def _load_datafile(self, root, data_name, images_subdir, embeds_name):
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

        self.embeds = None
        if embeds_name:
            self.embeds = torch.load(os.path.join(root, embeds_name))

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

    def _load_images(self, idx, probabilites):
        image_paths = []

        # Keep randomly adding images - if less, all will be included at least once
        while len(image_paths) < self.image_stack_size:
            generating_count = min(self.image_stack_size - len(image_paths), len(self.image_paths[idx]))
            image_paths.extend(self.rnd_state.choice(
                self.image_paths[idx],
                size=generating_count,
                replace=False,
                p=probabilites
            ))

        # Load images
        images = []
        for path in image_paths:
            image = torchvision.io.read_image(
                path,
                torchvision.io.ImageReadMode.RGB
            )
            images.append(image)

        return images

    def _load_embeds(self, idx, probabilites):
        embed_ids = []

        # Keep randomly adding images - if less, all will be included at least once
        while len(embed_ids) < self.image_stack_size:
            generating_count = min(self.image_stack_size - len(embed_ids), len(self.image_paths[idx]))
            embed_ids.extend(self.rnd_state.choice(
                range(len(self.image_paths[idx])),
                size=generating_count,
                replace=False,
                p=probabilites
            ))

        embeds = [self.embeds[idx][embed_id] for embed_id in embed_ids]
        return embeds

    def __getitem__(self, idx):
        probabilites = self._generate_probabilities(len(self.image_paths[idx]))

        if self.embeds is None:
            images = self._load_images(idx, probabilites)
        else:
            embeds = self._load_embeds(idx, probabilites)
            images = torch.stack(embeds).to(self.device)

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
        else:
            images = torch.stack(images).to(self.device)

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
