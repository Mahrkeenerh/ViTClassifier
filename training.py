import json
import os

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import app_dataset


class Callback:
    def __init__(self, **kwargs):
        self.run_name = kwargs.get("run_name", None)

    def start(self, **kwargs):
        pass

    def end(self, **kwargs):
        pass

    def before_train_epoch(self, **kwargs):
        pass

    def after_train_epoch(self, **kwargs):
        pass

    def after_val_epoch(self, **kwargs):
        pass


class Callbacks:
    def __init__(self, callbacks, **kwargs):
        self.callbacks = callbacks

    def start(self, **kwargs):
        for callback in self.callbacks:
            callback.start(**kwargs)

    def end(self, **kwargs):
        for callback in self.callbacks:
            callback.end(**kwargs)

    def before_train_epoch(self, **kwargs):
        for callback in self.callbacks:
            callback.before_train_epoch(**kwargs)

    def after_train_epoch(self, **kwargs):
        for callback in self.callbacks:
            callback.after_train_epoch(**kwargs)

    def after_val_epoch(self, **kwargs):
        for callback in self.callbacks:
            callback.after_val_epoch(**kwargs)


class SaveCallback(Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.target_dir = "models"
        if self.run_name is not None:
            self.target_dir = os.path.join(self.run_name, self.target_dir)

        os.makedirs(self.target_dir, exist_ok=True)

    def start(self, **kwargs):
        kwargs["model"].save_pretrained(
            os.path.join(self.target_dir, "vit_0")
        )

    def after_train_epoch(self, **kwargs):
        kwargs["model"].save_pretrained(
            os.path.join(self.target_dir, f"vit_{kwargs['epoch'] + 1}")
        )


class LogCallback(Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.target_dir = "logs"
        if self.run_name is not None:
            self.target_dir = os.path.join(self.run_name, self.target_dir)

        os.makedirs(self.target_dir, exist_ok=True)

        self.train_losses = []
        self.val_losses = []
        self.accuracy = []
        self.lrs = []
        self.correct_classes = []
        self.total_classes = []

    def before_train_epoch(self, **kwargs):
        self.lrs.append(kwargs["lr"])

    def after_train_epoch(self, **kwargs):
        self.train_losses.append(kwargs["train_loss"])

    def after_val_epoch(self, **kwargs):
        self.val_losses.append(kwargs["val_loss"])
        self.accuracy.append(kwargs["accuracy"])

        self.correct_classes.append(kwargs["correct_classes"])
        self.total_classes.append(kwargs["total_classes"])

        self.end()

    def end(self, **kwargs):
        with open(
            os.path.join(self.target_dir, "loss.json"),
            "w"
        ) as f:
            json.dump(
                {
                    "train": self.train_losses,
                    "val": self.val_losses,
                    "accuracy": self.accuracy,
                    "lrs": self.lrs
                },
                f,
                indent=4
            )

        # Create plots
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Train loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and validation loss")
        plt.savefig(os.path.join(self.target_dir, "loss.png"))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(self.accuracy, label="Accuracy")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation accuracy")
        plt.savefig(os.path.join(self.target_dir, "accuracy.png"))
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(self.lrs, label="Learning rate")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.title("Learning rate")
        plt.savefig(os.path.join(self.target_dir, "lr.png"))
        plt.close()

        # Plot correct histogram
        plt.figure(figsize=(10, 5))
        plt.bar(range(17), self.correct_classes[-1], label="Correct")
        plt.bar(range(17), self.total_classes[-1], label="Total", alpha=0.5)
        plt.legend()
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title("Correct predictions by class")
        plt.savefig(os.path.join(self.target_dir, "correct.png"))
        plt.close()

        # Plot correct histogram scaled to 0-1
        plt.figure(figsize=(10, 5))
        plt.bar(range(17), [c / t for c, t in zip(self.correct_classes[-1], self.total_classes[-1])], label="Correct")
        plt.legend()
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.title("Correct predictions by class scaled to 0-1")
        plt.savefig(os.path.join(self.target_dir, "correct_scaled.png"))
        plt.close()

        # Plot accuracy by class
        plt.figure(figsize=(10, 5))
        for i in range(17):
            plt.plot(
                [c[i] / t[i] for c, t in zip(self.correct_classes, self.total_classes)],
                label=f"Class {i}"
            )
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation accuracy by class")
        plt.savefig(os.path.join(self.target_dir, "accuracy_by_class.png"))
        plt.close()


def load_data(
    processor,
    embeds_name,
    data_config,
    device,
    seed,
):
    dataset = app_dataset.AppDataset(
        root=data_config["root"],
        data_name=data_config["data_name"],
        images_subdir="images",
        embeds_name=embeds_name,
        image_stack_size=data_config["image_stack_size"],
        device=device,
        seed=seed
    )

    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [data_config["data_split"], 1 - data_config["data_split"]],
        generator=torch.Generator().manual_seed(seed) if seed is not None else None
    )

    if embeds_name:
        processor = None

    train_loader = app_dataset.AppDataLoader(
        train_ds,
        batch_size=data_config["minibatch_size"],
        shuffle=seed is None,
        processor=processor,
        image_size=data_config["image_size"],
        device=device
    )

    val_loader = app_dataset.AppDataLoader(
        val_ds,
        batch_size=data_config["minibatch_size"],
        shuffle=False,
        processor=processor,
        image_size=data_config["image_size"],
        device=device
    )

    return train_loader, val_loader


def train(
    vit_classifier,
    optimizer,
    scheduler,
    train_loader, 
    val_loader,
    epochs,
    callbacks
):
    def get_rolling_loss(rolling_loss, loss):
        if rolling_loss == 0:
            rolling_loss = loss.item()
        else:
            rolling_loss = rolling_loss * 0.99 + loss.item() * 0.01

        return rolling_loss

    rolling_loss = 0

    if not isinstance(callbacks, Callbacks):
        callbacks = Callbacks(callbacks)

    callbacks.start(model=vit_classifier)

    for epoch in range(epochs):
        optimizer.zero_grad()

        callbacks.before_train_epoch(lr=scheduler.get_last_lr()[0])

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} - Training, Loss: -"
        )
        for data in train_bar:
            image_batches, labels = data

            loss = vit_classifier.loss(image_batches, labels).mean()
            rolling_loss = get_rolling_loss(rolling_loss, loss)

            train_bar.set_description(
                f"Epoch {epoch + 1}/{epochs} - Training, Loss: {rolling_loss:.4f}"
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        callbacks.after_train_epoch(
            model=vit_classifier,
            epoch=epoch,
            train_loss=rolling_loss
        )

        with torch.no_grad():
            total_loss = 0
            correct = 0
            total = 0
            correct_classes = [0] * 17
            total_classes = [0] * 17

            val_bar = tqdm(
                val_loader,
                desc=f"Validation, Accuracy: -"
            )
            for data in val_bar:
                image_batches, labels = data

                output = vit_classifier(image_batches)
                loss = vit_classifier.loss(x=output, y=labels).mean()
                total_loss += loss.item()

                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for i in range(len(labels)):
                    total_classes[labels[i]] += 1
                    if labels[i] == predicted[i]:
                        correct_classes[labels[i]] += 1

                val_bar.set_description(
                    f" - Validation, Accuracy: {correct / total:.4f}"
                )

            callbacks.after_val_epoch(
                val_loss=total_loss / len(val_loader),
                accuracy=correct / total,
                correct_classes=correct_classes,
                total_classes=total_classes
            )

    callbacks.end()
