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
            os.path.join(self.target_dir, "vit_0.pt")
        )

    def after_train_epoch(self, **kwargs):
        kwargs["model"].save_pretrained(
            os.path.join(self.target_dir, f"vit_{kwargs['epoch'] + 1}.pt")
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
        self.accuracy_1 = []
        self.accuracy_3 = []
        self.accuracy_5 = []
        self.lrs = []
        self.correct_classes = []
        self.predicted_classes = []
        self.correct_class_counts_1 = []
        self.correct_class_counts_3 = []
        self.correct_class_counts_5 = []
        self.predicted_class_counts = []
        self.total_classes = []

    def start(self, **kwargs):
        self.labels = [None] * 17
        for label, i in kwargs["label_map"].items():
            self.labels[i] = label

    def before_train_epoch(self, **kwargs):
        self.lrs.append(kwargs["lr"])

    def after_train_epoch(self, **kwargs):
        self.train_losses.append(kwargs["train_loss"])

    def after_val_epoch(self, **kwargs):
        self.val_losses.append(kwargs["val_loss"])
        self.accuracy_1.append(kwargs["accuracy_1"])
        self.accuracy_3.append(kwargs["accuracy_3"])
        self.accuracy_5.append(kwargs["accuracy_5"])

        self.correct_classes.append(kwargs["correct_classes"])
        self.predicted_classes.append(kwargs["predicted_classes"])

        self.correct_class_counts_1.append(kwargs["correct_class_counts_1"])
        self.correct_class_counts_3.append(kwargs["correct_class_counts_3"])
        self.correct_class_counts_5.append(kwargs["correct_class_counts_5"])
        self.predicted_class_counts.append(kwargs["predicted_class_counts"])
        self.total_classes.append(kwargs["total_classes"])

        self.end()

    def end(self, **kwargs):
        with open(
            os.path.join(self.target_dir, "dump.json"),
            "w"
        ) as f:
            json.dump(
                {
                    "train": self.train_losses,
                    "val": self.val_losses,
                    "accuracy_1": self.accuracy_1,
                    "accuracy_3": self.accuracy_3,
                    "accuracy_5": self.accuracy_5,
                    "correct_classes_1": self.correct_class_counts_1,
                    "correct_classes_3": self.correct_class_counts_3,
                    "correct_classes_5": self.correct_class_counts_5,
                    "predicted_classes": self.predicted_class_counts,
                    "total_classes": self.total_classes,
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
        plt.plot(self.accuracy_1, label="Accuracy top 1")
        plt.plot(self.accuracy_3, label="Accuracy top 3")
        plt.plot(self.accuracy_5, label="Accuracy top 5")
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
        plt.bar(
            [self.labels[i] for i in range(17)],
            self.total_classes[-1],
            label="Total"
        )
        plt.bar(
            [self.labels[i] for i in range(17)],
            self.correct_class_counts_5[-1],
            label="Correct top 5"
        )
        plt.bar(
            [self.labels[i] for i in range(17)],
            self.correct_class_counts_3[-1],
            label="Correct top 3"
        )
        plt.bar(
            [self.labels[i] for i in range(17)],
            self.correct_class_counts_1[-1],
            label="Correct top 1"
        )
        plt.legend()
        plt.xlabel("Class")
        plt.xticks(rotation=75)
        plt.ylabel("Count")
        plt.title("Correct predictions by class")
        plt.tight_layout()
        plt.savefig(os.path.join(self.target_dir, "correct.png"))
        plt.close()

        # Plot correct histogram scaled to 0-1
        plt.figure(figsize=(10, 5))
        plt.bar(
            [self.labels[i] for i in range(17)],
            [c / t for c, t in zip(self.correct_class_counts_5[-1], self.total_classes[-1])],
            label="Correct top 5"
        )
        plt.bar(
            [self.labels[i] for i in range(17)],
            [c / t for c, t in zip(self.correct_class_counts_3[-1], self.total_classes[-1])],
            label="Correct top 3"
        )
        plt.bar(
            [self.labels[i] for i in range(17)],
            [c / t for c, t in zip(self.correct_class_counts_1[-1], self.total_classes[-1])],
            label="Correct top 1"
        )
        plt.legend()
        plt.xlabel("Class")
        plt.xticks(rotation=75)
        plt.ylabel("Count")
        plt.title("Correct predictions by class scaled to 0-1")
        plt.tight_layout()
        plt.savefig(os.path.join(self.target_dir, "correct_scaled.png"))
        plt.close()

        # Plot predicted histogram
        plt.figure(figsize=(10, 5))
        plt.bar(
            [self.labels[i] for i in range(17)],
            self.predicted_class_counts[-1],
            label="Predicted"
        )
        plt.legend()
        plt.xlabel("Class")
        plt.xticks(rotation=75)
        plt.ylabel("Count")
        plt.title("Predicted classes")
        plt.tight_layout()
        plt.savefig(os.path.join(self.target_dir, "predicted.png"))
        plt.close()

        # Plot accuracy by class
        plt.figure(figsize=(10, 5))
        for i in range(17):
            plt.plot(
                [c[i] / t[i] for c, t in zip(self.correct_class_counts_1, self.total_classes)],
                label=self.labels[i]
            )
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation accuracy by class")
        plt.savefig(os.path.join(self.target_dir, "accuracy_by_class.png"))
        plt.close()

        # Confusion matrix
        confusion_matrix = torch.zeros(17, 17)
        for correct_classes, predicted_classes in zip(
            self.correct_classes[-1], self.predicted_classes[-1]
        ):
            confusion_matrix[correct_classes, predicted_classes] += 1

        # Normalize by rows
        confusion_matrix = confusion_matrix / confusion_matrix.sum(1).view(-1, 1)

        plt.figure(figsize=(15, 15))
        plt.imshow(confusion_matrix, cmap="Blues")
        # Add numbers to the cells
        for i in range(17):
            for j in range(17):
                plt.text(j, i, f"{confusion_matrix[i, j]:.2f}", ha="center", va="center", color="black")
        plt.colorbar()
        plt.xticks(range(17), [self.labels[i] for i in range(17)], rotation=75)
        plt.yticks(range(17), [self.labels[i] for i in range(17)])
        plt.xlabel("Predicted")
        plt.ylabel("Correct")
        plt.title("Confusion matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.target_dir, "confusion_matrix.png"))
        plt.close()


def load_data(
    processor,
    embeds_name,
    data_config,
    device,
    seed,
):
    dataset = app_dataset.AppDataset(
        root="/home/xbuban1/Games",
        data_name=f'apps_{data_config["dataset"]}.json',
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

    return train_loader, val_loader, dataset


def train(
    vit_classifier,
    optimizer,
    scheduler,
    train_loader, 
    val_loader,
    dataset,
    epochs,
    grad_accum_steps,
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

    callbacks.start(
        model=vit_classifier,
        label_map=dataset.label_map
    )

    for epoch in range(epochs):
        optimizer.zero_grad()
        grad_accum_step = 0

        # vit_classifier.train()
        callbacks.before_train_epoch(lr=scheduler.get_last_lr()[0])

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} - Training, Loss: -"
        )
        for data in train_bar:
            image_batches, labels = data

            loss = vit_classifier.loss(image_batches, labels).mean() / grad_accum_steps
            rolling_loss = get_rolling_loss(rolling_loss, loss * grad_accum_steps)

            train_bar.set_description(
                f"Epoch {epoch + 1}/{epochs} - Training, Loss: {rolling_loss:.4f}"
            )

            loss.backward()

            grad_accum_step += 1
            if grad_accum_step == grad_accum_steps:
                grad_accum_step = 0
                optimizer.step()
                optimizer.zero_grad()

        if grad_accum_step > 0:
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        callbacks.after_train_epoch(
            model=vit_classifier,
            epoch=epoch,
            train_loss=rolling_loss
        )
        # vit_classifier.eval()

        with torch.no_grad():
            total_loss = 0
            correct_1 = 0
            correct_3 = 0
            correct_5 = 0
            total = 0
            correct_classes = []
            predicted_classes = []
            correct_class_counts_1 = [0] * 17
            correct_class_counts_3 = [0] * 17
            correct_class_counts_5 = [0] * 17
            predicted_class_counts = [0] * 17
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

                correct_classes.extend(labels.tolist())
                predicted_classes.extend(predicted.tolist())

                total += labels.size(0)
                correct_1 += (predicted == labels).sum().item()
                _, predicted_3 = torch.topk(output, 3)
                correct_3 += sum([1 for i in range(len(labels)) if labels[i] in predicted_3[i]])
                _, predicted_5 = torch.topk(output, 5)
                correct_5 += sum([1 for i in range(len(labels)) if labels[i] in predicted_5[i]])

                for i in range(len(labels)):
                    total_classes[labels[i]] += 1
                    predicted_class_counts[predicted[i]] += 1
                    if labels[i] == predicted[i]:
                        correct_class_counts_1[labels[i]] += 1
                    if labels[i] in predicted_3[i]:
                        correct_class_counts_3[labels[i]] += 1
                    if labels[i] in predicted_5[i]:
                        correct_class_counts_5[labels[i]] += 1

                val_bar.set_description(
                    f" - Validation, Accuracy: {correct_1 / total:.4f}"
                )

            callbacks.after_val_epoch(
                val_loss=total_loss / len(val_loader),
                accuracy_1=correct_1 / total,
                accuracy_3=correct_3 / total,
                accuracy_5=correct_5 / total,
                correct_classes=correct_classes,
                predicted_classes=predicted_classes,
                correct_class_counts_1=correct_class_counts_1,
                correct_class_counts_3=correct_class_counts_3,
                correct_class_counts_5=correct_class_counts_5,
                predicted_class_counts=predicted_class_counts,
                total_classes=total_classes
            )

    callbacks.end()
