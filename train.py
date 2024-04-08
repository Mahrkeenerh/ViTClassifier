import os

from peft import LoraConfig
import torch
from tqdm import tqdm

import app_dataset
import modeling


def load_simple_classifier(image_size, device):
    vit_model, vit_processor = modeling.load_VED_vit(
        model_path="/home/xbuban1/ved_model",
        image_size=image_size,
        device=device
    )

    classifier = modeling.SimpleClassifier(
        input_dim=vit_model.embeddings.patch_embeddings.projection.out_channels,
        output_dim=17
    ).to(device)

    vit_classifier = modeling.ViTClassifier(
        vit=vit_model,
        classifier=classifier
    )

    return vit_classifier, vit_processor


def load_data(
    processor,
    image_size,
    max_image_stack_size,
    minibatch_size,
    data_split,
    device,
    seed,
):
    dataset = app_dataset.AppDataset(
        root="/home/xbuban1/Games",
        data_name="apps_filtered.json",
        images_subdir="images",
        max_image_stack_size=max_image_stack_size,
        device=device,
        seed=seed
    )

    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [data_split, 1 - data_split],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = app_dataset.AppDataLoader(
        train_ds,
        batch_size=minibatch_size,
        shuffle=False,
        processor=processor,
        image_size=image_size,
        device=device
    )

    val_loader = app_dataset.AppDataLoader(
        val_ds,
        batch_size=minibatch_size,
        shuffle=False,
        processor=processor,
        image_size=image_size,
        device=device
    )

    return train_loader, val_loader


def train(
    vit_classifier,
    optimizer,
    scheduler,
    train_loader, 
    val_loader,
    epochs
):
    def get_rolling_loss(rolling_loss, loss):
        if rolling_loss == 0:
            rolling_loss = loss.item()
        else:
            rolling_loss = rolling_loss * 0.99 + loss.item() * 0.01

        return rolling_loss

    rolling_loss = 0

    vit_classifier.save_pretrained("vit_classifier_0")

    for epoch in range(epochs):
        optimizer.zero_grad()

        print(f'Learing rate: {scheduler.get_last_lr()[0]:.6f}')

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} - Training, Rolling loss: - | Loss: -"
        )
        for data in train_bar:
            image_batches, labels = data

            loss = vit_classifier.loss(image_batches, labels).mean()
            rolling_loss = get_rolling_loss(rolling_loss, loss)

            train_bar.set_description(
                f"Epoch {epoch + 1}/{epochs} - Training, Rolling loss: {rolling_loss:.4f} | Loss: {loss:.4f}"
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        vit_classifier.save_pretrained(f"vit_classifier_{epoch + 1}")

        with torch.no_grad():
            correct = 0
            total = 0

            val_bar = tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{epochs} - Validation, Precision: -"
            )
            for data in val_bar:
                image_batches, labels = data

                output = vit_classifier(image_batches)

                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                val_bar.set_description(
                    f"Epoch {epoch + 1}/{epochs} - Validation, Precision: {correct / total:.4f}"
                )


def main():
    device = torch.device("cuda")

    image_size = 448
    vit_classifier, vit_processor = load_simple_classifier(image_size, device)
    vit_classifier.print_parameters()

    epochs = 50

    optimizer = vit_classifier.set_optimizer(torch.optim.Adam, lr=2e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader, val_loader = load_data(
        processor=vit_processor,
        image_size=image_size,
        max_image_stack_size=1,
        minibatch_size=128,
        data_split=0.8,
        device=device,
        seed=42
    )

    train(
        vit_classifier,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        epochs=epochs
    )


if __name__ == "__main__":
    main()
