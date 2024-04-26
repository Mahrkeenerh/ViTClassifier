import torch

import modeling
import training


def main():
    device = torch.device("cuda")

    # image_size = 448
    image_size = 224
    image_stack_size = 10
    epochs = 10

    preprocessed = True

    vit_model, vit_processor = modeling.load_VED_vit(
        model_path="/home/xbuban1/ved_model",
        image_size=image_size,
        device=torch.device("cpu")
    )

    # classifier = modeling.SimpleClassifier(
    #     input_dim=vit_model.embeddings.patch_embeddings.projection.out_channels,
    #     output_dim=17
    # ).to(device)
    classifier = modeling.SimpleDeepClassifier(
        input_dim=vit_model.embeddings.patch_embeddings.projection.out_channels,
        output_dim=17,
        hidden_layers=[512, 256]
    ).to(device)
    # classifier = modeling.MultiImageClassifier(
    #     input_dim=vit_model.embeddings.patch_embeddings.projection.out_channels,
    #     output_dim=17,
    #     image_count=image_stack_size
    # ).to(device)
    # classifier = modeling.MultiImageDeepClassifier(
    #     input_dim=vit_model.embeddings.patch_embeddings.projection.out_channels,
    #     output_dim=17,
    #     hidden_layers=[512, 256],
    #     image_count=image_stack_size
    # ).to(device)

    vit_classifier = modeling.ViTClassifier(
        vit=vit_model if not preprocessed else None,
        classifier=classifier
    )

    vit_classifier.print_parameters()

    optimizer = vit_classifier.set_optimizer(torch.optim.Adam, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader, val_loader = training.load_data(
        processor=vit_processor if not preprocessed else None,
        embeds_name=f"vit_cls_embeds_{image_size}.pt" if preprocessed else None,
        image_size=image_size,
        image_stack_size=image_stack_size,
        minibatch_size=128,
        data_split=0.8,
        device=device,
        seed=42
    )

    training.train(
        vit_classifier,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        epochs=epochs
    )


if __name__ == "__main__":
    main()
