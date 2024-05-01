import json
import os

import torch

import modeling
import training


def init_run(name):
    if os.path.exists("runs"):
        run_is = [int(run.split("_")[-1]) for run in os.listdir("runs")]

        if len(run_is) == 0:
            run_i = 0
        else:
            run_i = sorted(run_is)[-1] + 1
    else:
        run_i = 0

    run_name = f"runs/{name}_{run_i}"

    os.makedirs(run_name, exist_ok=True)
    print(f"Training run {run_i}")

    return run_name


def get_model(model_config, data_config, device):
    vit_device = device
    if data_config["preprocessed"]:
        vit_device = torch.device("cpu")

    vit_model, vit_processor = modeling.load_VED_vit(
        model_path="/home/xbuban1/ved_model",
        image_size=data_config["image_size"],
        device=vit_device
    )

    hidden_layers = model_config.get("hidden_layers", None)

    classifier = model_config["class_type"](
        input_dim=vit_model.embeddings.patch_embeddings.projection.out_channels,
        output_dim=17,
        hidden_layers=hidden_layers,
        image_stack_size=data_config["image_stack_size"]
    ).to(device)

    vit_classifier = modeling.ViTClassifier(
        vit=vit_model if not data_config["preprocessed"] else None,
        classifier=classifier
    )

    vit_processor = None if data_config["preprocessed"] else vit_processor

    return vit_classifier, vit_processor


def save_config(run_name, model_config, data_config, train_config):
    model_config = {
        "class_type": model_config["class_type"].__name__,
        "hidden_layers": model_config["hidden_layers"]
    }

    with open(f"{run_name}/config.json", "w") as f:
        json.dump({
            "model": model_config,
            "data": data_config,
            "train": train_config
        }, f, indent=4)


def main():
    device = torch.device("cuda")

    run_name = init_run("CLS_DeepClassifier")

    model_config = {
        "class_type": modeling.SimpleDeepClassifier,
        "hidden_layers": [512, 256],
        # "hidden_layers": []
    }

    data_config = {
        "dataset": "filtered",
        "imbalance_compensation": True,
        "preprocessed": True,
        "image_size": 224,
        "image_stack_size": 10,
        "minibatch_size": 128,
        "data_split": 0.8,
        # "seed": 42
    }

    train_config = {
        "epochs": 1000,
        "learning_rate": 1e-3
    }

    save_config(
        run_name,
        model_config,
        data_config,
        train_config
    )

    vit_classifier, vit_processor = get_model(model_config, data_config, device)
    vit_classifier.print_parameters()

    optimizer = vit_classifier.set_optimizer(torch.optim.Adam, lr=train_config["learning_rate"])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config["epochs"])
    scheduler = modeling.NoScheduler(optimizer)

    train_loader, val_loader, dataset = training.load_data(
        processor=vit_processor,
        embeds_name=f"vit_{data_config['dataset']}_cls_embeds_{data_config['image_size']}.pt" if data_config['preprocessed'] else None,
        data_config=data_config,
        device=device,
        seed=data_config.get("seed", None),
    )

    if data_config["imbalance_compensation"]:
        vit_classifier.set_weights(
            weights=dataset.class_weights
        )

    callbacks = [
        training.SaveCallback(
            run_name=run_name
        ),
        training.LogCallback(
            run_name=run_name
        )
    ]

    training.train(
        vit_classifier,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        dataset,
        epochs=train_config["epochs"],
        callbacks=callbacks
    )


if __name__ == "__main__":
    main()
