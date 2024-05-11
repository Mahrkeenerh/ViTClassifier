import random

import torch
import transformers


def load_VED_vit(model_path, image_size, device):
    """Load the ViT model from a VED model."""

    ved_model = transformers.VisionEncoderDecoderModel.from_pretrained(
        model_path,
        device_map=device
    )

    # delete the language head
    del ved_model.decoder
    torch.cuda.empty_cache()

    vit_processor = transformers.ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    vit_processor.size = {"width": image_size, "height": image_size}

    vit_model = ved_model.encoder

    # Upscale ViT
    vit_model.embeddings.patch_embeddings.image_size = [image_size, image_size]

    # Upscale position embeddings
    old_position_embeddings = vit_model.embeddings.position_embeddings
    first_row = old_position_embeddings[:, 0, :].clone().unsqueeze(0)
    rest_upscaled = torch.nn.functional.interpolate(
        old_position_embeddings[:, 1:, :].clone().unsqueeze(0),
        size=((image_size // 16) ** 2, vit_model.embeddings.patch_embeddings.projection.out_channels),
        mode="nearest"
    )[0]

    new_position_embeddings = torch.nn.Parameter(
        torch.cat((first_row, rest_upscaled), dim=1),
        requires_grad=True
    ).to(device)

    vit_model.embeddings.position_embeddings = new_position_embeddings

    # Disable unused layer
    vit_model.pooler = None

    # Freeze the whole model
    # for param in vit_model.parameters():
    #     param.requires_grad = False

    # vit_model.eval()

    return vit_model, vit_processor


# Classifier taking 1 image as input
class SimpleClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)

        self.training = True

    def forward(self, x):
        def single_pass(x):
            x = self.fc(x)
            x = self.softmax(x)
            return x

        if self.training:
            # Take a random image from each batch
            i = random.randint(0, x.size(1) - 1)
            x = x[:, i, :]
            x = x.view(x.size(0), -1)
            x = single_pass(x)
        else:
            # Evaluate on all images one by one, then average the results
            out = torch.zeros(x.size(0), 17).to(x.device)
            for i in range(x.size(1)):
                out += single_pass(x[:, i, :].view(x.size(0), -1))

            x = out / x.size(1)

        return x

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class EmbedClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim * 2, output_dim)
        self.conv1 = torch.nn.Conv1d(196, 32, kernel_size=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(32, 1, kernel_size=1)
        self.relu2 = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # Take a random image from each batch
        i = random.randint(0, x.size(1) - 1)
        x = x[:, i, :]

        # Split output into cls and image embeddings
        cls = x[:, 0, :]
        img = x[:, 1:, :]
        img = img.view(img.size(0), 196, -1)
        img = self.conv1(img)
        img = self.relu1(img)
        img = self.conv2(img)
        img = self.relu2(img)
        img = img.view(img.size(0), -1)

        x = torch.cat((cls, img), dim=1)
        x = self.fc(x)
        x = self.softmax(x)

        return x


# Classifier taking 1 image as input with hidden layers
class SimpleDeepClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, **kwargs):
        super().__init__()
        self.fc_in = torch.nn.Linear(input_dim, hidden_layers[0])
        self.fc_hidden = torch.nn.ModuleList([
            torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1])
            for i in range(len(hidden_layers) - 1)
        ])
        self.fc_out = torch.nn.Linear(hidden_layers[-1], output_dim)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # Take a random image from each batch
        i = random.randint(0, x.size(1) - 1)
        x = x[:, i, :]
        x = x.mean(dim=1)
        x = self.fc_in(x)
        x = self.relu(x)
        for fc in self.fc_hidden:
            x = fc(x)
            x = self.relu(x)
        x = self.fc_out(x)
        x = self.softmax(x)
        return x


# Classifier taking multiple images as input
class MultiImageClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, image_stack_size, **kwargs):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim * image_stack_size, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


# Classifier taking multiple images as input with hidden layers
class MultiImageDeepClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, image_stack_size, hidden_layers, **kwargs):
        super().__init__()
        self.fc_in = torch.nn.Linear(input_dim * image_stack_size, hidden_layers[0])
        self.fc_hidden = torch.nn.ModuleList([
            torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1])
            for i in range(len(hidden_layers) - 1)
        ])
        self.fc_out = torch.nn.Linear(hidden_layers[-1], output_dim)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc_in(x)
        x = self.relu(x)
        for fc in self.fc_hidden:
            x = fc(x)
            x = self.relu(x)
        x = self.fc_out(x)
        x = self.softmax(x)
        return x


class ViTClassifier(torch.nn.Module):
    def __init__(self, vit, classifier):
        super().__init__()
        self.vit = vit
        self.classifier = classifier
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        if self.vit is not None:
            enc_outs = []
            # i = random.randint(0, x.size(0) - 1)
            # out = self.vit(pixel_values=x[i, :, :, :, :]).last_hidden_state[:, 0, :].unsqueeze(1)
            # x = out.unsqueeze(1)

            for image_batch in x:
                enc_out = self.vit(pixel_values=image_batch).last_hidden_state
                # Only take first token - [CLS]
                enc_outs.append(enc_out[:, 0, :].unsqueeze(1))

            x = torch.cat(enc_outs, dim=1)

        x = self.classifier(x)

        return x

    def train(self):
        if self.vit is not None:
            self.vit.train()
        self.classifier.train()

    def eval(self):
        if self.vit is not None:
            self.vit.eval()
        self.classifier.eval()

    def set_weights(self, weights):
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    def loss(self, image_batches=None, y=None, x=None):
        assert image_batches is not None or x is not None, "Either image_batches or x must be provided"
        assert image_batches is None or x is None, "Only one of image_batches or x must be provided"
        assert y is not None, "y must be provided"

        if image_batches is not None:
            x = self(image_batches)

        return self.loss_fn(x, y)

    def set_optimizer(self, optimizer, lr):
        # self.optimizer = optimizer(self.classifier.parameters(), lr=lr)
        self.optimizer = optimizer(self.parameters(), lr=lr)

        return self.optimizer

    def print_parameters(self):
        if self.vit is not None:
            vit_params = sum(p.numel() for p in self.vit.parameters())
            vit_trainable_params = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
            print(f"ViT trainable params: {vit_trainable_params:,} || all params: {vit_params:,} || trainable/all: {vit_trainable_params / vit_params * 100:.2f}%")

        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        classifier_trainable_params = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        print(f"Classifier trainable params: {classifier_trainable_params:,} || all params: {classifier_params:,} || trainable/all: {classifier_trainable_params / classifier_params * 100:.2f}%")

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)

    def load_pretrained(self, path):
        self.load_state_dict(torch.load(path))

    def load_pretrained_head(self, path):
        temp_vit = self.vit
        self.vit = None
        self.load_state_dict(torch.load(path))
        self.vit = temp_vit


class NoScheduler():
    def __init__(self, optimizer, **kwargs):
        self.optimizer = optimizer

    def get_last_lr(self):
        return [self.optimizer.param_groups[0]["lr"]]

    def step(self):
        pass
