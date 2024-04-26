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
    for param in vit_model.parameters():
        param.requires_grad = False

    vit_model.eval()

    return vit_model, vit_processor


# Classifier taking 1 image as input
class SimpleClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # Only take first image from each batch
        x = x[:, 0, :]
        x = x.mean(dim=1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


# Classifier taking 1 image as input with hidden layers
class SimpleDeepClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
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
        # Only take first image from each batch
        x = x[:, 0, :]
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
    def __init__(self, input_dim, output_dim, image_count):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim * image_count, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


# Classifier taking multiple images as input with hidden layers
class MultiImageDeepClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, image_count, hidden_layers):
        super().__init__()
        self.fc_in = torch.nn.Linear(input_dim * image_count, hidden_layers[0])
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
        if self.vit is None:
            enc_outs = x
        else:
            enc_outs = []
            for image_batch in x:
                enc_out = self.vit(pixel_values=image_batch).last_hidden_state
                # Only take first token - [CLS]
                enc_outs.append(enc_out[:, 0, :].unsqueeze(1))

            x = torch.cat(enc_outs, dim=1)

        x = self.classifier(x)

        return x

    def loss(self, x, y):
        x = self(x)
        return self.loss_fn(x, y)

    def set_optimizer(self, optimizer, lr):
        self.optimizer = optimizer(self.classifier.parameters(), lr=lr)

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
