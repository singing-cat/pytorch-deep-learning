"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils
import argparse

from torchvision import transforms

parser = argparse.ArgumentParser('Train and save an image classification model.')

parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hidden_units', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--train_dir', type=str, default="data/pizza_steak_sushi/train")
parser.add_argument('--test_dir', type=str, default="data/pizza_steak_sushi/test")


# # Setup hyperparameters
# NUM_EPOCHS = 5
# BATCH_SIZE = 32
# HIDDEN_UNITS = 10
# LEARNING_RATE = 0.001

# # Setup directories
# train_dir = "data/pizza_steak_sushi/train"
# test_dir = "data/pizza_steak_sushi/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

args = parser.parse_args()
train_dir, test_dir = args.train_dir, args.test_dir
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
NUM_EPOCHS = args.num_epochs

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_argparse_model.pth")
