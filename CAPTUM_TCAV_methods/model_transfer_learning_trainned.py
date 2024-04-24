import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_v2_s
from torchvision.models import EfficientNet_V2_S_Weights

# Initialize model
def get_model():
  weights = EfficientNet_V2_S_Weights.DEFAULT
  model = efficientnet_v2_s(weights=weights)
  model.eval()

  # Initialize inference transforms
  auto_transforms = weights.transforms()

  return model, auto_transforms
  
  
model, auto_transforms = get_model()
print('The inout of the model are transformed with: ')
print(auto_transforms)


batch_size=8


train_dataset_raw = torchvision.datasets.OxfordIIITPet(
        root="Data",target_types='category', transform=transforms.Compose([ transforms.Resize(236),transforms.CenterCrop(228),transforms.ToTensor()]), download=True
    )

train_loader_raw = DataLoader(
        dataset=train_dataset_raw,
        batch_size=batch_size,
        shuffle=True
    )

dataset_labels=train_dataset_raw.class_to_idx.keys()  
print('Categories of the dataset:\n',  ','.join(dataset_labels))



# Set the manual seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def make_trainable(module):
    """Unfreezes a given module.
    Args:
        module: The module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def _recursive_freeze(module):
    """Freezes the layers of a given module.
    Args:
        module: The module to freeze
        train_bn: If True, leave the BatchNorm layers in training mode
    """
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES)):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:
            # Make the BN layers trainable to keep mean and std 
            #_make_trainable(module)
            # Make the BN layers freezed
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
    else:
        for child in children:
            _recursive_freeze(module=child)


def freeze(module):
    children = list(module.children())

    for child in children:
        _recursive_freeze(module=child)
        
        
        

# Get the length of class_names
number_classes = len(dataset_labels)

model,_ = get_model()

model.eval()
freeze(model)
# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=number_classes, # same number of output units as our number of classes
                    bias=True))

make_trainable(model.classifier)




from torchmetrics.functional  import classification
import pytorch_lightning as pl

# Setting the seed
pl.seed_everything(42)

model_dict = {}
model_dict['efficient_net'] = model

def create_model(model_name):
     return model_dict[model_name]


class OxfordIIITPetModule(pl.LightningModule):
    def __init__(self, model_name,  optimizer_name, optimizer_hparams):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()


        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = create_model(model_name)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        self.learning_rate = self.hparams.optimizer_hparams["lr"]

        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "AdamW":
            #optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                   lr=self.learning_rate)
            
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                   lr=self.lr)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 =10)

        #return [optimizer], [scheduler]
        

        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 =10),
            "interval": "step",
            "frequency": 30
            
        }
    }  

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = classification.accuracy(preds,labels, task="multiclass",num_classes=number_classes)
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        

        return loss  

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs) 
        acc = classification.accuracy(preds, labels, task="multiclass", num_classes=number_classes)
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs) 
        acc = classification.accuracy(preds,labels, task="multiclass", num_classes=number_classes)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        
        
        
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print('device: ', device)


from torchinfo import summary

#load model from the path printed above after training 
model = OxfordIIITPetModule.load_from_checkpoint('/content/drive/My Drive/Colab Notebooks/Labs/Computer Vision/CAPTUM_methods/saved_models/efficient_net/model_checkpoint/epoch=3-step=920.ckpt')
#this one does not works without training or not?????
#model = OxfordIIITPetModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training
model.freeze()
model = model.to(device)
model.eval()


