import pytorch_lightning as pl
from torch import nn
from torchvision import transforms, models


class ArtfifactDetectorSingle(pl.LightningModule):
    def __init__(self, hidden_size=64, learning_rate=2e-4, train_data_dir='./', val_data_dir='./', test_data_dir='./',
                 batch_size=32):

        super().__init__()

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.batch_size = batch_size

        self.train_set, self.val_set, self.test_set = None, None, None

        self.loss = nn.BCELoss()

        # We hardcode dataset specific stuff here.
        num_classes = 1
        channels, height, width = (3, 300, 300)
        self.train_transform = transforms.Compose([transforms.ToPILImage(),
                                                   transforms.Resize((width, height)),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor()])
        self.test_transform = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize((width, height)),
                                                  transforms.ToTensor()])

        # Build model
        resnet = models.resnet18()
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.05),
            nn.Linear(in_features=resnet.fc.in_features, out_features=num_classes)
        )

        self.base_model = resnet
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.sigm(self.base_model(x))
        return x
