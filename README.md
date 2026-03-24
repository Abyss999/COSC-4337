### Authenticate Kaggle Account

1.  Go to your Kaggle account settings (kaggle.com/your_username/account).
2.  Under the 'API' section, click 'Create New API Token'. This will download a `kaggle.json` file.
3.  Upload this `kaggle.json` file to your Colab environment. You can do this by clicking the 'Files' icon on the left sidebar (folder icon), then 'Upload to session storage' icon, and selecting your `kaggle.json` file.


### Mode2 Backup_1 Using
1. When using it on streamlit, the ImageNet CNN must be like
   ''' model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),

    nn.Linear(256, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.2),

    nn.Linear(64, 3)
   )'''
