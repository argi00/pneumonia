import torch
from pneumonia_cls.models import Orchestrator
from pneumonia_cls.dataset import MyDataModule
from pytorch_lightning.callbacks  import EarlyStopping,ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as L

if __name__ == "__main__":

    # Select dataset to use for experiments
    use_huggingface_data=True # set to False to use data @ "../data/PediatricChestX-rayPneumonia"

    # build datamodule
    datamodule = MyDataModule(batchsize=32,
                              resample_val=True,
                              resample_train=False,
                              use_pediatricdataset=not use_huggingface_data, 
                              pediatricdata_path="/teamspace/studios/this_studio/pneumonia-classifier/data/Pediatric Chest X-ray Pneumonia")

    orch = Orchestrator(pos_weight=torch.tensor([2.0]),
                        lr=1e-3)
    
    early_stop_callback = EarlyStopping(monitor="val_f1", 
                                        min_delta=1e-4, 
                                        patience=10, 
                                        verbose=False, 
                                        mode="max")
    
    wandb_logger = WandbLogger(project='pneumonia-cls', name="Using hugging to test", log_model=False)

    checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_f1",
    mode="max",
    dirpath="../models/ckpt",
    filename="sample-mnist-{epoch:02d}-{val_f1:.2f}",
)
    callbacks = [early_stop_callback,checkpoint_callback]

    trainer = L.Trainer(max_epochs=50, 
                        max_steps=-1,
                        accelerator='gpu',
                        num_sanity_val_steps=2, 
                        logger=wandb_logger,
                        precision='16-mixed',
                        deterministic=False,
                        reload_dataloaders_every_n_epochs=3,
                        callbacks = callbacks
                        )
    
    #trainer.fit(orch, datamodule)


    #trainer.test(orch,datamodule,ckpt_path="/teamspace/studios/this_studio/pneumonia-classifier/models/ckpt/sample-mnist-epoch=29-val_f1=0.99.ckpt")
    
    #Exportation du modèle vers ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    import torch.onnx
    onnx_file_path = "pneumonia.onnx"
    torch.onnx.export(orch,              # Le modèle PyTorch
                      dummy_input,        # Exemple de donnée (batch de taille 1, image 224x224)
                      onnx_file_path,     # Chemin où enregistrer le fichier ONNX
                      export_params=True, # Enregistrer les poids du modèle
                      opset_version=12,   # Version de l'API ONNX
                      do_constant_folding=True,  # Optimisation de constante
                      input_names=['input'],  # Noms des entrées
                      output_names=['output'],  # Noms des sorties
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})  # Pour des tailles de batch dynamiques