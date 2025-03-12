import argparse
import ml_collections
import os
from glob import glob
import torch
from torch.optim import lr_scheduler
import numpy as np
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from ViT import VisionTransformer3d as proposed_network
from pure_ViT import Pure_ViT 
from datetime import datetime
from trunet_train import trainer
from torchvision import transforms
from augmentations import *
from torch.utils.data import Dataset
from functions import *



# NETWORKS

from monai.networks.nets import UNet # Import Residual U-Net from Monai 
from monai.networks.nets import UNETR #Import UNETR from Monai
from monai.networks.nets import BasicUNet #The most basic unet

from other_networks.TRUNet_network.ViT import VisionTransformer3d as TRUNet #TODO Chage testing name here later to TRUNET
from other_networks.TRUNet_network.trunet_main import TransUNet_configs as TRUNet_configs

from other_networks.TRUNet_network_no_vit.ViT import VisionTransformer3d as TRUNet_no_vit
from other_networks.TRUNet_network_no_vit.trunet_main import TransUNet_configs as TRUNet_no_vit_configs




def parser_argument():
    parser = argparse.ArgumentParser(
        usage = "Testing parser",
        description="Testing description of parser"
    ) #Anmar: Creates an ArgumentParser object



    parser.add_argument('--num_classes', type=int,
                        default = 2, help='number of classes') #Treat the problem as multiclass (for now)
    parser.add_argument('--batch_size', type=int,
                        default = 2, help='training batch size')
    parser.add_argument('--root_path', type=str,
                        default='/proj/berzelius-2023-86/users/x_anmka/', 
                        help='path with train and val directories')

    parser.add_argument('--save_path', type=str,
                        default='/proj/berzelius-2023-86/users/x_anmka/thesis/TRUNet/', 
                        help='Inital path to save things')
    parser.add_argument('--info_path', type=str,
                        default='/proj/berzelius-2023-86/users/x_anmka/myo_280524_info', 
                        help='Path for information when saving as nifti')

    parser.add_argument('--max_epochs', type=int,
                        default = 500, help='number of epochs')

    parser.add_argument('--checkpoint', type=str,
                        default = None, help='path to partially trained model (i.e. the pth file)')

    parser.add_argument('--seed', type=int,
                        default = None, help='Seed that we will use')

    parser.add_argument('--base_lr', type=int,
                        default = 0.010, help='Learning rate') 
    parser.add_argument('--img_size', type=int,
                        default = 224, help='Isotropic input image size')


    parser.add_argument('--log_name', type=str, default="", help="if deafault, overwrites test.log.")
    parser.add_argument('--network', type=str, default="ours", help="Decide on which netowrk to use")



    args_ = parser.parse_args()
    return args_



#Anmar: This is the main config setting we will use for training. 
def proposed_network_configs(img_size, num_classes):
    #Anmar: Initializes an empty dictionary-like object. 
    configs_trunet = ml_collections.ConfigDict()
    
    #TODO Change the config name from trunet to something else

    # The maximum size of heads and layers 

    configs_trunet.hidden_size = 768 #This is the embbedings, i.e. how each "patch" is defined. 
    
    configs_trunet.transformer_mlp_dim = configs_trunet.hidden_size * 4 # Expansion factor of 4 has found to work emperically
    configs_trunet.transformer_num_heads = 12
    configs_trunet.transformer_num_layers = 12 
    configs_trunet.transformer_attention_dropout_rate = 0.0 #Dropout used for the self-attention
    configs_trunet.transformer_dropout_rate = 0.0 # Dropout used in the MLP 

    
    configs_trunet.in_channels = 1 #Change to whatever your data has
    
     
    configs_trunet.encoder_channels = [32, 64, 128, 256, 512]
    #configs_trunet.encoder_channels = [16, 32, 64, 128, 256]
    #configs_trunet.encoder_channels = [32, 64, 128, 256, 512, 1024]
    
    configs_trunet.decoder_channels = configs_trunet.encoder_channels[::-1]
    configs_trunet.n_classes = num_classes 
    
    configs_trunet.n_patches = 14 #TODO We can always make this 14 for now since we want to have to same latent space. but yeah fix later


    #TODO Residual network dosen't seem to work. No idea why to be honest. Dice just stuck at zero.
    configs_trunet.use_residual = False
    configs_trunet.use_stride_conv_downsampling = True #If it's false then we do simple max-pooling

    configs_trunet.activation = "leaky_relu"
    configs_trunet.normalization = "instance"


    return configs_trunet


#Anmar: This is our custom dataset? WHy doesn't it inhert from Dataset pytorch class? I gues it's fine as long as we have a len and getitem. 
#I've added it in for now, maybe no diff at all. but it seems to be the convetion at least
class fetch_dataset(Dataset):
    def __init__(self, base_dir, transform): #TODO Move this to parser later.
        self.transform = transform

        #self.data_dir = base_dir  
        sample_list = sorted(glob(os.path.join(base_dir, '*.npz')))

        self.class_frequency = {'background': 0, 'foreground': 0}
        
        self.sample_list = sample_list
        
        #self.calculate_class_frequency()

    # TODO Fix this function
    def calculate_class_frequency(self):
        print("Calculating weights...")
        for data_path in self.sample_list:
            data = np.load(data_path)
            labels =  data['arr_1']
            
            # Zero-pad only the last dimension of the volume to the target shape
            target_shape = (300, 300, 100)
            padded_labels = np.zeros(target_shape, dtype=np.int32)
            padded_labels[..., :labels.shape[-1]] = labels
            
            unique_labels, counts = np.unique(padded_labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                if label == 0:  # Assuming background class is labeled as 0
                    self.class_frequency['background'] += count
                else:
                    self.class_frequency['foreground'] += count
                
        print("Done!")
        return self.class_frequency
    
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        #print("Call getitem")
        data_path = self.sample_list[idx]
        data = np.load(data_path)

        

        image, label = data[data.files[0]], data[data.files[1]] 
        sample = {'image': image, 'label': label, 'case_name': self.sample_list[idx].split('/')[-1].split('.npz')[0]} #nii or npz depends on data.

        
        #TODO Add normalization as parser argument.
        # Min-Max normalization
        #min_val = np.min(image)
        #max_val = np.max(image)
        #image = (image - min_val) / (max_val - min_val)

        # Standardization (Z-score normalization)
        #mean = np.mean(image)
        #std = np.std(image)
        #image = (image - mean) / std
                
        #print("Distriubtion: ", calculate_class_distribution(label))
        
        #TODO Is zero-padding better or worse than using zoom (i.e. interpolation)??
        #image = pad_data(image, 512, 512, self.target_slices)
        #label = pad_data(label, 512, 512, self.target_slices)
        
        #sample = {'image': image, 'label': label, 'case_name': self.sample_list[idx].split('/')[-1].split('.npz')[0]} #nii or npz depends on data.
        if self.transform: #Anmar: Transform of course changes pixel values. 
            #We do this to keep the case_name. prob a better way to do it.
            image_label_sample = {'image': sample['image'], 'label': sample['label']}
            transformed_sample = self.transform(image_label_sample)
            sample['image'] = transformed_sample['image']
            sample['label'] = transformed_sample['label']
            
        
        #print(sample["image"].shape, sample["label"].shape)
        #sample['case_name':self.sample_list[idx].split('/')[-1].split('.npz')[0]]
        #Anmar: image and label in the sample dict are numpy:s. In other words, they are NOT torch tensors yet. Or MAYBE IF !SELF_TRANSFORM?
        #print(sample["image"].shape, sample["label"].shape)


        return sample


if __name__ == "__main__":

    args_ = parser_argument()

    #Anmar: Trainer takes args, and args sets its values using args_ from the parser above. I mean sure...
    args = ml_collections.ConfigDict()
    args.max_epochs = args_.max_epochs
    args.start_epoch = 1 # Value is changed when doing checkpoints. 
    args.save_path = args_.save_path
    args.root_path = args_.root_path
    args.info_path = args_.info_path
    args.num_classes = args_.num_classes
    args.batch_size = args_.batch_size
    args.seed = args_.seed
    args.base_lr = args_.base_lr
    args.checkpoint = args_.checkpoint
    args.img_size = args_.img_size
    args.log_name = args_.log_name
    args.network = args_.network
    args.learning_rate_scheduler = "Poly" #TODO Prob won't change this anyways (right)?

    

    
    #AUGMENTATIONS
    train_transforms = transforms.Compose(
        [RandomGenerator3d_zoom(output_size=(args.img_size,args.img_size,args.img_size), use_aug=True)])
    val_transforms = transforms.Compose(
    [RandomGenerator3d_zoom(output_size=(args.img_size, args.img_size,args.img_size), use_aug=False)])
    

    if args.seed is not None:
        set_seed(args.seed)
    

    # TODO Add boolen to remove the vit part from lee's paper
    if args.network == "TRUNet":
        network_settings = "TRUNet settings, havent changed anything, look in other_networks/TRUNet"
        print("Using TRUNet Network:")
        config_model_TRUNET = TRUNet_configs(args.img_size)
        model = TRUNet(config_model_TRUNET, img_size=args.img_size, zero_head=False,
                            vis=False)

    elif args.network == "TRUNet_no_vit":
        print("Using TRUNet Network without ViT:")
        network_settings = "TRUNet without vit settings, havent changed anything, look in other_networks/TRUNet_no_vit"
        config_model_TRUNET = TRUNet_no_vit_configs(args.img_size)
        model = TRUNet_no_vit(config_model_TRUNET, img_size=args.img_size, zero_head=False,
                            vis=False)
        

    elif args.network == "U-Net":
        print("Using Basic U-Net Network (from Monai):")

        # network_settings = ml_collections.ConfigDict()
        # network_settings.in_channels = 1
        # network_settings.spatial_dims=3
        # network_settings.featuers = generate_encoder_channels(32, 5, monai_unet=True) #48,5 works but you need two fat gpu nodes. 
        # network_settings.norm = "instance"
        # network_settings.act = "relu"
        
       
        # model = UNet(spatial_dims=3,
        #     in_channels=1,
        #     out_channels=2,
        #     channels=(16, 32, 64, 128, 256),
        #     strides=(2, 2, 2, 2),
        #     num_res_units=2,)

        network_settings = proposed_network_configs(args.img_size, args.num_classes)
        network_settings.use_stride_conv_downsampling = False # Use max-pooling
        network_settings.activation = "relu" # Use ReLU activation function

        model = proposed_network(network_settings, use_transformer=False)


    #TODO Add the original settings that people trained on
    elif args.network == "UNETR":
        print("Using UNETR Network (from Monai):") 
        
        network_settings = ml_collections.ConfigDict()
        network_settings.in_channels = 1
        network_settings.out_channels = 2
        network_settings.img_size = (args.img_size, args.img_size, args.img_size)
        network_settings.feature_size = 16 #TODO, change to 8 perhaps?
        network_settings.hidden_size = 768
        network_settings.mlp_dim = 3072
        network_settings.num_heads = 12
        network_settings.proj_type = "perceptron"
        network_settings.norm_name = "instance"
        network_settings.res_block = True
        network_settings.dropout_rate = 0.0
       
        model = UNETR(
        in_channels = network_settings.in_channels, 
        out_channels = network_settings.out_channels,  
        img_size = network_settings.img_size,
        feature_size = network_settings.feature_size, 
        hidden_size = network_settings.hidden_size, 
        mlp_dim = network_settings.mlp_dim,
        num_heads = network_settings.num_heads, 
        proj_type = network_settings.proj_type,
        norm_name = network_settings.norm_name, 
        res_block = network_settings.res_block, 
        dropout_rate = network_settings.dropout_rate 
        )


    # TODO Find a more suitable name
    elif args.network == "ours":
            print("Using Default (our) network")
            network_settings = proposed_network_configs(args.img_size, args.num_classes)
            network_settings.network_name = args.network
            
            model = proposed_network(network_settings, use_transformer=True)

        
    elif args.network == "ours_no_vit":
        print("Removed Vit from our network, i.e. a modifed u-net")
        network_settings = proposed_network_configs(args.img_size, args.num_classes)
        model = proposed_network(network_settings, use_transformer=False)

    elif args.network == "pure_vit":
        print("Pure ViT")
        network_settings = proposed_network_configs(args.img_size, args.num_classes)
        network_settings.hidden_size = 180 #Seems to the maxium right now
        network_settings.transformer_num_heads = 12
        network_settings.transformer_num_layers = 12
        network_settings.transformer_mlp_dim = network_settings.hidden_size * 4
        network_settings.network_name = args.network
        

        #print(network_settings)
        
        model = Pure_ViT(network_settings)    
        
            
    else:
        raise ValueError(f"Unsupported network type: {args.network}. Please choose from 'TRUNet', 'TRUNet_no_vit', 'U-Net', 'UNETR', 'ours', 'ours_no_vit'.")

    #OPTIMIZERS
    optimizer = torch.optim.Adam(model.parameters(), args.base_lr)


    # LEARNING RATE SCHEDULERS
    #scheduler =torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9) #Higer gamma is slower decay
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.max_epochs, power=0.9)

    # TODO For testing random seed and checkpoint. Remove later
    #weights = list(model.parameters())
    #first_weight = weights[0].flatten()[0].item()
    #print("First weight in the first tensor BEFORE loading:", first_weight)


    if args.checkpoint is None:
        print("Using No checkpoint")
    
    #TODO Fix so that everything basically just resumes: Textlogs, tensorboard, saving models, 
    else:
        print('loading checkpoint ', args.checkpoint.split("/")[-2] + "/" + args.checkpoint.split("/")[-1])
        #model.load_state_dict(torch.load(args.checkpoint))
        
        args.start_epoch = load_checkpoint(model, optimizer, scheduler, args.checkpoint)
        
        
        print("\nLOAD CHECKPOINT TEST")
        

    
        
    #weights = list(model.parameters())
    #first_weight = weights[0].flatten()[0].item()
    #print("First weight in the first tensor AFTER loading:", first_weight)
    

    #DATASETS
    train_dataset = fetch_dataset(os.path.join(args.root_path, "combined_myo_segs_npz/train"),  #myo_280524_npz/train
                                        transform=train_transforms)
    val_dataset = fetch_dataset(os.path.join(args.root_path, "combined_myo_segs_npz/validation"), #myo_280524_npz/validation 
                                    transform=train_transforms)
    

    #Adding weights
    #TODO Short fix soultion, will this solution work with multiple gpus?
    #TODO DO I want to try this again and see if I get better results?
    #device = torch.device('cuda:0')
    #class_train_freq = train_dataset.calculate_class_frequency()
    #print(class_train_freq)
    #positive_weight = torch.tensor([class_train_freq['background'] / class_train_freq['foreground']], dtype=torch.float,  device=device)
    #loss = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    #loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(100.))
    #loss = nn.BCEWithLogitsLoss()

    config = {'ds_val': val_dataset,
              'ds_train': train_dataset,
              'loss_function': DiceCELoss(include_background=True, to_onehot_y=False, sigmoid=False, softmax=False),
              #'loss_function': DiceLoss(include_background=True, to_onehot_y=False, sigmoid=False, softmax=False),
              'optimizer': optimizer,
              'scheduler': scheduler,
              'save_interval': 50}
    

    

    #Anmar: args and config is only used in trunet_train.py afterwards, doesn't get sent anywehere after that. 
    print(trainer(args, config, network_settings, model, args.save_path))
