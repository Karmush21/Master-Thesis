import argparse
import logging
import os
import torch
import numpy as np
import ml_collections
from tqdm import tqdm #For progress bar.

from trunet_main import fetch_dataset, proposed_network_configs, proposed_network
from augmentations import *
from torchvision import transforms
from torch.utils.data import DataLoader
from statistics import fmean
from functions import *
from monai.metrics import HausdorffDistanceMetric
from scipy.ndimage import zoom
from scipy.ndimage import label as label_scipy



from monai.networks.nets import UNet # Import Residual U-Net from Monai 
from monai.networks.nets import UNETR #Import UNETR from Monai
from monai.networks.nets import BasicUNet #The most basic unet

import torch.nn.functional as F #Softmax

from other_networks.TRUNet_network.ViT import VisionTransformer3d as TRUNet #TODO Chage testing name here later to TRUNET
from other_networks.TRUNet_network.trunet_main import TransUNet_configs as TRUNet_configs

from other_networks.TRUNet_network_no_vit.ViT import VisionTransformer3d as TRUNet_no_vit
from other_networks.TRUNet_network_no_vit.trunet_main import TransUNet_configs as TRUNet_no_vit_configs

from ViT_pure import ViT_pure 

from collections import defaultdict

from datetime import datetime

#Anova-things
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns


from tensorboardX import SummaryWriter

def extract_numeric_part(timepoint):
    # Assumes timepoint format is 'dt' followed by numbers
    return int(timepoint[2:])

#x = {'dt1': {'dice_scores': [0.9134020209312439, 0.8724507689476013], 'hausdorff_distances': [4.123105525970459, 4.123105525970459]}, 'dt2': {'dice_scores': [0.8885387182235718, 0.9137925505638123], 'hausdorff_distances': [5.196152210235596, 3.0]}, 'dt3': {'dice_scores': [0.9319407343864441, 0.9407413601875305], 'hausdorff_distances': [3.316624879837036, 2.0]}, 'dt4': {'dice_scores': [0.9156784415245056, 0.905627429485321], 'hausdorff_distances': [4.898979663848877, 4.242640495300293]}, 'dt5': {'dice_scores': [0.9039714336395264, 0.9145911931991577], 'hausdorff_distances': [8.602325439453125, 5.0]}, 'dt6': {'dice_scores': [0.9107851386070251, 0.9480836987495422], 'hausdorff_distances': [10.24695110321045, 2.0]}, 'dt7': {'dice_scores': [0.9494940042495728, 0.9315363168716431], 'hausdorff_distances': [3.7416574954986572, 3.0]}, 'dt8': {'dice_scores': [0.9511286616325378, 0.9517112374305725], 'hausdorff_distances': [3.605551242828369, 2.2360680103302]}, 'dt9': {'dice_scores': [0.9429810047149658, 0.9313744902610779], 'hausdorff_distances': [3.605551242828369, 7.141428470611572]}, 'dt10': {'dice_scores': [0.9228166937828064, 0.918399453163147], 'hausdorff_distances': [8.485280990600586, 7.8102498054504395]}, 'dt11': {'dice_scores': [0.9458327889442444, 0.9594014286994934], 'hausdorff_distances': [3.605551242828369, 2.2360680103302]}, 'dt12': {'dice_scores': [0.9150511622428894, 0.9243527054786682], 'hausdorff_distances': [7.141428470611572, 8.124038696289062]}, 'dt13': {'dice_scores': [0.9410247802734375, 0.9578423500061035], 'hausdorff_distances': [3.316624879837036, 2.0]}, 'dt14': {'dice_scores': [0.9422567486763, 0.9504834413528442], 'hausdorff_distances': [3.0, 2.0]}, 'dt15': {'dice_scores': [0.8938301801681519, 0.9180431962013245], 'hausdorff_distances': [7.0, 5.099019527435303]}, 'dt16': {'dice_scores': [0.9395511746406555, 0.8910173773765564], 'hausdorff_distances': [3.316624879837036, 5.0]}, 'dt17': {'dice_scores': [0.9175952672958374, 0.9459518790245056], 'hausdorff_distances': [6.324555397033691, 1.7320507764816284]}, 'dt18': {'dice_scores': [0.9430069327354431, 0.9356118440628052], 'hausdorff_distances': [3.1622776985168457, 2.2360680103302]}, 'dt19': {'dice_scores': [0.8984259366989136, 0.9003782272338867], 'hausdorff_distances': [6.4031243324279785, 3.605551242828369]}, 'dt20': {'dice_scores': [0.8827693462371826, 0.9207040071487427], 'hausdorff_distances': [5.385164737701416, 2.4494898319244385]}}


def parser_argument():

    parser = argparse.ArgumentParser(
        usage = "Testing parser",
        description="Testing description of parser"
    ) #Anmar: Creates an ArgumentParser object



    parser.add_argument('--num_classes', type=int,
                        default = 2, help='number of class') #Treat the problem as multiclass (for now)

    parser.add_argument('--root_path', type=str,
                        default='/proj/berzelius-2023-86/users/x_anmka/', 
                        help='path to test folder')

    parser.add_argument('--root_model_path', type=str,
                        default='/proj/berzelius-2023-86/users/x_anmka/thesis/TRUNet/models', 
                        help='path to root model folder')

    parser.add_argument('--model_folder', type=str,
                        default='deeper_network_ours_model/best_metric_model_epoch_1.pth', 
                        help='path to model')

    parser.add_argument('--save_path', type=str,
                        default='/proj/berzelius-2023-86/users/x_anmka/thesis/TRUNet/inference_test/', 
                        help='Inital path to save things')

    parser.add_argument('--info_path', type=str,
                        default='/proj/berzelius-2023-86/users/x_anmka/myo_280524_info', 
                        help='Path for information when saving as nifti')
    
    parser.add_argument('--dataset_path', type=str,
                        default='/proj/berzelius-2023-86/users/x_anmka/combined_myo_segs_npz/test', 
                        help='Path for information when saving as nifti')
    
    parser.add_argument('--epoch', type=str,
                        default='0', 
                        help='Which epoch for best model')


    parser.add_argument('--img_size', type=int,
                        default = 224, help='Isotropic input image size')

    parser.add_argument('--network', type=str, default="ours", help="Decide on which netowrk to use")

    parser.add_argument('--volume_path', type=str, help="Path for inference on a specific volume. For now this will have to be an npz file") #TODO Remove the npz req later.

    parser.add_argument('--do_logging',action='store_true', help="Useful in interactive mode. Overwrites old test log" )
    parser.add_argument('--log_name', type=str, default="default_log", help="Name of the log file.")

    parser.add_argument('--do_all_networks', action="store_true", help = "Does all the networks used in comparisons")


    parser.add_argument('--volumes_to_save', type=str, default="", help="Comma-separated list of volume identifiers to save, in 'ptX_dtY' format.")


    args = parser.parse_args()

    return args



if __name__ == "__main__":
    print("Creating box plots \n")

    args = parser_argument()

    #TODO, add random seed function later. #TODO Why?
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print(f"{torch.cuda.device_count()} GPUs available!")
            # Use all available GPUs
            #model = nn.DataParallel(model) #TODO let's not use this now
    
        device = torch.device("cuda") #cuda: gpu 0, cuda:1 gpu 1
    
    else:
        device = torch.device("cpu")
        print("CUDA not available")
    
    
    test_transforms = transforms.Compose(
    [RandomGenerator3d_zoom(output_size=(args.img_size, args.img_size, args.img_size), use_aug=False)])

    #dataset_test = fetch_dataset(os.path.join(args.root_path, "myo_280524_npz/test"),  #TODO Remove later
    #                            transform=test_transforms)


        
    dataset_test = fetch_dataset(os.path.join(args.root_path, "combined_myo_segs_npz/test"),  #TODO Change this later
                            transform=test_transforms)

    
    # TODO no point having shuffle to True right?
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, 
                            pin_memory=True) #TODO removed worker_init_fn and the function. really needed here??
    
    
    

    
    # network_name_list = ["U-Net", "TRUNet", "TRUNet_no_vit" , "UNETR", "ours", "ours_no_vit", "pure_vit"]
    # model_folder_list = [
    #     "new_basic_unet", 
    #         "TRUNet",
    #     "new_TRUNet_no_vit",
    #     "UNETR",
    #     "ours_with_org_u_net",
    #     "ours_no_vit_without_augs_in_val",
    #     "pure_vit"
    #                 ]

    #TODO, maybe investigate the three unet variants as well??
    network_name_list =  ["ours_no_vit", "TRUNet_no_vit"]
    model_folder_list =  ["ours_no_vit_without_augs_in_val", "new_TRUNet_no_vit"]

        
    
    
    assert len(network_name_list) == len(model_folder_list), "The lengths of 'network_name_list' and 'model_folder_list' do not match."

    results_box_plot = {
    network_name: {
        f'patient{j+1}': {'dice_scores': [None] * 20}  # Placeholder for 20 timepoints
        for j in range(len(test_loader) // 20)  # 20 timepoints per patient.
    }
    for network_name in network_name_list  # For each network in the list
    }

    

    for network_name, model_folder in zip(network_name_list, model_folder_list):        
        full_model_folder_path = os.path.join(args.root_model_path, model_folder)
        pth_files = list_pth_files_in_folder(full_model_folder_path)
        model_file = pth_files[-1]  #[-1] is the last epoch saved. [0] is the first saved. 
        epoch = int(model_file.split("_")[-1].replace(".pth", ""))
        
        if network_name == "TRUNet":
            network_settings = "TRUNet settings, havent changed anything, look in other_networks/TRUNet"
            print("Using TRUNet Network:")
            config_model_TRUNET = TRUNet_configs(args.img_size)
            model = TRUNet(config_model_TRUNET, img_size=args.img_size, zero_head=False,
                                vis=False)
            

        elif network_name == "TRUNet_no_vit":
            print("Using TRUNet Network without ViT:")
            network_settings = "TRUNet without vit settings, havent changed anything, look in other_networks/TRUNet_no_vit"
            config_model_TRUNET = TRUNet_no_vit_configs(args.img_size)
            model = TRUNet_no_vit(config_model_TRUNET, img_size=args.img_size, zero_head=False,
                                vis=False)

            
        #TODO Add your modifed network instead of this shit monai u-net
        elif network_name == "U-Net":
            print("Using Basic U-Net Network:")

            #MONAI UNET
            #network_settings = ml_collections.ConfigDict()
            #network_settings.in_channels = 1
            #network_settings.spatial_dims=3
            #network_settings.featuers = generate_encoder_channels(32, 5, monai_unet=True) #48,5 works but you need two fat gpu nodes. 
            #network_settings.norm = "instance"
            #network_settings.act = "relu"
            
        
            # model = BasicUNet(in_channels = network_settings.in_channels, 
            #                   spatial_dims = network_settings.spatial_dims, 
            #                   features = network_settings.featuers, 
            #                   norm = network_settings.norm, 
            #                   act = network_settings.act)   
            

            network_settings = proposed_network_configs(args.img_size, args.num_classes)
            network_settings.use_stride_conv_downsampling = False # Use max-pooling
            network_settings.activation = "relu" # Use ReLU activation function
            network_settings.encoder_channels = [32, 64, 128, 256, 512]

            model = proposed_network(network_settings, use_transformer=False)  



        elif network_name == "UNETR":
            print("Using UNETR Network (from Monai):") 
            
            network_settings = ml_collections.ConfigDict()
            network_settings.in_channels = 1
            network_settings.out_channels = 2
            network_settings.img_size = (args.img_size, args.img_size, args.img_size)
            network_settings.feature_size = 16 #TODO Change back later to 16
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
        elif network_name == "ours":
                print("Using Default (our) network")
                network_settings = proposed_network_configs(args.img_size, args.num_classes)
                network_settings.encoder_channels = [32, 64, 128, 256, 512]
                model = proposed_network(network_settings, use_transformer=True)

                network_settings.save_attention_maps = True
                network_settings.name = network_name

                

        
        elif network_name == "ours_no_vit":
            print("Removed Vit from our network, i.e. a modifed u-net")
            network_settings = proposed_network_configs(args.img_size, args.num_classes)
            network_settings.encoder_channels = [32, 64, 128, 256, 512]
            model = proposed_network(network_settings, use_transformer=False)

        elif network_name == "pure_vit":
            print("Pure ViT")
            network_settings = proposed_network_configs(args.img_size, args.num_classes)
            network_settings.hidden_size = 180 #Seems to the maxium right now
            network_settings.transformer_num_heads = 12
            network_settings.transformer_num_layers = 12
            network_settings.transformer_mlp_dim = network_settings.hidden_size * 4
            
            network_settings.save_attention_maps = True
            network_settings.name = network_name
            
            model = ViT_pure(network_settings)

        #TODO Add functionality for the three other U-Nets
        elif network_name == "other_U-Net":
            print("One of the three other u-nets")    
            
                
        else:
            raise ValueError(f"Unsupported network type: {args.network}. Please choose from 'TRUNet', 'U-Net', 'UNETR', 'ours', 'ours_no_vit', 'pure_vit'.")

        
        #print(f"Loading model for {network_name} at epoch {epoch}")
        logging.info(f"Loading model for {network_name} at epoch {epoch}")
        # Load model

        total_path = os.path.join(full_model_folder_path, model_file)
        
        load_model(model, total_path)
        
        #model.load_state_dict(torch.load(path_to_model))
    
        patient_id = 1 #reset per model
        model.eval()
        model.to(device)
        with torch.no_grad():
            #dice_tmp_test = []
            #hausdorff_tmp_test = []
            data = []
            for epoch in range(1): #why????
                for i_batch, sampled_batch in enumerate(tqdm(test_loader, desc="Running on test set")):
                    
                    # if (i_batch != 0 and i_batch % 20 == 0):
                    #     result = defaultdict(lambda: {'dice_scores': [], 'hausdorff_distances': []})
                    #     # Populate the dictionary
                    #     for name, dice_scores, hausdorff_distance in data:
                    #         timepoint = name.split("_")[-1]
                    #         result[timepoint]['dice_scores'].append(dice_scores)
                    #         result[timepoint]['hausdorff_distances'].append(hausdorff_distance)
                        
                    
                    #     result = dict(result)
                        
                    #     sorted_result = dict(sorted(result.items(), key=lambda x: extract_numeric_part(x[0]))) 

                    #     for (index, values) in enumerate(sorted_result.items()):
                    #         results_box_plot[network_name]["dice_scores"][patient_id][index] = values[1]["dice_scores"][0] #this is really bad i know. but 1 is dice score, 2 is HD95

                    #     print(results_box_plot)

                    #     print("\n")

                    #     patient_id += 1
                    #     data = []
                        
                    
                    
                    inputs, targets, name = sampled_batch['image'], sampled_batch['label'], sampled_batch["case_name"][0]                    
                    
            
                    inputs, targets = inputs.unsqueeze(1), targets.unsqueeze(1)
                    inputs, targets = inputs.to(device), targets.to(device)

                    inputs = inputs.float()
                    targets = targets.float()
                    
                    test_outputs = model(inputs)
                    
                    #TODO Add the U-Net here later as well
                    if network_name not in ["ours", "ours_no_vit", "U-Net", "pure_vit"]:
                        test_outputs = F.softmax(test_outputs, dim=1)

                    
                    # --------------------- CALCULATING THINGS ---------------------------------------
                    
                    targets_one_hot = one_hot_encoder(targets, args.num_classes)

                    dice_lee = dice_score(test_outputs, targets_one_hot)

                    dice_lee = dice_lee.item()
                    
                    
                    
                    #print("Dice Score:", dice_lee) #TODO Remove later
                    

                    #dice_tmp_test.append(dice_lee)

                    # Argmax the prediction
                    test_outputs_haus = torch.argmax(test_outputs, dim=1, keepdim=True) 

                    # One hot encode the prediction
                    test_outputs_haus = one_hot_encoder(test_outputs_haus, args.num_classes)
                    

                    hausdorff_metric = HausdorffDistanceMetric(include_background=False, distance_metric="euclidean", percentile=95)

                    # Calculate Hausdorff distance
                    hausdorff_distance = hausdorff_metric(y_pred = test_outputs_haus, y = targets_one_hot)

                    hausdorff_distance = hausdorff_distance.item()

                    #hausdorff_tmp_test.append(hausdorff_distance.item())

                    
                    data.append((name, dice_lee, hausdorff_distance))
                    if len(data) % 20 == 0:
                        result = defaultdict(lambda: {'dice_scores': [], 'hausdorff_distances': []})
                        # Populate the dictionary
                        for name, dice_scores, hausdorff_distance in data:
                            timepoint = name.split("_")[-1]
                            result[timepoint]['dice_scores'].append(dice_scores)
                            result[timepoint]['hausdorff_distances'].append(hausdorff_distance)
                        
                    
                        result = dict(result)
                        
                        sorted_result = dict(sorted(result.items(), key=lambda x: extract_numeric_part(x[0]))) 

                        
                        #print(sorted_result)
                        
                        #this is really bad i know. But code came from the averaging of timepoints which worked good there. 
                        for (index, values) in enumerate(sorted_result.items()): #values is a tuple, values[1] is where the values are.
                            #results_box_plot[network_name][f'patient{patient_id}']["dice_scores"][index] = values[1]["dice_scores"][0] #this is really bad i know. but 1 is dice score, 2 is HD95
                            results_box_plot[network_name][f'patient{patient_id}']["dice_scores"][index] = values[1]["hausdorff_distances"][0] #0 because only 1 value in the list. 
                        
                        
                        #print(results_box_plot)
                        #quit()
                        #print("\n")


                        patient_id += 1
                        data = []
                    

        #print("DONE WITH ONE MODEL")
    

    # Define your colors
    network_colors = {
    "Pure ViT": "#ffba21",
    "UNETR": "#e22020",
    "Ours": "#d88de4",
    "Ours no ViT": "#9334e6",
    "TRUNet": "#1715b3",
    "TRUNet no ViT": "#1bbee8",
    "U-Net": "#08e07b"
    }


    model_names = []
    timepoints = []
    dice_scores = []
    patients = []

    # Process the data for each model
    for model_name, patients_data in results_box_plot.items():
        for patient, metrics in patients_data.items():
            scores = metrics['dice_scores']
            timepoints.extend(range(1, len(scores) + 1))
            dice_scores.extend(scores)
            patients.extend([patient] * len(scores))
            model_names.extend([model_name] * len(scores))

    # Create a DataFrame
    df = pd.DataFrame({
        'Model': model_names,
        'Timepoint': timepoints,
        'Dice Score': dice_scores,
        'Patient': patients
    })

    # Map the network names in the DataFrame to match your color dictionary
    df['Model'] = df['Model'].replace({
        "pure_vit": "Pure ViT",
        "UNETR": "UNETR",
        "ours": "Ours",
        "ours_no_vit": "Ours no ViT",
        "TRUNet": "TRUNet",
        "TRUNet_no_vit": "TRUNet no ViT",
        "U-Net": "U-Net"
    })


    # Plotting
    fig, ax1 = plt.subplots(figsize=(20, 12))
    #plt.gca().set_facecolor('#2e2e2e')  # Slightly lighter gray for figure background

    boxplot = sns.boxplot(x='Timepoint', y='Dice Score', hue='Model', data=df, palette=network_colors, ax=ax1,
                medianprops={'label': '_median_', 'linewidth': 3})
    
    #sns.scatterplot(data=df[df['Network']=='modelx'], x='Timepoint', y='Dice Score', hue='Patient') #lees comment regarding hue
    
    median_colors = ['#FFFFFF']
    median_lines = [line for line in ax1.get_lines() if line.get_label() == '_median_']
    for i, line in enumerate(median_lines):
        line.set_color(median_colors[i % len(median_colors)])
    
    _fontsize = 14
    plt.title('Boxplot of Dice Scores Across Timepoints.', fontsize=_fontsize, fontweight='bold')
    plt.xlabel('Timepoint', fontsize=_fontsize, fontweight='bold')
    plt.ylabel('Dice Score', fontsize=_fontsize, fontweight='bold')
    plt.legend(title='Model')


    plt.savefig('boxplots/hybrid_models_no_vit.png', format='png')
    plt.savefig('boxplots/hybrid_models_no_vit.svg', format='svg')









#TESTING COLORS

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np

# # Sample DataFrame
# np.random.seed(0)
# df = pd.DataFrame({
#     'Model': np.random.choice(['Model1', 'Model2'], size=100),
#     'Timepoint': np.random.choice([1, 2, 3, 4], size=100),
#     'Dice Score': np.random.rand(100),
#     'Patient': np.random.choice([f'Patient{i}' for i in range(1, 8)], size=100)
# })

# # Define colors
# network_colors = {
#     "Model1": "#1715b3",  
#     "Model2": "#08e07b",  
# }

# # Plotting
# fig, ax1 = plt.subplots(figsize=(20, 12))
# boxplot = sns.boxplot(x='Timepoint', y='Dice Score', hue='Model', data=df, palette=network_colors, ax=ax1,
#             medianprops={'label': '_median_', 'linewidth': 3},)

# median_colors = ['#FFFFFF']
# median_lines = [line for line in ax1.get_lines() if line.get_label() == '_median_']
# for i, line in enumerate(median_lines):
#     line.set_color(median_colors[i % len(median_colors)])

# plt.title('Boxplot of Dice Scores Across Timepoints.')
# plt.xlabel('Timepoint')
# plt.ylabel('Dice Score')
# plt.legend(title='Model')

# # Save the plots
# plt.savefig('boxplots/testing.png', format='png')
# plt.savefig('boxplots/testing.svg', format='svg')

# # Show plot (optional)
# plt.show()
# #     # Define your colors
# #     network_colors = {
# #     "Pure ViT": "#ffba21",
# #     "UNETR": "#e22020",
# #     "Ours": "#d88de4",
# #     "Ours no ViT": "#9334e6",
# #     "TRUNet": "#1715b3",
# #     "TRUNet no ViT": "#1bbee8",
# #     "U-Net": "#08e07b"
# #     }