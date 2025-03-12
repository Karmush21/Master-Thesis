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

from pure_ViT import Pure_ViT 

from collections import defaultdict

from datetime import datetime




from tensorboardX import SummaryWriter

def extract_numeric_part(timepoint):
    # Assumes timepoint format is 'dt' followed by numbers
    return int(timepoint[2:])



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

    parser.add_argument('--network', type=str, default="ours", help="Decide on which netowrk to use") #TODO Should really be --model... 

    parser.add_argument('--volume_path', type=str, help="Path for inference on a specific volume. For now this will have to be an npz file") #TODO Remove the npz req later.

    parser.add_argument('--do_logging',action='store_true', help="Useful in interactive mode. Overwrites old test log" )
    parser.add_argument('--log_name', type=str, default="default_log", help="Name of the log file.")

    parser.add_argument('--do_all_networks', action="store_true", help = "Does all the networks used in comparisons")


    parser.add_argument('--volumes_to_save', type=str, default="", help="Comma-separated list of volume identifiers to save, in 'ptX_dtY' format.")


    args = parser.parse_args()

    return args



if __name__ == "__main__":
    print("\nSTARTING INFERENCE \n")

    args = parser_argument()

    #-----------------LOGGING-----------------
    #TODO This is kinda bad, but it works for now. 
    
    text_log_dir = "/proj/berzelius-2023-86/users/x_anmka/thesis/TRUNet/inference_logs/text_logs/"
    tensorboard_log_dir = "/proj/berzelius-2023-86/users/x_anmka/thesis/TRUNet/inference_logs/tensorboard_logs"
    run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
   
    if not args.do_logging:
        # logging
        #prev: args.save_path + "/log.txt" #Anmar: filemode="w" resets the test_log.txt after every run, sometimes you want this i guess and sometimes not.
        logging.basicConfig(filename="/proj/berzelius-2023-86/users/x_anmka/thesis/TRUNet/inference_logs/text_logs/test_log.txt", filemode="w", level=logging.INFO, 
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        #delete_folder_and_contents("/proj/berzelius-2023-86/users/x_anmka/thesis/TRUNet/inference_logs/tensorboard_logs/", "test_tb_log")
        #writer = SummaryWriter('/proj/berzelius-2023-86/users/x_anmka/thesis/TRUNet/inference_logs/tensorboard_logs/test_tb_log')
    
    else:
        logging.basicConfig(filename = os.path.join(text_log_dir, run_id + "_" + args.log_name + ".txt"), 
                            filemode="w", level=logging.INFO, 
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        writer = SummaryWriter(os.path.join(tensorboard_log_dir, run_id + "_" + args.log_name))
    
    logging.info("STARTING A NEW LOG")

    logging.info("\n--------------------------------------\n")

    #-----------------------------------------
    
    #e.g. args.model_folder = UNETR and args.epoch = 394 
    #path_to_model = os.path.join(args.root_model_folder,args.model_folder, f"best_metric_model_epoch_{args.epoch}.pth")
    #print(path_to_model)

    
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

    if args.volumes_to_save:
        dataset_test = []
        volume_list = args.volumes_to_save.split(',')
        for volume in volume_list:

            volume_path = os.path.join(args.dataset_path, f'ct_{volume}.npz')  
            data = np.load(volume_path)
            
            image, label = data[data.files[0]], data[data.files[1]] 
            sample = {'image': image, 'label': label, 'case_name': volume_path.split("/")[-1].split(".")[0]} #nii or npz depends on data.

            image_label_sample = {'image': sample['image'], 'label': sample['label']}
            
            transformed_sample = test_transforms(image_label_sample)
            
            sample['image'] = transformed_sample['image']
            sample['label'] = transformed_sample['label']
            
            dataset_test.append(sample)
        
    else:
        
        dataset_test = fetch_dataset(os.path.join(args.root_path, "combined_myo_segs_npz/test"),  #TODO Change this later
                            transform=test_transforms)

    
    # TODO no point having shuffle to True right?
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, 
                            pin_memory=True) #TODO removed worker_init_fn and the function. really needed here??
    
    print("Length: ", len(test_loader))
    #TODO Investigate old ours, is that decoder better or not?
    

    if args.do_all_networks:
        network_name_list = ["U-Net", "TRUNet", "TRUNet_no_vit", "UNETR", "ours", "ours_no_vit", "pure_vit"]
        model_folder_list = [
            "new_basic_unet", 
            "TRUNet",
            "new_TRUNet_no_vit",
            "UNETR",
            "ours_with_org_u_net",
            "ours_no_vit_without_augs_in_val",
            "pure_vit"
                       ]
        
        #network_name_list = ["pure_vit"]
        #model_folder_list = ["pure_vit"]
    else:
        network_name_list = [args.network]
        model_folder_list = [os.path.join(args.root_model_path, args.model_folder)]
    
    
    assert len(network_name_list) == len(model_folder_list), "The lengths of 'network_name_list' and 'model_folder_list' do not match."

    
    for network_name, model_folder in zip(network_name_list, model_folder_list):        
        
        full_model_folder_path = os.path.join(args.root_model_path, model_folder)
        pth_files = list_pth_files_in_folder(full_model_folder_path)
        model_file = pth_files[-1]  #[-1] is the last epoch saved. [0] is the first saved. 
        epoch = int(model_file.split("_")[-1].replace(".pth", ""))

        
        #TODO Maybe use this!
        #print(select_evenly_spaced_epochs(pth_files, 5))


        
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
            
            model = Pure_ViT(network_settings)

        #TODO Add functionality for the three other U-Nets. or some better fucking thing than this at least.
        elif network_name == "U-Net_variant":
            print("U-Net Variant")
            network_settings = proposed_network_configs(args.img_size, args.num_classes)
            
            #network_settings.encoder_channels = [32, 64, 128, 256, 512, 1024] # Max ours and Max U-Net
            network_settings.encoder_channels = [16, 32, 64, 128, 256] # Simple U-Net

            network_settings.decoder_channels = network_settings.encoder_channels[::-1]
            
            #For the Max U-Net
            network_settings.use_stride_conv_downsampling = False # Use max-pooling
            network_settings.activation = "relu" # Use ReLU activation function

            if network_settings.encoder_channels == [16, 32, 64, 128, 256] and network_settings.use_stride_conv_downsampling == False and network_settings.activation == "relu":
                tag = "simple_u_net"
            elif network_settings.use_stride_conv_downsampling == False and network_settings.activation == "relu":
                tag = "max_u_net"
            else:
                tag = "max_ours_no_vit"

            print(f"Tag is {tag}")
            model = proposed_network(network_settings, use_transformer=False) 
            
                
        else:
            raise ValueError(f"Unsupported network type: {args.network}. Please choose from 'TRUNet', 'U-Net', 'UNETR', 'ours', 'ours_no_vit', 'pure_vit'.")

        
        #print(f"Loading model for {network_name} at epoch {epoch}")
        logging.info(f"Loading model for {network_name} at epoch {epoch}")
        # Load model

        total_path = os.path.join(full_model_folder_path, model_file)
        
        load_model(model, total_path)
        
        #model.load_state_dict(torch.load(path_to_model))
    
      
        model.eval()
        model.to(device)
        with torch.no_grad():
            #dice_tmp_test = []
            #hausdorff_tmp_test = []
            cluster_list = []
            data = []
            segmentation_data = {}
            for epoch in range(1):
                for i_batch, sampled_batch in enumerate(tqdm(test_loader, desc="Running on test set")):
                    inputs, targets, name = sampled_batch['image'], sampled_batch['label'], sampled_batch["case_name"][0]                    
                    

                    inputs, targets = inputs.unsqueeze(1), targets.unsqueeze(1)
                    inputs, targets = inputs.to(device), targets.to(device)

                    inputs = inputs.float()
                    targets = targets.float()
                    
                    test_outputs = model(inputs)
                    
                    
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


                    segmentation_data[name] = {
                        'inputs': inputs,
                        'targets': targets,
                        'outputs': test_outputs
                    }

                    
                    #Check how many small clusters we have
                    pred = torch.argmax(test_outputs, dim=1)
                    pred = pred.squeeze(0)  # Remove the batch dimension
                    pred = pred.float()
                    pred = pred.permute(1, 2, 0) #TODO When i dont permute, the pred has same orientation and input and target. but when i do pred, i get z-dimension where i should get in itk-snap.

                    pred = pred.cpu().numpy()
                    
                    
                    
                    # Retain only largest component
                    l, nl = label_scipy(pred)
                    cluster_list.append(nl) #We wanna keep one cluster as that is the "right one"
                    sizes_l = [len(np.where(l==n+1)[0]) for n in range(nl)]
                    keeper = np.where(sizes_l==np.nanmax(sizes_l))[0][0]+1 #finds the largets component
                    pred = np.where(l==keeper,1,0)

                

            #print(cluster_list)
            #print(sum(cluster_list) / len(cluster_list))
            #quit()
            #print(data)

            # --------------------- TIME RESOLVED STATISTICS, AVERAGED OVER PATIENTS ---------------------------------------
            result = defaultdict(lambda: {'dice_scores': [], 'hausdorff_distances': []})
            # Populate the dictionary
            for name, dice_scores, hausdorff_distance in data:
                timepoint = name.split("_")[-1]
                result[timepoint]['dice_scores'].append(dice_scores)
                result[timepoint]['hausdorff_distances'].append(hausdorff_distance)
            
        
            result = dict(result)
            
            sorted_result = dict(sorted(result.items(), key=lambda x: extract_numeric_part(x[0])))


            models = []
            timepoints = []
            dice_scores_list = []

            # Iterate over timepoints and scores
            for timepoint, values in sorted_result.items():
                # Assuming there's only one score per timepoint per network
                dice_score_pd = values['dice_scores'][0]

                models.append(network_name)
                timepoints.append(timepoint)
                dice_scores_list.append(dice_score_pd)

        
            writer = SummaryWriter(os.path.join(tensorboard_log_dir, model_folder))
            #To get the time-resolved stats.
            for name, values in sorted_result.items():
                timepoint = name.split("_")[-1]
                
                writer.add_scalar('Dice_avg', fmean(values["dice_scores"]),  int(timepoint[2:]))
                writer.add_scalar('HD95_avg', fmean(values["hausdorff_distances"]),  int(timepoint[2:]))
                #print(f'Timepoint {timepoint}:  Average Dice: {fmean(values["dice_scores"])} | Average HD95: {fmean(values["hausdorff_distances"])}')
                logging.info(f'Timepoint {timepoint}:  Average Dice: {fmean(values["dice_scores"])} | Average HD95: {fmean(values["hausdorff_distances"])}')

            logging.info("\n")
            

            # --------------------- FINDING BEST WORST AND MEDIAN AND COMBINING SCORES ---------------------------------------
            # Extract Dice scores and Hausdorff distances
            dice_scores = np.array([t[1] for t in data])
            hausdorff_distances = np.array([t[2] for t in data])

            dice_val = fmean(dice_scores)
            haus_val = fmean(hausdorff_distances)

            logging.info(f"Mean Dice score on test set for {network_name}: {dice_val}")
            logging.info(f"Mean 95th Hausdorff distance on test set for {network_name}: {haus_val}")
            logging.info("\n")
            
            #print(f"Mean Dice score on test set: {dice_val}")
            #print(f"Mean 95th Hausdorff distance on test set: {haus_val}")

            
            # Normalize Hausdorff distances (range [0, 1])
            hausdorff_distances_normalized = (hausdorff_distances - hausdorff_distances.min()) / (hausdorff_distances.max() - hausdorff_distances.min())

            
            # Combine the metrics (simple average here, but you can use weighted sums if needed)
            combined_scores = (dice_scores + (1 - hausdorff_distances_normalized)) / 2

            # Create a list of tuples with combined scores
            combined_data = list(zip(data, combined_scores))


            # Sort by combined score
            sorted_combined_data = sorted(combined_data, key=lambda x: x[1], reverse=True)

        
            # Extract best, median, and worst segmentations, however not used right now.
            best_segmentation = sorted_combined_data[0]
            worst_segmentation = sorted_combined_data[-1]
            median_segmentation = sorted_combined_data[len(sorted_combined_data) // 2]

            
            # Order is best, worst, medain
            names = [(best_segmentation[0][0], "best"), (worst_segmentation[0][0], "worst"), (median_segmentation[0][0], "median")]
            logging.info(names)
            logging.info("DONE WITH ONE NETWORK NOW \n")

            
            #Continue if you don't want to save
            if not args.volumes_to_save:
                continue
    
            #this is shit i know, ct to pt to remove, but fuck it, this is what i was given. 
            names = args.volumes_to_save.split(',') 
            names = [f"ct_{volume_id}" for volume_id in names]

            
            #TODO Add args parser for saving volumes
            inference_folder_dir = args.save_path + network_name
            
            #I know, don't @ me.
            if not os.path.exists(inference_folder_dir):
                if network_name == "U-Net_variant":
                    inference_folder_dir = args.save_path + tag
                    os.makedirs(inference_folder_dir, exist_ok=True)
                else:
                    os.makedirs(inference_folder_dir, exist_ok=True)

            
            
            for name in tqdm(names, "Saving Volumes"):
                data = segmentation_data[name]
                inputs = data["inputs"]
                targets = data["targets"]
                pred = data["outputs"]

                replace_name = name.replace("ct_", "") # Remove ct part from name
                full_name = replace_name + "_myoseg" + ".nii.gz"
                full_name_path = os.path.join(args.info_path, full_name)

                nifti_image = nib.load(full_name_path)
                nifti_data = nifti_image.get_fdata()
                z_dim = nifti_data.shape[-1]



                # All have to be permuted in order for it to be right orientation in ITK-snap.
                # Resave in the original format
                inputs = torch.squeeze(inputs, dim=0)
                inputs = torch.squeeze(inputs, dim=0)
                inputs = inputs.float()
                inputs = inputs.permute(1, 2, 0)                

                targets = torch.squeeze(targets, dim=0)
                targets = torch.squeeze(targets, dim=0)
                targets = targets.float()
                targets = targets.permute(1, 2, 0)

                pred = torch.argmax(pred, dim=1)
                pred = pred.squeeze(0)  # Remove the batch dimension
                pred = pred.float()
                pred = pred.permute(1, 2, 0) #TODO When i dont permute, the pred has same orientation and input and target. but when i do pred, i get z-dimension where i should get in itk-snap.

                

                inputs = inputs.cpu().numpy()
                targets = targets.cpu().numpy()
                pred = pred.cpu().numpy()
                
                
                 
                # Retain only largest component
                l, nl = label_scipy(pred)
                sizes_l = [len(np.where(l==n+1)[0]) for n in range(nl)]
                keeper = np.where(sizes_l==np.nanmax(sizes_l))[0][0]+1 #finds the largets component
                pred = np.where(l==keeper,1,0)

                inputs = zoom(inputs, (512 / 224, 512 / 224 , z_dim / 224), order=3) 
                targets = zoom(targets, (512 / 224 , 512 / 224 , z_dim / 224), order=0) 
                    
                #TODO Order 3 seems to just get the edges? i.e. it doesn't fill anything?  
                pred = zoom(pred, (512 / 224, 512 / 224 , z_dim / 224), order=0) 
                
                
                
                inference_savepath_pred = os.path.join(inference_folder_dir, f"{network_name}_{name}_pred.nii.gz")
                inference_savepath_input = os.path.join(inference_folder_dir, f"{network_name}_{name}_input.nii.gz")
                inference_savepath_target = os.path.join(inference_folder_dir, f"{network_name}_{name}_target.nii.gz")
                
                save_as_nifti(pred, inference_savepath_pred, info_ = full_name_path, use_info = True)
                save_as_nifti(inputs, inference_savepath_input, info_ = full_name_path, use_info = True)
                save_as_nifti(targets, inference_savepath_target, info_ = full_name_path, use_info = True)

