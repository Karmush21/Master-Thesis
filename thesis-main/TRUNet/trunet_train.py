import logging
import os
import shutil
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from statistics import fmean
from datetime import datetime
from functions import *
import torch.nn.functional as F
from tqdm import tqdm #For progress bar.



#TODO Move these later to main file.
text_log_dir = "/proj/berzelius-2023-86/users/x_anmka/thesis/TRUNet/text_logs/"
tensorboard_log_dir = "/proj/berzelius-2023-86/users/x_anmka/thesis/TRUNet/tensorboard_logs/"



#TODO Double check (again) if this is correct
def dice_score(pred, gt):  # data in shape [batch, classes, h, w, d], where gt is one-hot encoded
    dice = []

    # Loop over each batch_item
    for batchloop in range(gt.shape[0]):
        dice_tmp = []
        # Assuming classes include background as roi 0, single class without background should be handled
        # Loops over each channel for a particular volume in the batch
        for roi in range(gt.shape[1]):
            # Skip the background if there are multiple classes
            if gt.shape[1] > 1 and roi == 0:
                continue
            
            # Selects the correct volumes in the correct batch to do the prediction. 
            pred_tmp = pred[batchloop, roi]
            gt_tmp = gt[batchloop, roi]

            # a checks where the two overlap. b,c just adds the total set. 
            a = torch.sum(pred_tmp[gt_tmp == 1])
            b = torch.sum(pred_tmp)
            c = torch.sum(gt_tmp)
            
            #Deal with division by zero
            if b + c == 0:
                metric = 0.0
            else:
                metric = a * 2.0 / (b + c)
            # Append for the current class and then continue with a new class.
            dice_tmp.append(metric)
        
        if dice_tmp:
            dice.append(torch.mean(torch.tensor(dice_tmp, device=pred.device)))
        else:
            dice.append(torch.tensor(0.0, device=pred.device))
    return torch.mean(torch.tensor(dice, device=pred.device))



def to_one_arr_encoding(input_tensor):  # input shape: [batch, channels, h, w, d]
    new_arr = torch.zeros(input_tensor.shape)
    for batchloop in range(input_tensor.shape[0]):
        for d in range(input_tensor.shape[1]):
            new_arr[batchloop, d] = torch.where(input_tensor[batchloop, d] == 1, d + 1, 0)
    return new_arr.sum(1).unsqueeze(1)


def trainer(args, config, network_settings, model, savepath):
    
    # Initializations
    if torch.cuda.is_available():
        print("CUDA available")
        if torch.cuda.device_count() > 1:
            print(f"{torch.cuda.device_count()} GPUs available!")
            # Use all available GPUs
            model = nn.DataParallel(model)
        device = torch.device("cuda") #cuda: gpu 0, cuda:1 gpu 1
    else:
        device = torch.device("cpu")
        print("CUDA not available")
    
    
    model.to(device)
    #print(model)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    #to check a specifc block in any of the networks
    #num_parameters_1 = sum(p.numel() for name, p in model.named_parameters() if name.startswith('vit'))
    #num_parameters_2 = sum(p.numel() for name, p in model.named_parameters() if name.startswith(''))
    #num_parameters_3 = sum(p.numel() for name, p in model.named_parameters() if name.startswith(''))
    #print(num_parameters_1)
    #print(num_parameters_2)
    #print(num_parameters_3)
    #print("\n")
    #print(num_parameters_1 + num_parameters_2 + num_parameters_3)



    # Parameters
    loss_function = config['loss_function']
    optimizer = config['optimizer']
    scheduler = config['scheduler']
    dataset_train = config['ds_train']
    dataset_val = config['ds_val']
    save_interval = config['save_interval']

    #TODO Works? Why?
    loss_function.to(device)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    
    #TODO Investigate if using more num_workers is worth it?
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, #Anmar: inital num_workers:16
                              worker_init_fn=worker_init_fn)


    #TODO Shuffle = False will save same image over multiple epochs, can see how it "evolves"
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=0, pin_memory=True,
                            worker_init_fn=worker_init_fn)

    max_iterations = args.max_epochs * len(train_loader)
    
    run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # For saving models 
    save_dir_model = os.path.join(args.save_path, "models", args.log_name)
    os.makedirs(save_dir_model, exist_ok=True)
    
    if not args.log_name:
        #prev: args.save_path + "/log.txt" #Anmar: filemode="w" resets the test_log.txt after every run, sometimes you want this i guess and sometimes not.
        logging.basicConfig(filename="/proj/berzelius-2023-86/users/x_anmka/thesis/TRUNet/text_logs/test_log.txt", filemode="w", level=logging.INFO, 
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        delete_folder_and_contents("/proj/berzelius-2023-86/users/x_anmka/thesis/TRUNet/tensorboard_logs/", "test_tb_log")
        writer = SummaryWriter('/proj/berzelius-2023-86/users/x_anmka/thesis/TRUNet/tensorboard_logs/test_tb_log')
    
    else:
        logging.basicConfig(filename= text_log_dir + run_id + "_" + args.log_name + ".txt", 
                            filemode="w", level=logging.INFO, 
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        writer = SummaryWriter(os.path.join(tensorboard_log_dir, run_id + "_" + args.log_name))
        
    logging.info("STARTING A NEW LOG")

    logging.info("\n--------------------------------------\n")
    
    logging.info("Training Details: \n {}".format(args))
    
    logging.info("\n--------------------------------------\n")
    
    logging.info("Network Setting Details: \n {}".format(network_settings))
    
    logging.info("\n--------------------------------------\n")
    
    logging.info("{} iterations per epoch. {} max iterations \n".format(len(train_loader), max_iterations))
    

    logging.info("Model: \n {} \n".format(model))
    logging.info("Num of trainalbe parmeters parameters: {}".format(num_parameters))

    logging.info("\n--------------------------------------\n")

    

    best_metric = -1
    best_metric_epoch = -1
    iter_num = 0
    
    ############################
    #         Training         #
    ############################
    
    for epoch in tqdm(range(args.start_epoch, args.max_epochs), desc = f"Epochs"):
        epoch_loss_train = 0
        dice_tmp_train = []
        model.train() #Set the model to training mode
        for i_batch, sampled_batch in enumerate(tqdm(train_loader, desc=f'Running training for epoch {epoch}')):
            #print(f"Train Minibatch index: {i_batch}")
            #Start logging at 1 :)
            iter_num += 1
            
            # get inputs and targets
            inputs, targets = sampled_batch['image'], sampled_batch['label']


       
            # here the input and target have the shape [batch, H, L, D]
            # so we need to add the channel dimension
            inputs, targets = inputs.unsqueeze(1), targets.unsqueeze(1)
            inputs, targets = inputs.to(device), targets.to(device)
            

            #Zero out gradients
            optimizer.zero_grad()
            #Forward pass
         
            #outputs, loss = model(inputs, targets, loss_function)
            outputs = model(inputs)

            
            # TODO Investigate if you even need to use softmax before calcualting loss and dice score??
                # To me it makes sense because target values is between 0 and 1, that's why we do it.
                # TODO Ask Lee
            
            # Because our network has a softmax in the decoder directly, can't be bothered to change this for now...
            if args.network not in ["ours", "ours_no_vit", "U-Net", "pure_vit"]:
                outputs = F.softmax(outputs, dim=1)


            targets = one_hot_encoder(targets, args.num_classes)

            
            loss_train = loss_function(outputs, targets)
            
            epoch_loss_train += loss_train.item()


            #TODO Ask Lee if outputs needs to be binarized using argmax or not??
            
            #Calculcate dice score
            dice_lee = dice_score(outputs, targets)
            dice_lee = dice_lee.item()
            dice_tmp_train.append(dice_lee)
            
            
                
            #Calculates gradients
            loss_train.backward()
          
            #Takes the step based on the gradients                  
            optimizer.step()
            
            #Current learning rate
            #TODO, WHY IS THIS LINE SO IMPORTANT??
            current_lr = optimizer.param_groups[0]['lr'] 

            
            #print(loss_train.item(), dice_lee)
            #TODO Maybe still needed, at least now in the begining to see that things work.
            #---LOGS PER MINIBATCH----
            
            #logging.info('iteration %d : Loss (Training) : %f' % (iter_num, loss_train.item()))
            #logging.info('iteration %d : Dice Lee (Training) : %f' % (iter_num, dice_lee))
            
            #writer.add_scalar('TRAIN_Loss_per_minibatch', loss.item(), iter_num)
            #Logging dice per minibatch
            #logging.info('iteration %d : Dice (Training) : %f' % (iter_num, average_dice_value))
            #writer.add_scalar('TRAIN_Dice_score_per_minibatch', average_dice_value, iter_num)
            #----------------------------------------------------------------------------------
            
            
        #Update learning per epoch
        scheduler.step()
        current_lr = optimizer.param_groups[-1]['lr']

        
        #Average loss
        epoch_loss_train = epoch_loss_train / len(train_loader)
       
        #Average dice
        metric_train = fmean(dice_tmp_train)
       
        #Logging learning rate (after change)
        
        #Logging loss, dice score and learning rate, per epoch
        writer.add_scalar('TRAIN_mean_loss_per_epoch', epoch_loss_train, epoch)
        writer.add_scalar('TRAIN_mean_dice_score_per_epoch', metric_train, epoch)
        writer.add_scalar('Learning_rate_per_epoch', current_lr, epoch)
        
        logging.info('\n Epoch %d: Learning rate %f' % (epoch, current_lr))
        logging.info('epoch %d : mean loss (Training) : %f' % (epoch, epoch_loss_train))
        logging.info('epoch %d : mean Dice (Training) : %f\n' % (epoch, metric_train))
        

        print(f"DONE WITH TRAINING LOOP FOR EPOCH {epoch}")
        
        ############################
        #        Validation        #
        ############################
        print("\n")
        print(f"STARTING VALIDATION LOOP FOR EPOCH {epoch}")

        model.eval()
        with torch.no_grad():
            dice_tmp_val = []
            epoch_loss_val = 0
            for i_batch, sampled_batch in enumerate(tqdm(val_loader, desc=f'Running validation for epoch {epoch}')):
                #print(f"Val Minibatch index: {i_batch}")


                # iter_num_val +=1
                # # get inputs and targets
                inputs, targets = sampled_batch['image'], sampled_batch['label']

                # #Adding the channel dimension
                inputs, targets = inputs.unsqueeze(1), targets.unsqueeze(1)
                inputs, targets = inputs.to(device), targets.to(device)
                
                # # Because we don't do any augmenations to val, so they still float64s
                # # They need to be float 32s. #TODO just add a check to transform instead.
                inputs = inputs.float()
                targets = targets.float()
                
                val_outputs = model(inputs)
                
                # Because our network has a softmax in the decoder directly, can't be bothered to change this for now...
                if args.network not in ["ours", "ours_no_vit", "U-Net", "pure_vit"]: 
                    val_outputs = F.softmax(val_outputs, dim=1)
           
                
                targets_one_hot = one_hot_encoder(targets, args.num_classes)
                
                loss_val = loss_function(val_outputs, targets_one_hot)
                epoch_loss_val += loss_val.item()
                
                
                #Dice score #TODO you're a fucking retard...
                dice_lee = dice_score(val_outputs, targets_one_hot)
                dice_lee = dice_lee.item()
                dice_tmp_val.append(dice_lee)       
                 
                      
                             
                
            #Average our the epoch loss over entire batch
            epoch_loss_val = epoch_loss_val / len(val_loader)

            # aggregate the final mean dice result
            metric_val = fmean(dice_tmp_val)
            
            
            # write to log
            writer.add_scalar('VAL_mean_loss_per_epoch', epoch_loss_val, epoch)
            writer.add_scalar('VAL_mean_dice_score_per_epoch', metric_val, epoch)
            
            logging.info('epoch %d : mean loss (Validation) : %f' % (epoch, epoch_loss_val))
            logging.info('epoch %d : mean Dice (Validation) : %f\n' % (epoch, metric_val))
            
            #Not used anywhere
            #metric_values.append(metric)

            #TODO Add early stopping as parser argument later
            if metric_val > best_metric:
                best_metric = metric_val
                best_metric_epoch = epoch #IDK Why... Don't ask.
                save_path = os.path.join(save_dir_model, f"best_metric_model_epoch_{best_metric_epoch}.pth")
                #torch.save(model.state_dict(), save_path)
                
                #TODO Investigate if this works as you expect
                save_checkpoint(model, optimizer, scheduler, best_metric_epoch, save_path)

                
                logging.info('\nSAVED new best metric model at epoch %d\n', best_metric_epoch)

                #---------FOR SAVING AND LOOKING IN ITK-SNAP---------
                #We save whenever the model gets better. And we always save the last in validation, which i think is ok.
                print("SAVING")
                case_name = sampled_batch['case_name'][0]  # Get the case_name from the sample
                info_ = f"{args.info_path}/{case_name[3:]}_myoseg.nii.gz" #3: to remove the "ct_" part.

                
                inference_savepath = os.path.join(savepath, 'inference_images_val', args.log_name, f'{case_name}_epoch_{epoch}')

                if not os.path.exists(inference_savepath):
                    os.makedirs(inference_savepath, exist_ok=True)
                
                
                
                inputs = torch.squeeze(inputs, dim=0)
                inputs = torch.squeeze(inputs, dim=0)

                targets = torch.squeeze(targets, dim=0)
                targets = torch.squeeze(targets, dim=0)

                #Since we treat it as multi-class segmentation, we use argmax intead of thresholding
                pred_class = torch.argmax(val_outputs, dim=1)
                pred_class = pred_class.squeeze(0)  # Remove the batch dimension
                pred_class = pred_class.float()


                # Save the original image
                original_filename = os.path.join(inference_savepath, f'original_epoch_{epoch}_{case_name}.nii.gz')
                save_as_nifti(inputs, original_filename, info_)
                #print(f"Saved original image for epoch {epoch+1} to {original_filename}")
                
                # Save the ground truth
                ground_truth_filename = os.path.join(inference_savepath, f'ground_truth_epoch_{epoch}_{case_name}.nii.gz')
                save_as_nifti(targets, ground_truth_filename, info_)
                #print(f"Saved ground truth for epoch {epoch+1} to {ground_truth_filename}")

                # Save the model output
                output_filename = os.path.join(inference_savepath, f'output_epoch_{epoch}_{case_name}.nii.gz')
                save_as_nifti(pred_class, output_filename, info_)
                #print(f"Saved output for epoch {epoch+1} to {output_filename}")

            
            logging.info(
                f"CURRENT EPOCH: {epoch} | current VAL mean dice: {metric_val:.4f} | Current learning rate {current_lr}"
                f"\n best VAL mean dice: {best_metric:.4f} at epoch: {best_metric_epoch} \n"
            )
            
            
            logging.info('\n\nDONE WITH EPOCH %d\n', epoch)
            #print("\n")
            #print(f"DONE WITH EPOCH {epoch+1}")
        
    writer.close()
    return "Training Finished!"
