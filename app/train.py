import os
import torch 
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from src.model.data import CropDataset
from src.model.rpn_loss import RegLoss, ClsLoss, ValClsLoss
from src.util.util import sample_anchor_boxes, corners_2_xyzd, apply_bb_deltas, nms

from rpn import RPN, get_centers, get_anc_boxes

#from clearml import Task
#from torch.utils.tensorboard import SummaryWriter 

#task = Task.init(project_name="Pulmonary Nodule Detection", task_name="Model Logging")
#writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataPath = '/data/marci/dlewis37/luna16/'

# Initialize all non-RPN layers
def weight_init(m): 
    if isinstance(m, torch.nn.Conv3d): 
        torch.nn.init.xavier_uniform_(m.weight) 
    elif isinstance(m, torch.nn.Linear): 
        torch.nn.init.xavier_uniform_(m.weight)

def write_epoch_cm(cm, epoch, file): 
    file.write(f'epoch {epoch}. cm: {cm}\n')

def write_epoch_vis_result(pred_binary, y, pred_locs, gt_anc_locs, anc_box_list, epoch, fname, out_file):

    out_file.write(f'epoch: {epoch}\n')

    for i in range(len(pred_binary)): 
        out_file.write(f'sample: {fname[i]}\n')

        tp_idxs = ((pred_binary[i] == 1) & (y[i] == 1)).squeeze()
        fp_idxs = ((pred_binary[i] == 1) & (y[i] == 0)).squeeze()

        if len(tp_idxs) != 0: 
            bb_pred_out = pred_locs[i][tp_idxs]
            bb_y_out = gt_anc_locs[i][tp_idxs]
            anc_boxes = anc_box_list[tp_idxs]

            out_file.write(f'true_positives:\n')
            out_file.write(f'   gt_anchor_loc: {bb_y_out.tolist()}\n')
            out_file.write(f'   pred_anchor_deltas: {bb_pred_out.tolist()}\n')
            out_file.write(f'   anchor_box_locs: {anc_boxes.tolist()}\n')

        if len(fp_idxs) != 0: 
            bb_pred_out = pred_locs[i][fp_idxs]
            anc_boxes = anc_box_list[fp_idxs]

            out_file.write(f'false_positives:\n')
            out_file.write(f'   pred_anchor_deltas: {bb_pred_out.tolist()[:32]}\n')
            out_file.write(f'   anchor_box_locs: {anc_boxes.tolist()[:32]}\n')

        out_file.write('\n')

def main(): 
    img_paths = [os.path.join(dataPath, 'dataset', f) for f in os.listdir(os.path.join(dataPath, 'dataset'))]
    label_paths = [os.path.join(dataPath, 'rpn_labels', f) for f in os.listdir(os.path.join(dataPath, 'rpn_labels'))]

    train_img_idxs = int(len(img_paths) * 0.8)
    train_img_paths = img_paths[:train_img_idxs - 1]
    val_img_paths   = img_paths[train_img_idxs:]

    train_label_idxs = int(len(img_paths) * 0.8)
    train_label_paths = label_paths[:train_label_idxs - 1]
    val_label_paths   = label_paths[train_label_idxs:]

    train_data = CropDataset(img_paths=train_img_paths, label_paths=train_label_paths)
    val_data   = CropDataset(img_paths=val_img_paths, label_paths=val_label_paths)
    batch_size = 32
    
    # 708 positive samples in training set 
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size, 
        shuffle=True
    )

    # 159 positive samples in validation set 
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size, 
        shuffle=True
    )

    centers_list = get_centers(0, 0)

    anc_box_list = get_anc_boxes(centers=centers_list)
    anc_box_list = [corners_2_xyzd(x) for x in anc_box_list]
    anc_box_list = torch.tensor(anc_box_list)

    model = RPN(128, 512, 3)
    model.apply(weight_init)
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    model.to(f'cuda:{model.device_ids[0]}')

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    reg_loss_func = RegLoss()
    cls_loss_func = ClsLoss()
    val_cls_loss_func = ValClsLoss()

    train_losses = []
    val_losses = []

    train_accs = []
    val_accs = []

    out_file = open('model_data.txt', 'w')
    cm_out_file = open('cm.txt', 'w')

    epochs = 100
    for e in range(epochs): 

        train_loss = 0
        avg_cls_loss = 0
        avg_reg_loss = 0
        val_loss = 0

        val_acc = 0
        train_acc = 0

        t_tp = 0
        t_tn = 0
        t_fp = 0
        t_fn = 0

        v_tp = 0
        v_tn = 0
        v_fp = 0
        v_fn = 0
        
        model.train()
        with tqdm(train_loader) as pbar:
            for idx, data in enumerate(pbar):
                fname, x, y, bb_y = data

                # This is used for logging 
                fname = list(fname)

                # Make labels correct size 
                y = y.unsqueeze(1)

                x = x.to(f'cuda:{model.device_ids[0]}')
                y = y.to(f'cuda:{model.device_ids[0]}')
                bb_y = bb_y.to(f'cuda:{model.device_ids[0]}')
                            
                optimizer.zero_grad()

                pred_anch_locs, pred_cls_scores = model(x)

                sampled_pred, sampled_target = sample_anchor_boxes(pred_cls_scores, y, f'cuda:{model.device_ids[0]}')
    
                cls_loss = cls_loss_func(sampled_pred, sampled_target, batch_size)
                bb_loss = reg_loss_func(y, pred_anch_locs, bb_y, batch_size)

                loss = cls_loss + bb_loss

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                avg_cls_loss += cls_loss
                avg_reg_loss += bb_loss

                pred_binary = torch.where(pred_cls_scores > 0.7, 1., 0.)

                t_tp += ((pred_binary == 1.) & (y == 1.)).sum().item()
                t_tn += ((pred_binary == 0.) & (y == 0.)).sum().item()
                t_fp += ((pred_binary == 1.) & (y == 0.)).sum().item()
                t_fn += ((pred_binary == 0.) & (y == 1.)).sum().item()

                #write_epoch_vis_result(pred_binary, y, pred_anch_locs, bb_y, anc_box_list, e, fname, out_file)

                if e == 90: 
                    pred_cls_scores = pred_cls_scores.squeeze()
                    pred_cls_scores = torch.sigmoid(pred_cls_scores)
                    y = y.squeeze()

                    proposal_cls_scores  = torch.zeros(size=(pred_cls_scores.shape[0], 4000))
                    proposal_labels      = torch.zeros(size=(pred_cls_scores.shape[0], 4000))
                    proposal_pred_deltas = torch.zeros(size=(pred_cls_scores.shape[0], 4000, 4))
                    proposal_gt_deltas   = torch.zeros(size=(pred_cls_scores.shape[0], 4000, 4))
                    anc_boxs             = torch.zeros(size=(pred_cls_scores.shape[0], 4000, 4))

                    for i in range(len(x)): 
                        valid_idxs = (y[i] != -1)
                        _, sorted_idxs = torch.sort(pred_cls_scores[i][valid_idxs], descending=True)
                        idxs = sorted_idxs[:4000]

                        proposal_cls_scores[i] = pred_cls_scores[i][idxs]
                        proposal_labels[i] = y[i][idxs]
                        proposal_pred_deltas[i] = pred_anch_locs[i][idxs]
                        proposal_gt_deltas[i] = bb_y[i][idxs]
                        anc_boxs[i] = anc_box_list[idxs]

                    final_idxs = nms(proposal_cls_scores, proposal_labels, proposal_pred_deltas, proposal_gt_deltas, anc_boxs)

                    for i in range(len(proposal_cls_scores)): 
                        print(len(final_idxs[i]))

                pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

                #writer.add_scalar('Training Loss', loss.item(), e * len(train_loader) + idx)

        write_epoch_cm(cm=[t_tp, t_tn, t_fp, t_fn], epoch=e, file=cm_out_file)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for idx, data in enumerate(val_loader):
                fname, x, y, bb_y = data

                y = y.unsqueeze(1)

                x = x.to(f'cuda:{model.device_ids[0]}')
                y = y.to(f'cuda:{model.device_ids[0]}')
                bb_y = bb_y.to(f'cuda:{model.device_ids[0]}')

                pred_anch_locs, pred_cls_scores = model(x)
    
                cls_loss = val_cls_loss_func(pred_cls_scores, y, batch_size)
                bb_loss = reg_loss_func(y, pred_anch_locs, bb_y, batch_size)

                loss = cls_loss + bb_loss 
                val_loss += loss.item()

                #writer.add_scalar('Validation Loss', loss.item(), e * len(val_loader) + idx)

                pred_binary = torch.where(pred_cls_scores > 0.7, 1., 0.)

                v_tp += ((pred_binary == 1.) & (y == 1.)).sum().item()
                v_tn += ((pred_binary == 0.) & (y == 0.)).sum().item()
                v_fp += ((pred_binary == 1.) & (y == 0.)).sum().item()
                v_fn += ((pred_binary == 0.) & (y == 1.)).sum().item()
                
        scheduler.step()

        epoch_train_loss = train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f'finished epoch {e}. Train Loss: {epoch_train_loss}. Val Loss: {epoch_val_loss}.')
        print(f'train [tp, tn, fp, fn]: [{t_tp}, {t_tn}, {t_fp}, {t_fn}]. val [tp, tn, fp, fn]: [{v_tp}, {v_tn}, {v_fp}, {v_fn}].')

    out_file.close()
    cm_out_file.close()

    print(f'train losses: {train_losses}.')
    print(f'val losses: {val_losses}.')
    print(f'train accs: {train_accs}.')
    print(f'val accs: {val_accs}.')

    #writer.close()

    return 

if __name__ == "__main__": 
    main()