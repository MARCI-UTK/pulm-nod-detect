import os
import torch 
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from src.model.data import CropDataset
from src.model.rpn_loss import RegLoss, ClsLoss, ValClsLoss
from src.util.util import sample_anchor_boxes, corners_2_xyzd

from rpn import RPN, get_centers, get_anc_boxes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataPath = '/data/marci/dlewis37/luna16/'

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
 
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size, 
        shuffle=True
    )

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
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    model.to(f'cuda:{model.device_ids[0]}')

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    reg_loss_func = RegLoss()
    cls_loss_func = ClsLoss()
    val_cls_loss_func = ValClsLoss()

    train_losses = []
    val_losses = []

    train_accs = []
    val_accs = []

    epochs = 25
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

                optimizer.step()

                train_loss += loss.item()
                avg_cls_loss += cls_loss
                avg_reg_loss += bb_loss

                pred_binary = torch.where(pred_cls_scores > 0.5, 1., 0.)

                t_tp += ((pred_binary == 1.) & (y == 1.)).sum().item()
                t_tn += ((pred_binary == 0.) & (y == 0.)).sum().item()
                t_fp += ((pred_binary == 1.) & (y == 0.)).sum().item()
                t_fn += ((pred_binary == 0.) & (y == 1.)).sum().item()

                train_acc += (t_tp + t_tn) / (t_tp + t_tn + t_fp + t_fn)

                if e == 20: 
                    out_file = open('model_output.txt', 'w')

                    for i in range(len(pred_binary)): 
                        out_file.write(f'sample: {fname[i]}\n')

                        tp_idxs = ((pred_binary[i] == 1) & (y[i] == 1)).squeeze()
                        fp_idxs = ((pred_binary[i] == 1) & (y[i] == 0)).squeeze()

                        if len(tp_idxs) != 0: 
                            bb_pred_out = pred_anch_locs[i][tp_idxs]
                            bb_y_out = bb_y[i][tp_idxs]
                            anc_boxes = anc_box_list[tp_idxs]

                            out_file.write(f'true_positives:\n')
                            out_file.write(f'   gt_anchor_loc: {bb_y_out.tolist()}\n')
                            out_file.write(f'   pred_anchor_deltas: {bb_pred_out.tolist()}\n')
                            out_file.write(f'   anchor_box_locs: {anc_boxes.tolist()}\n')

                        if len(fp_idxs) != 0: 
                            bb_pred_out = pred_anch_locs[i][fp_idxs]
                            anc_boxes = anc_box_list[fp_idxs]

                            out_file.write(f'fale_positives:\n')
                            out_file.write(f'   pred_anchor_deltas: {bb_pred_out.tolist()}\n')
                            out_file.write(f'   anchor_box_locs: {anc_boxes.tolist()}\n')

                        out_file.write('\n')

                    out_file.close()

                pbar.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
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

                pred_binary = torch.where(pred_cls_scores > 0.5, 1., 0.)

                v_tp += ((pred_binary == 1.) & (y == 1.)).sum().item()
                v_tn += ((pred_binary == 0.) & (y == 0.)).sum().item()
                v_fp += ((pred_binary == 1.) & (y == 0.)).sum().item()
                v_fn += ((pred_binary == 0.) & (y == 1.)).sum().item()

                val_acc += (v_tp + v_tn) / (v_tp + v_tn + v_fp + v_fn)
                
        #scheduler.step()

        epoch_train_loss = train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        epoch_train_acc = train_acc / len(train_loader)
        train_accs.append(epoch_train_acc)

        epoch_val_acc = val_acc / len(val_loader)
        val_accs.append(epoch_val_acc)

        print(f'finished epoch {e}. Train Loss: {epoch_train_loss}. Val Loss: {epoch_val_loss}. Train Acc.: {epoch_train_acc}. Val Acc.: {epoch_val_acc}')
        print(f'train [tp, tn, fp, fn]: [{t_tp}, {t_tn}, {t_fp}, {t_fn}]')
        print(f'val [tp, tn, fp, fn]: [{v_tp}, {v_tn}, {v_fp}, {v_fn}]')

        #avg_reg_loss /= len(train_loader)
        #avg_cls_loss /= len(train_loader) 
        #print(f'avg cls loss: {avg_cls_loss}. avg reg loss: {avg_reg_loss}.')
    
    print(f'train losses: {train_losses}.')
    print(f'val losses: {val_losses}.')
    print(f'train accs: {train_accs}.')
    print(f'val accs: {val_accs}.')
    return 

if __name__ == "__main__": 
    main()