import numpy as np
import torch
from tqdm import tqdm

def compute_pcloud_partseg_metrics(all_preds, all_labels, class_dict):
    instance2parts, parts2instance, _ = class_dict
    instance_ious = {cls: [] for cls in instance2parts.keys()}

    for preds, labels in zip(all_preds, all_labels):
        labels = labels.numpy()
        batch_size = preds.shape[0]
        for i in range(batch_size):
            pred = preds[i, :]
            label = labels[i, :]
            cls = parts2instance[label[0]]
            part_ious = [0.0 for _ in range(len(instance2parts[cls]))]
            for l in instance2parts[cls]:
                if (np.sum(label == l) == 0) and (np.sum(pred == l) == 0):
                    part_ious[l-instance2parts[cls][0]] = 1.0
                else:
                    part_ious[l-instance2parts[cls][0]] = np.sum((label == l) & (pred == l)) / float(np.sum((label == l) | (pred == l)))
            instance_ious[cls].append(np.mean(part_ious))
            
    all_instance_ious = []
    for cls in instance_ious.keys():
        for iou in instance_ious[cls]:
            all_instance_ious.append(iou)
        instance_ious[cls] = np.mean(instance_ious[cls])
    class_mious = np.mean(list(instance_ious.values()))
    instance_mious = np.mean(all_instance_ious)
    return instance_ious, instance_mious, class_mious


def compute_pcloud_semseg_metrics(args, all_preds, all_labels):
    class_num = args.seg_class_num
        
    all_preds = all_preds.reshape(-1)
    all_labels = all_labels.reshape(-1)
    class_ious = []

    for cls in range(class_num):
        pred_mask = (all_preds == cls)
        target_mask = (all_labels == cls)

        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()

        if union == 0:
            iou = 0.0
        else:
            iou = intersection / union
        class_ious.append(iou)

    # Compute mean IoU for each class
    miou = np.mean(class_ious)
    return class_ious, miou

def compute_pcloud_cls_metrics(args, all_preds, all_labels):
    class_num = args.cls_class_num
    
    accuracy = (all_preds == all_labels).sum() / len(all_labels)

    precision_per_class = []
    recall_per_class = []

    for c in range(class_num):
        true_positives = ((all_preds == c) & (all_labels == c)).sum()
        predicted_positives = (all_preds == c).sum()
        actual_positives = (all_labels == c).sum()

        precision_c = true_positives / predicted_positives if predicted_positives > 0 else 0.0
        recall_c = true_positives / actual_positives if actual_positives > 0 else 0.0

        precision_per_class.append(precision_c)
        recall_per_class.append(recall_c)

    precision = sum(precision_per_class) / class_num
    recall = sum(recall_per_class) / class_num

    return accuracy, precision, recall

def part_seg_metrics(model, instances2parts, val_dataloader, epoch):
    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(50)]
        total_correct_class = [0 for _ in range(50)]
        shape_ious = {cat: [] for cat in instances2parts.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in instances2parts.keys():
            for label in instances2parts[cat]:
                seg_label_to_cat[label] = cat

        model = model.eval()

        for batch_id, (points, label, target) in tqdm(enumerate(val_dataloader), total=len(val_dataloader), smoothing=0.9):
            cur_batch_size, _, NUM_POINT = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            seg_pred, _ = model(points, label)
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[instances2parts[cat], :], 0) + instances2parts[cat][0]

            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(50):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(instances2parts[cat]))]
                for l in instances2parts[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - instances2parts[cat][0]] = 1.0
                    else:
                        part_ious[l - instances2parts[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float32))
        for cat in sorted(shape_ious.keys()):
            print('eval mIoU of {} {}'.format(cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

    print('Epoch {} test Accuracy: {}  Class avg mIOU: {}   Inctance avg mIOU: {}'.format(
        epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))