def compute_image_cls_metrics(args, all_preds, all_labels):
    class_num = args.class_num
    
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