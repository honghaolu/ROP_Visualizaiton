#Step 5: Calculate the evaluation indicators for the test set
import csv
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes):
    print('Num classes', num_classes)
    hist = np.zeros((num_classes, num_classes))

    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]

    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    for ind in range(len(gt_imgs)):

        pred = np.array(Image.open(pred_imgs[ind]))

        label = Image.open(gt_imgs[ind])
        # Determine if the label is a grayscale image
        if label.mode != 'L':
            label_gray = label.convert('L')

        else:
            label_gray = label
        label = np.array(label_gray)

        # If the image segmentation result is different from the size of the label, this image will not be calculated
        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        label = np.array([int(x) for x in label.flatten()])
        label[label == 255] = 1

        pred = np.array([int(x) for x in pred.flatten()])
        pred[pred == 255] = 1

        hist += fast_hist(label, pred, num_classes)
        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                ind,
                len(gt_imgs),
                100 * np.nanmean(per_class_iu(hist)),
                100 * np.nanmean(per_class_PA_Recall(hist)),
                100 * per_Accuracy(hist)
            )
            )

    #   Calculate the category wise mIoU values for all validation set images
    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)

    #   Output mIoU values by category
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
              + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2)) + '; Precision-' + str(
            round(Precision[ind_class] * 100, 2)))

    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(
        round(np.nanmean(PA_Recall) * 100, 2)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))
    return np.array(hist, np.int32), IoUs, PA_Recall, Precision


def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size=12, plt_show=True):
    fig = plt.gcf()
    axes = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val)
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values) - 1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()


def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size=12):
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs) * 100), "Intersection over Union", \
                   os.path.join(miou_out_path, "mIoU.png"), tick_font_size=tick_font_size, plt_show=True)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall) * 100), "Pixel Accuracy", \
                   os.path.join(miou_out_path, "mPA.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))

    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall) * 100), "Recall", \
                   os.path.join(miou_out_path, "Recall.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision) * 100), "Precision", \
                   os.path.join(miou_out_path, "Precision.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer_list = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))



if __name__ == "__main__":
    # Ground truth and prediction directories
    gt_dir = 'path_to_ground_truth_folder'
    pred_dir = 'path_to_pred_folder'

    # List of image names (without extensions)
    png_name_list = [os.path.splitext(f)[0] for f in os.listdir(gt_dir)]

    # Number of classes
    num_classes = 2  # Modify as needed
    name_classes = ['background', 'object']  # Modify as needed

    # Output directory for results
    miou_out_path = 'path_to_output_folder'
    if not os.path.exists(miou_out_path):
        os.makedirs(miou_out_path)

    # Compute mIoU and other metrics
    hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes)

    # Show and save results
    show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
