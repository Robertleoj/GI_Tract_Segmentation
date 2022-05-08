# Cell
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def make_seg_img(seg):
    img_tensor = torch.zeros(seg.size(0), seg.size(1), 3)
    for i in range(3):
        img_tensor[:,:, i] += seg == (i + 1)
    return img_tensor.numpy()

# Cell
def plot_sample(img_t, seg_mask):
    fig, ax = plt.subplots(figsize=(10, 10))
    # plt.subplots_adjust(bottom=0.25)
    # idx0 = 0

    im = plt.imshow(img_t.permute(1, 2, 0).numpy(), cmap='gray')

    seg_img = make_seg_img(seg_mask)    

    seg = plt.imshow(seg_img, cmap='jet', alpha=0.25)

    plt.show()

def plot_pair(img_t, real_mask, pred_mask, file=None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))

    # Show scan on both axes
    for ax, msk in zip(axs, (real_mask, pred_mask)):
        ax.imshow(img_t.permute(1, 2, 0).numpy(), cmap='gray')

        seg_img = make_seg_img(msk)        
        ax.imshow(seg_img, cmap='jet', alpha=0.25)

    axs[0].set_title("Real")
    axs[1].set_title("Predicted")

    if file is None:
        plt.show()
    else:
        plt.savefig(file)
        plt.close()


# Cell
def update_img(idx, im, seg, fig, imgs_t, segmentations):
    im.set(data=imgs_t[idx].unsqueeze(0).permute(1, 2, 0).numpy())
    seg_img = make_seg_img(segmentations[idx])
    seg.set(data=seg_img)
    fig.canvas.draw_idle()


def make_visual(imgs_t, segmentations):
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.25)
    idx0 = 0

    imgs_t = imgs_t.squeeze(0)

    im = plt.imshow(imgs_t[idx0].unsqueeze(0).permute(1, 2, 0).numpy(), cmap='gray')
    seg_img = make_seg_img(segmentations[idx0])
    seg = plt.imshow(seg_img, cmap='jet', alpha=0.25)

    allowed_values = range(imgs_t.size(0))

    ax_slice = plt.axes([0.25, 0.1, 0.65, 0.03])

    slice_slider = Slider(
        ax=ax_slice, 
        label="Slice", 
        valmin=0,
        valmax=imgs_t.size(0) -1,
        valinit=idx0,
        valstep=allowed_values,
        color="blue"
    )
    slice_slider.on_changed(lambda x: update_img(x, im, seg, fig, imgs_t, segmentations))
    plt.show()
    return slice_slider

def update_compare(idx, imgs_t, real, pred, fig, im_seg):
    for (im_o, seg_o), seg in zip(im_seg, (real, pred)):
        im_o.set(data=imgs_t[idx].unsqueeze(0).permute(1, 2, 0).numpy(), cmap='gray')
        seg_img = make_seg_img(seg[idx])
        seg_o.set(data=seg_img)
    fig.canvas.draw_idle()

def compare(imgs_t, real, pred):
    fig, axs = plt.subplots(1, 2,  figsize=(16, 10))
    plt.subplots_adjust(bottom=0.25)
    idx0 = 0

    imgs_t = imgs_t.squeeze(0)

    img_seg_obj = []
    for ax, s in zip(axs, (real, pred)):
        im = ax.imshow(imgs_t[idx0].unsqueeze(0).permute(1, 2, 0).numpy(), cmap='gray')
        seg_img = make_seg_img(s[idx0])
        seg = ax.imshow(seg_img, cmap='jet', alpha=0.25)
        img_seg_obj.append((im, seg))

    allowed_values = range(imgs_t.size(0))

    ax_slice = plt.axes([0.25, 0.1, 0.65, 0.03])

    slice_slider = Slider(
        ax=ax_slice, 
        label="Slice", 
        valmin=0,
        valmax=imgs_t.size(0) -1,
        valinit=idx0,
        valstep=allowed_values,
        color="blue"
    )
    slice_slider.on_changed(lambda x: update_compare(x, imgs_t, real, pred, fig, img_seg_obj))
    plt.show()
    return slice_slider

