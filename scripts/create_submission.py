# %% Setup
import torch
from torch.utils.data import Dataset, DataLoader
from fastai.vision.all import *
from tqdm import tqdm 
import segmentation_models_pytorch as smp
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
import pandas as pd
import cv2
from steel_segmentation.utils import mask2rle

# from segmentation_train.py to load the model
arch = "unet" # unet or fpn
encoder_name = "resnet34" # "efficientnet-b3"

path = Path(".") # "." if script, "../" if jupyter nb
train_path = path / "data" # where data dir is
model_dir = path / "models"
log_dir = path / "logs" 

model_weights_name = "unet_resnet34-bs_24-folds_1-only_faulty-fit_one_cycle-fp16-bce-epochs_30-lr_0.0003"

device = "cuda"
model_weights_file = model_dir / (model_weights_name + ".pth")

# %% loading weights
def get_model():
    if arch == "unet":
        model = smp.Unet(encoder_name=encoder_name, encoder_weights="imagenet", classes=4, activation=None)
    elif arch == "fpn":
        model = smp.FPN(encoder_name=encoder_name, encoder_weights="imagenet", classes=4, activation=None)

    weights = torch.load((model_weights_file))["model"]
    model.load_state_dict(weights)
    return model

model = get_model().to(device)
model.eval()

# %%
class SubmissionDataset(Dataset):

    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
        self.df = pd.read_csv(self.path / "data" / "sample_submission.csv")
        self.img_names = self.df["ImageId"].drop_duplicates().tolist()
        # self.test_img_path = self.path / "data" / "test_images"
        # self.img_names = get_image_files(self.test_img_path)
        self.transforms = alb.Compose([
            alb.Normalize(*imagenet_stats),
            ToTensorV2()
        ])
        self.num_samples = len(self.img_names)

    def __getitem__(self, index):
        img_path = self.path / "data" / "test_images" / self.img_names[index]
        img = np.array(PILImage.create(img_path))
        tensor_img = self.transforms(image=img)["image"]
        return img_path.name, tensor_img
    
    def __len__(self):
        return len(self.img_names)

def get_test_dl(path, bs, device="cpu", num_workers=4, shuffle=False, pin_memory=True, *args, **kwargs):
    return DataLoader(
        SubmissionDataset(path), 
        batch_size=bs, 
        shuffle=shuffle, 
        device=device,
        pin_memory=pin_memory,
        num_workers=num_workers,
        *args, **kwargs
    )

def predict_batch(images):
    return torch.sigmoid(model(images.to(device))).cpu().detach().numpy()

def post_process(pred, threshold, min_size_pixel_size):
    mask = cv2.threshold(pred, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size_pixel_size:
            predictions[p] = 1
            num += 1
    return predictions, num

# %%
def main():
    dl = get_test_dl(path, 12)
    threshold = 0.55
    min_pixel_sizes = [600, 600, 900, 2000]

    df_preds = []
    for _, batch in enumerate(tqdm(dl)):
        fnames, images = batch
        batch_preds = predict_batch(images)
        for fname, preds in zip(fnames, batch_preds):
            for cls, pred in enumerate(preds):
                pred, _ = post_process(pred, threshold, min_pixel_sizes[cls])
                rle = mask2rle(pred)
                name = fname + f"_{cls+1}"
                df_preds.append([name, rle])

    submission_df = pd.DataFrame(df_preds, columns=['ImageId_ClassId', 'EncodedPixels'])
    submission_df.to_csv(path/"data"/"submissions"/(model_weights_name+".csv"), index=False)

if __name__ == "__main__":
    main()