# AUTOGENERATED! DO NOT EDIT! File to edit: dev_nbs/07_predict.ipynb (unless otherwise specified).

__all__ = ['pred_path', 'sub_path', 'FastPredict', 'TestDataset', 'get_test_dls', 'Predict']

# Cell
from .metadata import *
from .masks import *
from .datasets import *
from .dataloaders import *
from .metrics import *
from .trainer import *

import fastai
from fastai.vision.all import *
from fastai.metrics import *
from fastai.data.all import *

import os
import cv2
import pathlib
import numpy as np
import pandas as pd
import warnings
import zipfile
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset, sampler
import albumentations as alb

import segmentation_models_pytorch as smp

warnings.filterwarnings("ignore")

pred_path = path / "predictions"
sub_path = path / "submissions"
pred_path.mkdir(parents=True, exist_ok=True)
sub_path.mkdir(parents=True, exist_ok=True)

# Cell
class FastPredict:
    pred_mask_path = pred_path / "prediction_masks"

    def get_df_source_list(self):
        """Load `source` if it's a DataFrame instance."""
        tfm = ColReader("ImageId", pref=self.source_path)
        return L([tfm(o) for o in self.source.itertuples()])

    def get_path_source_list(self):
        """Load `source` if it's a pathlib.Path instance."""
        return get_image_files(self.source)

    def __init__(self,
                 source,
                 learner,
                 source_path:pathlib.Path=None):
        """
        A FastPredict object has:
        - source: key to load a list of images
        - learner: a fastai Learner object
        - source_path: the folder path to the images
        """
        self.source = source
        self.learner = learner
        self.source_path = source_path if source_path else train_path

        self.single_prediction = False

        if isinstance(self.source, pathlib.Path):
            self.img_paths = self.get_path_source_list()
        elif isinstance(self.source, pd.DataFrame):
            self.img_paths = self.get_df_source_list()
        elif isinstance(self.source, list):
            self.img_paths = self.source
        elif isinstance(self.source, str):
            self.single_prediction = True
            self.img_paths = L(self.source_path / self.source)

        self.elems = len(self.img_paths)

    def __call__(self, size_fold:int, threshold:float, min_size:int):
        """Call the object with prediction attributes,
        it calls `Predict.get_predictions`
        and returns the `Predict.df` DataFrame with RLEs."""
        self.size_fold = min([self.elems, size_fold])
        self.threshold = threshold
        self.min_size = min_size
        self.folds = self.elems // self.size_fold

        if (self.elems % self.size_fold) != 0:
            self.folds += 1

        self.df = self.get_predictions()
        return self.df

    def get_predictions(self):
        """Iterate through `Predict.folds`, predict the mask and
        get the RLEs in a DataFrame."""
        df_preds = []

        for fold in range(self.folds):
            start, end = fold*self.size_fold, (fold+1)*self.size_fold
            print(f"From {start} to {end} of {self.elems}")

            selected_imgs = self.img_paths[start:end]

            img_names = selected_imgs.map(Path).map(lambda x: x.name)
            pred_probs = self.predict(selected_imgs)

            tmp_df = self.get_RLEs(img_names, pred_probs)
            df_preds.append(tmp_df)

            torch.cuda.empty_cache()

        if len(df_preds) == 1:
            return df_preds[0]

        df = pd.concat(df_preds, axis=0, ignore_index=True)
        return df.fillna("")

    def predict(self, selected_imgs):
        """Get the predictions on the `selected_imgs`."""
        if self.single_prediction:
            pred_full_dec, pred_dec, out = self.learner.predict(selected_imgs[0])
            return out.unsqueeze(dim=0)

        test_dl = self.learner.dls.test_dl(test_items=selected_imgs)
        pred_probs,_,_ = self.learner.get_preds(dl=test_dl, with_decoded=True)

        return pred_probs

    def get_RLEs(self, img_names, pred_probs):
        """For each image in the batch to predict,
        returns a list of tuples with `(img_id, rle)`."""
        predictions = []

        for num_pred, t_pred in enumerate(pred_probs): # img in bs
            np_pred = t_pred.numpy()

            # iterate through class_id without class_id 0
            for class_id, prob in enumerate(np_pred[1:]):
                pred, num = self.post_process(prob)
                rle = mask2rle(pred)
                name = img_names[num_pred] + f"_{class_id+1}"
                predictions.append([name, rle])

        return pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])

    def post_process(self, probability):
        """
        Post processing of each predicted mask, components with lesser number of pixels
        than `min_size` are ignored.
        """
        mask = cv2.threshold(probability, self.threshold, 1, cv2.THRESH_BINARY)[1]
        num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
        predictions = np.zeros((256, 1600), np.float32)
        num = 0
        for c in range(1, num_component):
            p = (component == c)
            if p.sum() > self.min_size:
                predictions[p] = 1
                num += 1
        return predictions, num

    def save_masks(self, df_export_fname=None, make_zip=True):
        """
        If `df_export_fname` save predicted masks in
        `pred_path / df_export_fname` in CSV and the JSON.
        Otherwise the file are named "predictions.csv" and "predictions.json".
        Saving the zip of the masks if `make_zip`.

        Run this method after __call__.
        """
        if not hasattr(self, 'df'):
            print("Predictions missing. Call the Prediction object first!")
            raise KeyError("Missing self.df, first run self.__call__ to get predictions")

        self.df_masks = self.make_masks()

        if not df_export_fname:
            df_export_fname = "predictions"

        csv_name = pred_path/(df_export_fname + ".csv")
        json_name = pred_path/(df_export_fname + ".json")
        self.df_masks.to_csv(csv_name, index=False)
        self.df_masks.to_json(json_name, orient="table", indent=4)

        if make_zip:
            zip_fname = df_export_fname + ".zip"
            with zipfile.ZipFile(pred_path/zip_fname, 'w', zipfile.ZIP_DEFLATED) as zipf:
                self.zipdir(zipf)

    def make_masks(self):
        """
        Iterate through the RLEs in `Predict.df` and save the masks.
        Returns the `Predict.df_masks` DataFrame (if not `json`) with
        `columns=['ImageId', 'ClassId', 'Mask_path']`.
        """
        self.pred_mask_path.mkdir(parents=True, exist_ok=True)

        rows = []
        for row in self.df.itertuples():
            if row.EncodedPixels != '':
                img_id, class_id = row.ImageId_ClassId.split("_")
                img_id = img_id.split(".")[0]
                img_label = img_id + "_pred.png"
                img_path = self.pred_mask_path / img_label

                mask = rle2mask(row.EncodedPixels, 1, (256, 1600))
                im = Image.fromarray(mask)
                im.save(img_path)

                rows.append((img_id + ".jpg", class_id, img_label))

        return pd.DataFrame(rows, columns=['ImageId', 'ClassId', 'Mask_path'])

    def zipdir(self, ziph:zipfile.ZipFile, pfolder=None):
        """Zip the current `self.pred_mask_path` directory,
        otherwise zip the `path` dir if provided."""
        if not pfolder:
            pfolder = self.pred_mask_path

        if not isinstance(pfolder, pathlib.Path):
            pfolder = Path(pfolder)

        for root,dirs,files in os.walk(pfolder):
            root = Path(root)
            for file in files:
                fname = root / file
                ziph.write(filename=fname,
                           arcname =os.path.relpath(fname,fname.parent))

    def save_submission(self, fname="test_submission"):
        """Save submission file in CSV in `sub_path`.
        It adds always .csv at the end of `fname`."""
        if not hasattr(self, 'df'):
            raise KeyError("Call the object first to get the predictions")

        file_path = sub_path / (fname + '.csv')
        self.df.to_csv(file_path, index=False)

#    def plot(self, n:int=5, rand=False):
#        """Plot `n` elements in `self.img_paths`.
#        If `rand` prints shuffle images."""
#        if (not hasattr(self, 'df_masks'))|(not hasattr(self, 'df')):
#            return "Nothing to plot, first call make_masks"

#        path_list = self.img_paths.map(Path)

#        if rand:
            #path_list = get_perm_imgs_path(self.img_paths.map(Path), self.df_masks)
#            path_list = path_list.shuffle()

#        df = self.df.copy()
#        splitted_cols = df["ImageId_ClassId"].str.split("_", expand=True)
#        df["ImageId"], df["ClassId"] = splitted_cols[0], splitted_cols[1].astype("int64")

#        for p in path_list[:n]:
#            imgid, mask = make_mask(p, flatten=True, df=df)
#            img = cv2.imread(str(p))
#            plot_mask_image(imgid, img, mask)

# Cell
class TestDataset(Dataset):
    """Dataset for test prediction"""

    def __init__(self, root, df, mean, std):
        """
        A TestDataset loads the images from the `root` folder
        and the `ImageId`s from `df` with normalization params.
        """
        self.root = root
        self.fnames = df['ImageId'].unique().tolist()
        self.num_samples = len(self.fnames)
        self.transform = alb.Compose(
            [
                alb.Normalize(mean=mean, std=std, p=1),
                alb.pytorch.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        fpath = self.root / fname
        image = cv2.imread(str(fpath))
        image_tensor = self.transform(image=image)["image"]
        return fname, image_tensor

    def __len__(self):
        return self.num_samples

# Cell
def get_test_dls(
        root = None,
        df = None,
        best_threshold=0.5,
        num_workers=2,
        batch_size=4,
        min_size=3500,
        mean=None, std=None,
        *args, **kwargs
    ):
    """Returns dataloader for testing."""
    if not mean and not std: mean, std = imagenet_stats
    if df is None: df = test_df
    if not root: root = test_path

    return DataLoader(
        TestDataset(root, df, mean, std),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        *args, **kwargs
    )

# Cell
class Predict:
    def __init__(self, test_dl, model, device="cuda"):
        """
        `test_dl`: test Dataloader already loaded with `get_test_dls`
        """
        self.test_dl = test_dl
        self.source = self.test_dl.dataset.fnames
        self.source_path = self.test_dl.dataset.root
        self.img_paths = L([self.source_path / p for p in self.source])
        self.elems = len(self.img_paths)

        self.device = torch.device(device)
        self.model = model.to(self.device)

    def __call__(self, threshold:float, min_size:int):
        """
        Call the object with prediction attributes,
        it calls `Predict.get_predictions`
        and returns the `Predict.df` DataFrame with RLEs.
        """
        self.threshold = threshold
        self.min_size = min_size
        self.df = self.get_predictions()
        return self.df

    def get_predictions(self):
        """
        Iterate through `test_dl`, predict the mask and
        save the RLEs in a DataFrame.
        Returns the DataFrame.
        """
        df_preds = []
        for i, batch in enumerate(tqdm(self.test_dl)):
            fnames, images = batch

            batch_preds = self.predict_batch(self.model, images)

            for fname, preds in zip(fnames, batch_preds):
                for cls, pred in enumerate(preds):
                    pred, num = self.post_process(pred)
                    rle = mask2rle(pred)
                    name = fname + f"_{cls+1}"
                    df_preds.append([name, rle])

        return pd.DataFrame(df_preds, columns=['ImageId_ClassId', 'EncodedPixels'])

    def predict_batch(self, model, images):
        """Predict a single batch, returns the probabilities into numpy array"""

        batch_preds = torch.sigmoid(model(images.to(self.device)))
        batch_preds = batch_preds.detach().cpu().numpy()
        return batch_preds

    def post_process(self, probability):
        """
        Post processing of each predicted mask, components with lesser number of pixels
        than `min_size` are ignored.
        """
        mask = cv2.threshold(probability, self.threshold, 1, cv2.THRESH_BINARY)[1]
        num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
        predictions = np.zeros((256, 1600), np.float32)
        num = 0
        for c in range(1, num_component):
            p = (component == c)
            if p.sum() > self.min_size:
                predictions[p] = 1
                num += 1
        return predictions, num

    def save_submission(self, fname="test_submission"):
        """Save submission file in CSV in `sub_path`.
        It adds always .csv at the end of `fname`."""
        if not hasattr(self, 'df'):
            raise KeyError("Call the object first to get the predictions")

        file_path = sub_path / (fname + '.csv')
        self.df.to_csv(file_path, index=False)