import os
import unittest
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

class TestDataLeakage(unittest.TestCase):
    def setUp(self):
        self.tr_path = os.path.join(PROCESSED_DIR, "train_patches.csv")
        self.val_path = os.path.join(PROCESSED_DIR, "val_patches.csv")
        self.te_path = os.path.join(PROCESSED_DIR, "test_patches.csv")

    def test_zero_image_id_leakage(self):
        tr = pd.read_csv(self.tr_path)
        val = pd.read_csv(self.val_path)
        te = pd.read_csv(self.te_path)

        tr_imgs = set(tr['image_id'])
        val_imgs = set(val['image_id'])
        te_imgs = set(te['image_id'])

        overlap_tr_val = tr_imgs.intersection(val_imgs)
        overlap_tr_te = tr_imgs.intersection(te_imgs)
        overlap_val_te = val_imgs.intersection(te_imgs)

        self.assertEqual(len(overlap_tr_val), 0, f"Leakage between train and val: {len(overlap_tr_val)} images!")
        self.assertEqual(len(overlap_tr_te), 0, f"Leakage between train and test: {len(overlap_tr_te)} images!")
        self.assertEqual(len(overlap_val_te), 0, f"Leakage between val and test: {len(overlap_val_te)} images!")

    def test_class_representation_across_splits(self):
        te = pd.read_csv(self.te_path)
        test_classes = set(te['defect_class'].unique())
        expected_classes = {'corrosion', 'crack', 'craze', 'hide_craze', 'surface_injure', 'thunderstrike'}
        self.assertEqual(test_classes, expected_classes, "All 6 defect classes must be present in the held-out test split!")

if __name__ == '__main__':
    unittest.main()