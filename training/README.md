# Training

`colony_training.ipynb` trains a colony detector on your labelled data using
a free Google Colab GPU.

## Steps

1. Zip your `labeller/data/` folder into `data.zip`.
2. Open the notebook in [Google Colab](https://colab.research.google.com).
3. `Runtime → Change runtime type → T4 GPU → Save`.
4. Run the cells top to bottom. Upload `data.zip` when prompted.
5. The notebook sanitises filenames, splits train/val, trains, and shows the
   model's detections on held-out plates.
6. Download `best.pt` and place it at `app/model/best.pt`.

Do the run in one sitting — Colab disconnects idle sessions, which wipes the
trained files. Keep the tab active, and download `best.pt` as soon as
training finishes.

## Notes

- Uses YOLOv8/11-nano: fast, resists overfitting on small datasets.
- A fixed random seed makes runs reproducible.
- Retrain any time with more labelled plates; just replace `app/model/best.pt`.
