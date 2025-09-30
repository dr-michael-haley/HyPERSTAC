# MultiplexSSL

To run the ssl-training script, use e.g. the command
```
python3 ssl-training.py \
    --data-path <path-to-data> \
    --save-path <path-to-weights> \
    --batch-size 256 \
    --image-size 224 \
    --num-channels 8 \
    --epochs 100
```

To save the representations from a model, use e.g.
```
python3 save-representations.py \
    --data-path <path-to-patches> \
    --save-path <path-to-saved-weights> \
    --rep-save-path <path-to-save-representations> \
    --encoder-train-date <date-model-was-trained> \
    --num-channels 8 \
    --image-size 224
```

The file ```create-anndata-files.py``` is used to create anndata objects for downstream analysis, including clustering the representations. ```create-anndata-for-patient-subset.py``` does this for only the PD-L1 positive patients, and can be easily adapted to any patient group.

To train and save a lifelines survival model, use
```
risk_groups_test, cph = plot_latticeA("/path/to/data/")
save_outputs("/save/path/", risk_groups_test, cph)
```

For combining multiple panels, use
```
risk_groups_test, cph = plot_latticeA(["/path/to/data/one/", "/path/to/data/two/"])
save_outputs("/save/path/", risk_groups_test, cph)
```
