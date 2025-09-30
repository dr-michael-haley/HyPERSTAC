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
