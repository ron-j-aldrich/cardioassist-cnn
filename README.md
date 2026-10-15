# CardioAssist CNN

This folder is the AI component of CardioAssist. It estimates left-ventricular ejection fraction (EF) from an apical four-chamber echocardiography video. The robotics component, ROS 2 Jazzy and Gazebo, is developed separately and is not installed here.

A probe moving in Gazebo does not create an ultrasound image. This component reads a video file. It does not send a prediction into ROS 2. The robotics computer can request one prediction over HTTP, as described under Connection to the robotic probe simulation.

## What is in this folder

| Path | Purpose |
|---|---|
| `third_party/dynamic` | Official EchoNet-Dynamic source, commit `e01d02947b20604b0ba1a427f3cf6f4890dc4e9c` |
| `src/cardioassist_cnn` | Thin wrapper: checkpoint loading, preprocessing, and inference |
| `models/r2plus1d_18_32_2_pretrained.pt` | Official EF checkpoint from release `v1.0.0` |
| `models/deeplabv3_resnet50_random.pt` | Official segmentation checkpoint from release `v1.0.0` |
| `models/normalization.json` | Training-split mean and standard deviation. Computed locally |
| `data/EchoNet-Dynamic/` | Local dataset. Not committed |
| `videos/` | Local videos. Empty. Not committed |
| `src/cardioassist_cnn/degradation.py` | Corrupts one video from a 3-DOF pose. Neutral pose is unchanged |
| `src/cardioassist_cnn/quality.py` | Predicts that pose severity. A view passes at or below 0.25 |
| `tests/test_degradation.py` | Checks that pose mapping on a synthetic pattern |
| `tests/test_synthetic_video_gate.py` | Decode, resize, and clip checks on a synthetic file. No EF percentage |
| `tests/test_official_clip_match.py` | Compares one synthetic clip with the official loader. No EF percentage |
| `.venv/` | Python 3.11 virtual environment. Not committed |

The dataset is stored locally at `data/EchoNet-Dynamic/` and is not committed. It contains `FileList.csv`, `VolumeTracings.csv`, and 10,030 videos in `Videos/`.

## Sources

- Code: https://github.com/echonet/dynamic
- Pinned commit: https://github.com/echonet/dynamic/commit/e01d02947b20604b0ba1a427f3cf6f4890dc4e9c
- Paper: Ouyang et al., *Nature*, 2020. https://doi.org/10.1038/s41586-020-2145-8
- Project site and research-use agreement: https://echonet.github.io/dynamic/
- Dataset access, linked from that site: https://stanford.redivis.com/datasets/66s1-2hsmzj5rn
- AIMI catalog and citation DOI `10.71718/yqp5-y078`: https://aimi.stanford.edu/datasets/echonet-dynamic-cardiac-ultrasound
- EF weights: https://github.com/echonet/dynamic/releases/download/v1.0.0/r2plus1d_18_32_2_pretrained.pt

The release was published by Bryan He on 23 January 2020. Its description is “Releasing weights for segmentation and EF prediction.” The file size on disk is 250,572,083 bytes, the same size as the GitHub release asset. GitHub did not publish a checksum for that asset. The SHA-256 computed locally is:

```text
dd8f8ef7b5d10b4e7994006eb49fd0344f06e1e2d86671b610c9d8343bd0339f
```

The same release contains `deeplabv3_resnet50_random.pt`, about 303 MB. That file segments the left ventricle. Clip-level EF inference does not use it. The paper’s beat-by-beat result does, because segmentation finds each contraction.

## Setup

Run these commands in macOS Terminal from this folder.

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e third_party/dynamic
python -m pip install --no-deps -e .
```

Check the install:

```bash
python -c "import torch, torchvision, echonet; print(torch.__version__, torchvision.__version__, echonet.__version__)"
```

On this machine that printed `2.14.1`, `0.29.1`, and `1.0.0`. `torch.backends.mps.is_available()` was `True`.

The upstream `requirements.txt` pins `torch==1.8.0` and `torchvision==0.9.0`. Those pins are from the authors’ older environment. They are not installed here. Current PyTorch was used, and the checkpoint was then compared with `torchvision.models.video.r2plus1d_18`.

To recreate the official source:

```bash
git clone https://github.com/echonet/dynamic.git third_party/dynamic
git -C third_party/dynamic checkout e01d02947b20604b0ba1a427f3cf6f4890dc4e9c
```

## Model

The network is torchvision’s R(2+1)D-18. The official script builds it and replaces the last layer with `Linear(512, 1)`:

```python
model = torchvision.models.video.r2plus1d_18(pretrained=pretrained)
model.fc = torch.nn.Linear(model.fc.in_features, 1)
```

That code is in `third_party/dynamic/echonet/utils/video.py`. The checkpoint filename records the chosen settings: `r2plus1d_18`, 32 frames, sampling period 2, and Kinetics-400 initialization before EF training.

R(2+1)D means each 3D convolution is split into a 2D spatial convolution and a 1D temporal convolution. The spatial filter looks inside one frame. The temporal filter looks across frames. “18” follows the ResNet-18 depth pattern. The final layer has one output because EF is one continuous number, trained with mean squared error against the EF label. There is no confidence score.

The saved dictionary includes `state_dict`, `frames=32`, `period=2`, and `epoch=43`. Every one of its 224 tensors has a `module.` prefix. The official script adds that prefix by wrapping the model in `DataParallel` when CUDA is available. This Mac has no CUDA. The wrapper removes only that prefix and then calls `load_state_dict(..., strict=True)`. After the prefix is removed, the checkpoint has no missing keys, no unexpected keys, and no shape mismatches. The saved output layer is shaped `(1, 512)`, which is the EF head.

`weights=None` is used when the architecture is created. That skips a separate Kinetics download because the EchoNet checkpoint replaces every parameter.

The file also stores `loss`, `best_loss`, and `r2` from the authors’ training run. In their script those fields are written from the validation phase. They are metadata inside the checkpoint. They are not an evaluation performed in this project.

## Preprocessing

An input clip has shape `(batch, 3, 32, 112, 112)` and dtype `float32`.

| Axis | Meaning |
|---|---|
| batch | One video at a time in this wrapper, so this is 1 |
| channel | RGB. OpenCV reads BGR, and the official loader converts to RGB |
| frames | 32 frames, taken every 2nd original frame |
| height, width | 112 and 112 |

The official steps, in order, are:

1. Decode the video in file order. Frame 0 is the first decoded frame.
2. Keep pixel values on a 0–255 scale. The official loader does not divide by 255.
3. Subtract the training-set mean and divide by the training-set standard deviation.
4. Select 32 frames with period 2. The clip therefore covers 64 original frames.
5. If the video is shorter than that, pad with zeros after normalization. The official dataset class treats that zero as the mean color.

Dataset videos are already 112×112. The project site says they were downsampled with cubic interpolation. This wrapper resizes with `cv2.INTER_CUBIC` only when a file is not already 112×112.

The training mean and standard deviation are not published as fixed numbers in the repository. `echonet video` computes them at runtime from 128 random training videos, after setting the random seed. This project will not invent those numbers. The earlier values `0.128` and `0.074` are not in the official code or the paper text checked for this setup, and they are not used.

This copy’s statistics are saved in `models/normalization.json` by the same function the authors call:

```bash
python -c "from cardioassist_cnn.preprocessing import write_training_normalization; write_training_normalization('data/EchoNet-Dynamic', 'models/normalization.json')"
```

The dataset directory must contain `FileList.csv`, `VolumeTracings.csv`, and `Videos/`. The official loader opens the tracing file while it builds the training split, including when the target is only EF, and then drops videos with fewer than two traced frames. The JSON file records `role` as `echonet_dynamic_training_split` and `pixel_range` as `0-255`. Statistics from a validation or test split are rejected, because those splits are for evaluation.

`clip_start` defaults to 0 so a run can be repeated. Official training draws a random start. On a synthetic 112×112 file, with the same start the official loader drew, `tensor_from_decoded_video` matched `Echo.__getitem__` exactly for both a 70-frame file and a 20-frame file. The 20-frame file is padded with zeros after normalization on both sides. The numbers used for that comparison were fixtures, and they were not saved.

If `FileList.csv` contains lowercase `Split` values, `root_with_uppercase_splits` builds a temporary directory with those values uppercased and with symlinks to `Videos/` and `VolumeTracings.csv`. The pinned file `third_party/dynamic/echonet/datasets/echo.py` is not edited. An uppercase `Split` column keeps the original directory.

The paper’s final number averages a prediction for every detected beat, using the segmentation model. A single clip from frame 0 is not that protocol.

## Run inference

From this folder, with the virtual environment activated:

```bash
python -m cardioassist_cnn.inference \
  --video videos/your-video.avi \
  --checkpoint models/r2plus1d_18_32_2_pretrained.pt \
  --normalization models/normalization.json \
  --device auto
```

`--device auto` uses MPS when it is available and CPU otherwise. `--device cpu` forces CPU. If an MPS forward pass fails, the same loaded model is moved to CPU and run once more.

The command prints a predicted EF percentage only after the checkpoint has loaded and the training-split normalization file has been read. The number is the raw regression output. It is not a diagnosis, and it does not include a confidence value.

From Python:

```python
from cardioassist_cnn import EFPredictor

predictor = EFPredictor(
    "models/r2plus1d_18_32_2_pretrained.pt",
    "models/normalization.json",
)
result = predictor.predict("videos/your-video.avi")
```

`EFPredictor` loads the weights once. `result` contains `status`, `ejection_fraction_percent`, `forward_seconds`, `device`, and `model_id`. `ejection_fraction_percent` is `None` unless `status` is `ok`.

## What has been checked

These are smoke checks on this Mac. They are not an accuracy evaluation.

- Python 3.11.14 imported PyTorch 2.14.1, torchvision 0.29.1, and echonet 1.0.0.
- MPS reported itself available. One all-zero tensor of shape `(1, 3, 32, 112, 112)` produced output shape `(1, 1)` on CPU and on MPS, and the two outputs matched. An all-zero tensor is not an echocardiogram, so that output is not an EF.
- The checkpoint loaded with `strict=True` after the `module.` prefix was removed.
- `python -m unittest tests.test_synthetic_video_gate` passed on this Mac. A temporary 6-frame MJPG file, 64×48, solid RGB `(220, 10, 20)`, was written with `echonet.utils.savevideo` and decoded with `echonet.utils.loadvideo`. The array was uint8 with shape `(3, 6, 48, 64)`, and the red channel stayed above the other two. The file is not an echocardiogram, and it was not kept.
- On separate in-memory arrays, cubic resize of a 64×48 solid frame produced `(3, frames, 112, 112)` and kept the solid color. A frame that was already 112×112 was left unchanged. Sampling 32 frames at period 2 kept indices `0, 2, …, 62`. A 4-frame array was padded with zeros after the real frames. Those arrays were not passed to the network.
- `python -m cardioassist_cnn.inference` on the synthetic file, with the official checkpoint, `--device cpu`, and no normalization file, printed `Status: missing_normalization` and `Decoded 6 frames at 64x48`. The process status was 2. The output contained no predicted EF. This path returns before resize, normalization, and clip sampling.
- A JSON file whose `role` was not `echonet_dynamic_training_split`, and one whose `pixel_range` was not `0-255`, were both rejected.
- `write_training_normalization` raised `FileNotFoundError` on an empty directory and on a directory that had `FileList.csv` and `Videos/` but no `VolumeTracings.csv`. It did not write `normalization.json`.
- `python -m unittest tests.test_official_clip_match` passed. A temporary 112×112 file was read by `echonet.datasets.Echo` with length 32 and period 2, and the same decoded pixels were passed to `tensor_from_decoded_video`. The arrays were identical for a 70-frame file and for a 20-frame file, with shape `(1, 3, 32, 112, 112)` and dtype `float32`. The comparison used fixture channel mean `[11, 22, 33]` and fixture standard deviation `[2, 4, 5]`. Those values were not written to `models/normalization.json`, and the checkpoint was not run. The loader returned the label stored in the synthetic `FileList.csv`.
- With `Split` set to `train`, the official loader reported zero videos. The temporary uppercase view made that row visible. A `FileList.csv` with no `Split` column made `write_training_normalization` raise `ValueError` before any statistics were computed.
- `EchoNet-Dynamic.zip` from the Stanford Redivis release was unpacked to `data/EchoNet-Dynamic/`. The zip is 7,038,260,018 bytes and lists 10,033 files with uncompressed size 7,926,466,430 bytes, the same uncompressed size shown on the release page. `FileList.csv` has 10,030 rows and columns `FileName,EF,ESV,EDV,FrameHeight,FrameWidth,FPS,NumberOfFrames,Split`. Every listed video is present. Split counts are `TRAIN` 7,465, `VAL` 1,288, and `TEST` 1,277. The official loader keeps 7,460 training videos; five training rows have fewer than two tracings.
- `write_training_normalization` wrote `models/normalization.json` from `Echo(split='train')`, NumPy seed 0, and 128 videos. `role` is `echonet_dynamic_training_split` and `pixel_range` is `0-255`. The per-channel mean is `[32.578743, 32.756481, 33.025372]` and the per-channel standard deviation is `[49.908653, 49.925823, 50.169361]`, on the 0–255 scale used by `loadvideo`. `read_normalization` accepted the file.
- One test-split video, the first `TEST` row in `FileList.csv`, was run with `--clip-start 0` and `models/normalization.json`. The file is `0X100CF05D141FF143.avi`: 248 frames, 112×112. The command decoded it to uint8 `(3, 248, 112, 112)` and the network input was `(1, 3, 32, 112, 112)`. On MPS the printed regression output was 61.54% in 0.737 s. The same command on CPU also printed 61.54%. The `EF` cell in `FileList.csv` for that file is 55.95179045. The later test-split run records the same prediction as 61.544898986816406.
- `python -m cardioassist_cnn.evaluation --split test --clip-start 0` scored one frame-0 clip for every test video the official loader keeps. `FileList.csv` has 1,277 `TEST` rows. The loader keeps 1,276. `0X5DD5283AC43CCDD1.avi` has no rows in `VolumeTracings.csv`, so it is dropped. All 1,276 scored videos returned status `ok` on MPS. MAE is 4.207, RMSE is 5.554, Pearson correlation is 0.892, and R² is 0.794, in ejection-fraction percentage points for the error terms. The per-video table is `output/test_frame0_predictions.csv`, which is gitignored because it repeats dataset labels.
- `models/deeplabv3_resnet50_random.pt` is the v1.0.0 segmentation asset, 317,401,511 bytes, SHA-256 `b32ea45b8200e11298ff4021166ff8d187204fb0c27a937bcdb4c518ffcbe1ed`. It loads into torchvision DeepLabV3 ResNet-50 with `strict=True` after the `module.` prefix is removed. `weights=None` and `weights_backbone=None` avoid a separate download. The saved output layer is shaped `(1, 256, 1, 1)`.
- `python -m cardioassist_cnn.beats --split test` averaged one EF clip around each segmented systole. The clip starts 32 frames before the systole frame, with a negative start clamped to 0, which is the rule in `scripts/beat_by_beat_analysis.R`. Of the 1,276 loader-kept test videos, 1,275 had at least one matched beat and 1, `0X350E5D4955052AFA.avi`, had none. There were no failed videos. On MPS, MAE is 4.054, RMSE is 5.323, Pearson correlation is 0.900, and R² is 0.809. The per-video table is `output/test_beat_predictions.csv`, also gitignored.

## What has not been measured

The frame-0 numbers and the beat-averaged numbers answer different procedures. The beat-averaged run is the procedure in the authors' beat script.

The authors report, on their held-out Stanford test set, MAE 4.1%, RMSE 5.3%, and R² 0.81. Their beat script comments list MAE 4.052, RMSE 5.325, and R² 0.809 for that procedure. This copy's beat-averaged run is MAE 4.054, RMSE 5.323, and R² 0.809 on 1,275 videos. The remaining difference includes the video with no detected beat and the one `TEST` row the loader drops for having no tracing.

The five older percentages from a previous experiment are not results of this project. Those video files are not in this folder.

This is an academic prototype. It is not a clinically validated diagnostic system. The EchoNet-Dynamic research-use agreement says the dataset has not been approved by the FDA and must not be used for diagnosis or patient care. The same limit applies to this prototype.

## When the dataset becomes necessary

Loading the checkpoint, decoding a video, resizing it, sampling 32 frames at period 2, matching the official clip on a synthetic file, and refusing to print an EF do not need the dataset. The clip comparison used fixture statistics, so it does not replace the training split.

Training-split statistics for this copy are in `models/normalization.json`. They were computed with:

```bash
python -c "from cardioassist_cnn.preprocessing import write_training_normalization; write_training_normalization('data/EchoNet-Dynamic', 'models/normalization.json')"
```

That call read `FileList.csv`, `VolumeTracings.csv`, and 128 training videos from `Videos/` after NumPy seed 0. One video path can now be passed to `python -m cardioassist_cnn.inference`. The printed number is the regression output for a single clip starting at frame 0. It is still not an accuracy measurement.

The frame-0 test above uses the held-out `TEST` rows and their `EF` labels. `evaluation.py` compares those two lists. It refuses the training split, because those videos were used for the normalization constants.

The beat-averaged test uses `models/deeplabv3_resnet50_random.pt`. That file is a separate release asset from the dataset zip. The run above is the result for this copy.

## Dataset, when you choose to obtain it

The dataset is separate from the code and the weights. Access requires registration and agreement to the Stanford research-use terms on https://echonet.github.io/dynamic/. The current portal link on that page is the Redivis dataset above. The public listing shows `EchoNet-Dynamic.zip` at about 7 GB behind an access application. Do not share the download link. Do not commit the files.

The paper describes 10,030 videos split into 7,465 train, 1,277 validation, and 1,288 test. This copy’s `FileList.csv` has the same total and the same training count. Its `VAL` count is 1,288 and its `TEST` count is 1,277. Each video has an EF label. `VolumeTracings.csv` stores human left-ventricle tracings.

The official `Echo` loader uppercases the requested split, but the line that would uppercase the `Split` column does not save its result. `root_with_uppercase_splits` compensates before `write_training_normalization` calls the loader. Two trace-only lines also call `np.int`, which NumPy 2 no longer provides. EF-only loading does not use those lines.

## Concepts for the project review

A convolution slides a small learned filter across an image and writes a feature map. Early filters respond to simple patterns such as edges. Later layers combine those maps into larger patterns. Pooling reduces the spatial size. An activation, such as ReLU, lets the network represent nonlinear relationships. A regression output is one number rather than a class label. Here that number is EF as a percentage.

A 2D CNN scores a single image. Ejection fraction depends on motion: it is `(EDV - ESV) / EDV`, the fraction of diastolic volume that leaves the ventricle during systole. A video CNN sees a sequence, so it can use that change. R(2+1)D does this without making every filter a full 3D cube. The spatial part processes each frame, and the temporal part mixes neighboring frames.

Training changes the weights using labeled videos. Validation is used to choose a checkpoint and settings. Testing measures the chosen model on videos that were held out. Inference applies a finished model to a new video and does not change the weights. Fine-tuning would continue training from these weights. This project does inference only.

A successful forward pass shows that the tensor flowed through the network. It does not show that the prediction is close to a cardiologist’s EF. That requires labeled videos and a fixed test split.

## Simulated acquisition

`degrade_video` is the Mac-side image step. A teammate’s probe does not render ultrasound. She sends a pose, three numbers `(tx, ty, angle)` in the range -1 to 1, and this function corrupts a prerecorded video.

`(0, 0, 0)` returns the original frames. Any other pose applies foreshortening from `angle`, an off-axis shift from `tx` and `ty`, a black occlusion band from `tx`, then blur, speckle, and frame dropout. The same pose always produces the same frames. `severity_label` is the root-mean-square of the three components. It is the training target. It is not the model’s prediction, and it is not an ejection fraction.

`SeverityNet` in `quality.py` is a separate network from the ejection-fraction model. It reads eight evenly spaced frames and predicts `severity_label`. A view is accepted when that prediction is at or below 0.25. The cutoff is a chosen operating point. It was not fit to ejection-fraction error.

Training reads `TRAIN` videos and keeps the epoch with the lower mean absolute error on `VAL`. Each training pose is drawn from the range -1 to 1. Each validation file uses one pose that stays the same if the run is repeated. `TEST` is not read. The score measures agreement with the synthetic severity label, not cardiologist image quality, and not ejection fraction.

```bash
.venv/bin/python -m cardioassist_cnn.quality \
  --data-dir data/EchoNet-Dynamic \
  --normalization models/normalization.json \
  --output models/severity_cnn.pt \
  --device auto
```

That command was run for 2 epochs, batch size 16, seed 0, on MPS. It fit 7,460 `TRAIN` videos. Epoch 2 had the lower `VAL` error and was saved: mean absolute error 0.067 on 1,288 videos (epoch 1 was 0.109). The target is the synthetic severity, which runs from 0 to 1. An error of 0.067 can still change a pass/fail decision for a view near the 0.25 cutoff. `TEST` was not used. `models/severity_cnn.pt` stays on this Mac and is not committed.

`POST /acquire` is the call the probe controller makes. The body is a file name already in the video directory and a pose. The reply contains `quality_score`, `passes`, and `threshold`. `ejection_fraction_percent` is filled in only when `passes` is true. The training label `severity_label` is not in the reply. A rejected view is still HTTP 200, so the controller can try another pose.

```bash
curl -s -X POST http://MAC_IP:8000/acquire \
  -H 'Content-Type: application/json' \
  -d '{"video": "0X100CF05D141FF143.avi", "pose": [0, 0, 0]}'
```

Point `--video-dir` at `data/EchoNet-Dynamic/Videos` when the source file is one of the local dataset videos. Those files stay on this Mac. The request sends the file name, not the video.

## Connection to the robotic probe simulation

The simulation stays on the other computer. This Mac does not run ROS 2 or Gazebo. A moving probe does not create an ultrasound image. The controller sends a pose. This Mac degrades the named local video, scores severity, and calculates ejection fraction only when the view passes.

```bash
.venv/bin/python -m cardioassist_cnn.serve \
  --video-dir data/EchoNet-Dynamic/Videos \
  --checkpoint models/r2plus1d_18_32_2_pretrained.pt \
  --quality-checkpoint models/severity_cnn.pt \
  --normalization models/normalization.json \
  --device auto
```

The process listens on port 8000. On the robotics computer, use this Mac’s address (`ipconfig getifaddr en0` on the Mac). Names that leave the video directory are rejected.

```bash
curl -s http://MAC_IP:8000/health
curl -s -X POST http://MAC_IP:8000/acquire \
  -H 'Content-Type: application/json' \
  -d '{"video": "0X100CF05D141FF143.avi", "pose": [0, 0, 0]}'
```

`POST /predict` still scores one unchanged file from frame 0 and does not apply a pose. The live ejection fraction, either way, is one frame-0 clip. The test-set figure to quote for the original videos is still the beat average: MAE 4.054, RMSE 5.323, R² 0.809 on 1,275 videos. A percentage after synthetic degradation is a different experiment, and it is a research prediction, not a clinical measurement. ROS 2 does not see this process by itself. The robotics controller has to make the HTTP request.

## Known limits

- The dataset is local and not committed. Frame-0 test scores are MAE 4.207, RMSE 5.554, Pearson correlation 0.892, and R² 0.794 on 1,276 videos.
- Beat-averaged test scores are MAE 4.054, RMSE 5.323, Pearson correlation 0.900, and R² 0.809 on 1,275 videos.
- Training-split normalization is in `models/normalization.json`.
- One clip starting at frame 0 is a different procedure from the beat average. The published MAE 4.1%, RMSE 5.3%, and R² 0.81 are the beat-averaged figures.
- Lowercase `Split` values are uppercased in a temporary view. The pinned loader is unchanged. `np.int` in the trace-only lines is still unfixed; EF-only loading does not use those lines.
- MPS and CPU both printed 61.54% for the one test clip above. That is one file, not a device-wide accuracy check.
- The severity model’s saved `VAL` mean absolute error is 0.067 on 1,288 videos after 2 epochs. That is agreement with the synthetic pose label, not an ejection-fraction score, and `TEST` was not used.
