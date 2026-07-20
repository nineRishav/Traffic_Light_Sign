# Project Changes Log

## [2026-07-20 03:45 PM IST]

### Description of Changes
* Created a unique output path generator `get_unique_save_path` in `traffic_light_detection.py` and `traffic_sign_speed_detection.py`.
* Integrated the auto-suffix logic (`_1`, `_2`, `_3`, etc.) on the `--save` parameter so successive runs of the scripts do not overwrite previous output files.

### Parameters & Status
* **Status:** Output suffix feature integrated successfully and verified.

## [2026-07-20 03:52 PM IST]

### Description of Changes
* Enhanced `traffic_light_detection.py` to dynamically resolve class names using `model.names` if they aren't pre-defined in the `ALL_CLASSES` taxonomy.
* Implemented case-insensitive text matching for the color mapping logic so that model-defined classes like `'red'`, `'yellow'`, and `'green'` map to their correct bounding box colors.
* Upgraded the output saving flow to create a dedicated run directory (e.g. `output_ZED_traffic_light_1`) for each run when `--save` is used, storing both the output `.mp4` video and every individual annotated frame as `.jpg` inside a `frames/` subfolder.

### Parameters & Status
* **Status:** Dynamic class mapping, custom color coding, and per-run directory frame saving features successfully integrated and logged.

## [2026-07-20 03:59 PM IST]

### Description of Changes
* Added functionality to automatically save an animated GIF of the processing run in `traffic_light_detection.py`.
* The script now accumulates every 3rd frame (resized to a memory-efficient width of 640px) and uses `imageio.mimsave` to generate the `.gif` file in the run directory alongside the `.mp4` video.

### Parameters & Status
* **Status:** Animated GIF saving functionality successfully integrated and logged.
