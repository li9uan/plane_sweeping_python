# plane_sweeping_python
This is a sample code to implement the basic plane-sweeping algorithm with Python and OpenCV.
The code has been tested with Python 3.9.12 and python opencv (4.5.5).

The execution is:

cd [path_to_code_folder]
python3 batch_compute_depth_maps.py ../fountain-P11

The output has been saved in tmp_result folder.

There are a few parameters in the batch_compute_depth_maps.py script that can be changed.
(1) number of sweeping depth - so far it's set at 3m ~ 12m with a step of 0.2m (https://github.com/li9uan/plane_sweeping_python/blob/main/code/batch_compute_depth_maps.py#L47)
(2) the original image resolution is downsampled to 1/8 to make the algorithm run reasonably fast (https://github.com/li9uan/plane_sweeping_python/blob/main/code/batch_compute_depth_maps.py#L183)
(3) related to (2), the local window size (half_window_size) is set to 4 (https://github.com/li9uan/plane_sweeping_python/blob/main/code/batch_compute_depth_maps.py#L38)
(4) number of frames used for computing depth for the reference view is set to 2 frames before the image and 2 frames after the image. (https://github.com/li9uan/plane_sweeping_python/blob/main/code/batch_compute_depth_maps.py#L213)
(5) the local patch score is computed with NCC, implemented via opencv, but other similarity metrics could be used as well (https://github.com/li9uan/plane_sweeping_python/blob/main/code/batch_compute_depth_maps.py#L104)

Some improvement could be implemented, given this basic implementation:
(1) multi-scale version (to prune a tighter bound of sweeping with local regions, instead of a large range from 3~12m)
(2) parallelization (e.g. multi-threading or GPU implementation)
(3) advanced algorithm for best depth choice given a sweeping volumn (spatial consistency and graph-based optimization e.g. BP, GC, could be implemented).
