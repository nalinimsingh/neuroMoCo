# dl-motion-correction

## Motion Simulation
A non-differentiable version of the motion simulator lives in `multicoil_motion_simulator.py`. 
Running this file as a script runs the function `generate_all_motion_scripts`. This function generates simulated pairs of motion corrupted and corrected k-space data along with the corresponding motion parameters.
The function writes these examples in .npz files.

A separate function within the same file, `generate_single_saved_motion_examples` produces a python generator which reads this data from the written .npz files and presents them in a format usable by the Keras `.fit_generator()` function.
