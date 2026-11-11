# COMSOL Simulation and Experimental Data

## Simulated Data Generation

1. **Create a Geometric Model**:
   - Construct a 3D model of a steel rail (100 × 70 × 45 mm) in COMSOL Multiphysics, incorporating customizable defects.
   - Define defect parameters to generate diverse defect cases, as described in paper.

2. **Set Material Properties**:
   - Assign standard physical properties for steel rails, including:
     - Electrical conductivity: 1.3 × 10^7 S/m
     - Thermal conductivity: 48 W/m·K
     - Density: 8000 kg/m³
     - Other thermal and electromagnetic properties as specified in paper.

3. **Define Electromagnetic Induction Heating**:
   - Configure coil parameters in COMSOL:
     - Excitation: Current-driven coil.
     - Wire model: Single wire.
     - Coil current: Variable parameter for 600 ms sinusoidal pulses at 1–100 kHz.
   - Apply electromagnetic field equations to simulate eddy current induction heating, as detailed in paper.

4. **Grid Generation**:
   - Generate a free tetrahedral grid, with finer meshing in defect areas and regions of eddy current penetration depth.
   - Implement adaptive grid refinement strategies to capture defect details accurately, as described in paper.

5. **Set Up the Solver**:
   - Use a time-dependent solver with a simulation time span of 0 to 0.6 seconds and a fixed time step of 0.05 seconds.
   - Employ the Backward Differentiation Formula (BDF) for time integration to ensure accurate capture of thermal diffusion dynamics.
   - Set boundary conditions:
     - Thermal convection with ambient air, using a heat transfer coefficient of 5 W/m²·K.

6. **Export Results**:
   - Output simulation results in VTU format, including volumetric heat source (Q) and temperature (T) fields for each of the 10 excitation frequencies.
   - Export both surface and internal data for all simulation cases, as detailed in paper.
   - Use the provided Python script [`comsol_control.py`](comsol_control.py) to batch configure defect shapes and export results, enabling efficient generation of large-scale simulation samples.

7. **Alternative for Non-COMSOL Users**:
   - Preprocessed VTU files are available in the Kaggle dataset.
   - You can directly use our data in benchmark tasks.

The [COMSOL model](Hand-built-model-type2.mph) and batch script are included in this directory. To run the script, use a Python environment with `mph` and COMSOL Multiphysics installed. Set `src_dir` to the output directory and run `python comsol/comsol_control.py` from the repository root.

## Experimental Data

Download the two MATLAB files directly from the **Assets** section of the **Experimental Data** release on the [GitHub Releases page](https://github.com/sherlock-0327/ALETHEIA/releases). These infrared thermography recordings support calibration and comparison with the COMSOL simulations.

| Release asset | Specimen | Excitation frequency | Applied bias |
| --- | --- | --- | --- |
| `exp_data_1.mat` | Defect No. 1 | 25 kHz | None |
| `exp_data_2.mat` | Defect No. 2 | 25 kHz | None |

Each MATLAB file contains 300 surface-temperature frames (`Frame1` through `Frame300`). Each frame is a 480 × 640 matrix with a corresponding `FrameN_DateTime` timestamp variable.
