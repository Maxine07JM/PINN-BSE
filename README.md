# PINN-BSE

PINN-Battery-State-Estimation
A Physics-Informed Neural Network (PFFNN) for batteries, featuring automated capacity calibration and polarization differential tracking.

An advanced, production-ready Physics-Informed Neural Network (PINN) framework developed to resolve terminal voltage tracking and degradation diagnostics for next-generation organic dynamic battery systems (alkali metal-based).

In industrial AI-for-Science deployments, modeling highly non-linear electrochemical systems using pure deep learning traditionally fails due to three mathematical traps. This repository delivers four architecture-level solutions.

Tested over structural inverse operational profiles (Charge-then-Discharge dynamic sequences containing full current cutoff intervals), the model yields top-tier convergence performance.

| Mean Absolute Error (MAE) | 39.75 mV | Pixel-level tracking capability across sharp polarizations. | | Root Mean Squared Error (RMSE) | 60.67 mV | Exceptional robustness against localized structural variance. | | Coefficient of Determination ((R^2)) | 0.97241 | Accurately accounts for (97.241%) of nonlinear dynamic variance. |

Thermodynamic Validation at abrupt relaxation is also considered following the suggestions from GPT.

This repository is open-sourced under the MIT License.

Due to privacy and copyright restrictions, the original dataset is not included in this repository. Please contact the author to request access or other information.
