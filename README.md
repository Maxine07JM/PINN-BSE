# PINN-BSE

PINN-Battery-State-Estimation

A Physics-Informed Neural Network for batteries, featuring automated capacity calibration and voltege tracking.

To resolve terminal voltage tracking and degradation diagnostics for next-generation dynamic organic battery systems (alkali metal-based).

In industrial AI-for-Science deployments, modeling highly non-linear electrochemical systems using pure deep learning traditionally fails due to three mathematical traps. This repository delivers four architecture-level solutions.

Tested over structural inverse operational profiles (Charge-then-Discharge dynamic sequences containing full current cutoff intervals), the model yields top-tier convergence performance.

| Mean Absolute Error (MAE) | 39.75 mV | Tracking capability across sharp polarizations. | | Root Mean Squared Error (RMSE) | 60.67 mV | Robustness against localized structural variance. | | Coefficient of Determination ((R^2)) | 0.97241 | Accounts for (97.241%) of nonlinear dynamic variance. |

Thermodynamic validation is also considered following the suggestions from GPT.  

This repository is open-sourced under the MIT License.

Due to privacy and copyright restrictions, the original dataset is not included in this repository. Please contact the author to request access or other information.

(This is my very first open-source project, and I am still learning the ropes! Code optimization, bug fixes, or general feedback are all highly welcome. If you see something that could be improved, please feel free to open an Issue or submit a Pull Request. Thank you. )
