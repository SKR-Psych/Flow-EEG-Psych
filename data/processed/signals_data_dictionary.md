| **Column Name**                             | **Description**                                                                                     |
|---------------------------------------------|-----------------------------------------------------------------------------------------------------|
| **EEG Channels**                            |                                                                                                     |
| `Fp1`, `Fp2`, `Fz`, `Cz`, etc.              | EEG channel data representing electrical activity recorded from specific scalp locations.           |
|                                             |                                                                                                     |
| **Physiological Signals**                   |                                                                                                     |
| `ECG`                                       | Electrocardiogram signal measuring heart electrical activity.                                       |
| `HR`                                        | Heart rate in beats per minute (BPM).                                                               |
| `HR-delta`                                  | Change in heart rate since the last measurement.                                                    |
| `HR-delta-abs`                              | Absolute value of `HR-delta`.                                                                       |
| `HRV-pNN35`                                 | Heart rate variability measured as the percentage of successive intervals differing by more than 35 ms. |
| `RESP`                                      | Respiratory signal indicating breathing patterns.                                                   |
| `EDA`                                       | Electrodermal activity measuring skin conductance.                                                  |
| `EDAz`                                      | Z-score normalized EDA signal.                                                                      |
| `EDA-phasic`                                | Phasic component of EDA representing rapid changes.                                                 |
| `EDA-SMNA`                                  | Sympathetic Microneurography Activity, an EDA-derived measure.                                      |
| `EDA-tonic`                                 | Tonic component of EDA representing slow-moving baseline changes.                                   |
|                                             |                                                                                                     |
| **Eye-Tracking Data**                       |                                                                                                     |
| `PUP-L`, `PUP-R`                            | Pupil size measurements for the left and right eyes, respectively.                                  |
| `PORX`, `PORY`                              | Point of regard coordinates on the screen (X and Y axes).                                           |
| `PORX-L`, `PORY-L`                          | Point of regard for the left eye.                                                                   |
| `PORX-R`, `PORY-R`                          | Point of regard for the right eye.                                                                  |
| `EYE-Time`                                  | Timestamps from the eye-tracking system.                                                            |
| `EYE-Validity`                              | Validity code indicating the reliability of the eye-tracking data.                                  |
| `PUP-int-L`, `PUP-int-R`                    | Intermediate processed pupil data for left and right eyes.                                          |
| `PORX-int`, `PORY-int`                      | Intermediate processed point of regard data.                                                        |
| `PUP-conv05-4-L`, `PUP-conv05-4-R`          | Convolved pupil data with specific parameters (left and right eyes).                                |
| `PUP-logbp06-L`, `PUP-logbp06-R`            | Log-bandpass filtered pupil data (left and right eyes).                                             |
|                                             |                                                                                                     |
| **Behavioral Data**                         |                                                                                                     |
| `Sac-Amp-Avg`                               | Average saccade amplitude (eye movement speed).                                                     |
| `Sac-Angle-Abs-Avg`                         | Average absolute saccade angle.                                                                     |
| `Sac-Dur-Avg`                               | Average saccade duration.                                                                           |
| `Sac-VMax-Avg`                              | Average maximum saccade velocity.                                                                   |
| `Fix-Dur-Avg`                               | Average fixation duration.                                                                          |
| `Sac-rate-pmin`                             | Saccade rate per minute.                                                                            |
| `Fix-rate-pmin`                             | Fixation rate per minute.                                                                           |
| `Blink-rate-pmin`                           | Blink rate per minute.                                                                              |
|                                             |                                                                                                     |
| **Joystick Data**                           |                                                                                                     |
| `Joy-Pitch`, `Joy-Roll`                     | Joystick pitch and roll angles.                                                                     |
| `Joy-Pitch-acc`, `Joy-Roll-acc`             | Acceleration data for joystick pitch and roll.                                                      |
| `Joy-Pitch-pw`, `Joy-Roll-pw`               | Power of joystick pitch and roll movements.                                                         |
| `Joy-Pitch-pw016-3`, `Joy-Roll-pw016-3`     | Joystick pitch and roll power in the 0.16–3 Hz band.                                                |
| `Joy-Pitch-pw3plus`, `Joy-Roll-pw3plus`     | Joystick pitch and roll power above 3 Hz.                                                           |
| `Joy-Pitch-pw0016`, `Joy-Roll-pw0016`       | Joystick pitch and roll power below 0.16 Hz.                                                        |
|                                             |                                                                                                     |
| **Miscellaneous Signals**                   |                                                                                                     |
| `BCI-raw`                                   | Raw output from a Brain-Computer Interface system, if applicable.                                   |
| `FB-HB-0-1`, `FB-HB-0-1-nrm`                | Feedback or heartbeat signals (raw and normalized).                                                 |
| `FB-HB-raw`                                 | Raw feedback heartbeat signal.                                                                      |
| `FB-Mix-fact`                               | Mixing factor used in feedback calculations.                                                        |
| `Shm-2-2`, `Shm-raw`                        | Signals related to shared memory operations or placeholders.                                        |
|                                             |                                                                                                     |
| **Flight Simulation Data**                  |                                                                                                     |
| `Plane-pos-len`, `Plane-pos-height`         | Position data of a simulated plane in length and height dimensions.                                 |
| `Path-steps`, `Path-low`, `Path-high`       | Parameters describing the simulated flight path steps and boundaries.                               |
| `Path-width`                                | Width of the flight path corridor.                                                                  |
| `Path-avg`, `Path-avg-2s-smooth`            | Average path parameters, possibly smoothed over 2 seconds.                                          |
|                                             |                                                                                                     |
| **Head Movement Data**                      |                                                                                                     |
| `Head-raw-x`, `Head-raw-y`, `Head-raw-z`    | Raw head position data along X, Y, Z axes.                                                          |
| `Head-x`, `Head-y`, `Head-z`                | Processed head position data.                                                                       |
|                                             |                                                                                                     |
| **Experimental Conditions**                 |                                                                                                     |
| `Condition`                                 | Identifier for the experimental condition during the recording.                                     |
| `Flight-time`                               | Time elapsed in the flight simulation.                                                              |
| `Course-type`                               | Type of course or task in the simulation.                                                           |
| `FB-type`                                   | Feedback type provided during the experiment.                                                       |
| `Ring-type`                                 | Type of rings or markers used in the simulation.                                                    |
|                                             |                                                                                                     |
| **Timestamps and Synchronization**          |                                                                                                     |
| `LSL-time`                                  | Timestamps from the Lab Streaming Layer, used for synchronization.                                  |
