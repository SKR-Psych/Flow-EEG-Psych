# Raw Data Files
This folder contains raw EEG data files that are too large to be included in the GitHub repository.  
Please contact the project maintainer for instructions on accessing these files.

# Note on Data File Naming Convention

The data files in the data/raw/ folder follow a consistent naming pattern, which helps to identify the experimental conditions, subjects, and phases of the experiment. Understanding this pattern is crucial for organising and analysing the EEG data effectively.

General Naming Pattern:

Where:

SXX represents the subject number, e.g., S01, S02, etc.

YY represents some experimental condition or data type, such as:

RWEO - Likely refers to Relaxed with Eyes Open.

OLoop - Possibly Open Loop trials.

PreOL, PstOL, PreCL, PstCL - Could represent pre- or post-session data for Open Loop or Closed Loop experiments.

CL_Sil_50_100 - Likely refers to Closed Loop conditions with Silence, possibly representing different percentages (50% and 100%).

Explanation of Conditions:

Experimental Phases:

RWEO: Relaxed with Eyes Open – baseline recording.

OLoop: Open Loop – conditions where the participant was exposed to no direct feedback.

PreOL / PstOL: Refer to pre- and post-open loop phases, possibly used for baseline comparison.

CL_Sil_50_100: Indicates Closed Loop runs with conditions involving silence or feedback. The 50_100 may refer to levels or types of feedback (e.g., partial vs. full).

Example Filenames:

S01_B_RWEO_PreOL.mat:

Subject 01.

B may indicate a baseline or the phase within the experiment.

RWEO: Relaxed with Eyes Open.

PreOL: Pre Open Loop condition data.

S02_C_OLoop.mat:

Subject 02.

C may indicate a particular session or experimental run.

OLoop: Open Loop trial.

S20_F_CL_Sil_50_100.mat:

Subject 20.

F may represent a different trial or experimental variation.

CL: Closed Loop.

Sil_50_100: Indicates different levels of auditory feedback or conditions (e.g., silence vs. feedback levels).



