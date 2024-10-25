| **Column Name** | **Description**                                                                                         |
|-----------------|---------------------------------------------------------------------------------------------------------|
| `init_index`    | Initial index of the event, possibly referencing an event list or marker.                               |
| `init_time`     | Time when the event was initiated, in seconds or as a timestamp.                                        |
| `latency`       | Latency of the event, possibly in milliseconds or sample points since the start of recording.           |
| `type`          | Type or label of the event (e.g., `EBlnk-Int-On`, `QRS`).                                               |
| `urevent`       | Unique identifier for the event, used for tracking across datasets.                                     |
| `sample_index`  | (If added) Sample index corresponding to `init_time`, useful for aligning events with signal data.      |
