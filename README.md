# Zero-Shot Protein Binder Classification with Jev

<h3 align="right">Colby T. Ford, Ph.D.</h3>

Using the Simple Jev API, this project evaluates the use of zero-shot classification of protein binders. This is to investigate the ability of Jev-based approaches to predict binder likelihoods with only predicted folding metrics.

## Input Data

Using the `design_summary` and `insilico_cofold_predictions` tables from the [Anthropic/claude-protein-binder-design](https://huggingface.co/datasets/Anthropic/claude-protein-binder-design) dataset on Hugging Face as input data, this project evaluates the zero-shot classification of protein binders.

Input data:
-  Predicted folding metrics (e.g., ipSAE, plDDT, sc_dockQ, n_interface_contacts)
- Generation information (e.g., generator, sequence_design_method)
- Target antigen (e.g., Nipah G, BBF-14)
- Generated sequence information (e.g., binder_length, epitope_n_residues)

Dependent variable:
-  Adaptv binding (or) Twist binding, indicating whether a protein binder is classified as a binder or non-binder.


Given that the folding process was performed numerous times for a given potential binder, these metrics were grouped by `uuid` and min, max, mean, standard deviation, and median values were calculated. Note: Only metrics derived from Boltz-2 predictions were used.


## Results

### Model: `featherless-ai/Qwen3.8-27B-classifier`

For the Adaptv binding class, the zero-shot classification approach using the Simple Jev API demonstrates high precision in the non_binder class and high recall in the binder class. Vice versa, precision is low in the binder class while recall is low in the non_binder class. Though the AUC score is relatively high, the overall accuracy remains relatively low.

- Total Scored: 1082
- Accuracy: 0.413
- ROC-AUC:  0.678

| **adaptyv_binder** | **precision** | **recall** | **f1-score** | **support** |
|:------------------:|:-------------:|:----------:|:------------:|:-----------:|
| non_binder         | 0.93          | 0.21       | 0.34         | 788         |
| binder             | 0.31          | 0.94       | 0.47         | 294         |


For the Twist binding class, the zero-shot classification approach using the Simple Jev API demonstrates lower performance compared to the Adaptv binding class, still with high precision in the non_binder class and high recall in the binder class, but low precision in the binder class and low recall in the non_binder class.

- Total Scored: 1087
- Accuracy: 0.381
- ROC-AUC:  0.639

|  **twist_binder**  | **precision** | **recall** | **f1-score** | **support** |
|:------------------:|:-------------:|:----------:|:------------:|:-----------:|
| non_binder         | 0.94          | 0.18       | 0.30         | 813         |
| binder             | 0.29          | 0.96       | 0.45         | 284         |
