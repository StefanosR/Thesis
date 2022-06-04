Information:

<< Write introduction/abstract here >>

Folders:

- experiments ->  120 logs produced by 60 scripts
- experiments_categorized -> organized logs for easier search
- flagged -> errors that were flagged from the user interface
- plots -> warning & error images, thumbnail, plots produced by the UI
- quantitative-explainability -> early work on QE part of the thesis
- recycling-bin -> old and irrelevant files

- robustness-evaluation 
     - explanations -> temp folder for explanations to be stored (while running the scripts)
     - notebooks -> early work on the scripts and debugging efforts
     - sripts -> the 66 scripts needed to run the experiments

Files:

main_global: automated script to perform multiple experiments

main_local: automated script to perform multiple experiments

surrogate_fidelity: notebook for measuring MASHAP fidelity between surrogate and original model

user_interface: UI implementation and server hosting (script and notebook format available)



Other info:

Log file format: Explainability Method (local/global + shap/lime/mashap) + Scenario/Script + Model used + Determinism + Dataset (+ Number of Loop)

e.g. -> log_GES_S3_MLPC_D_BD_2L.txt
     -> Global Explainability SHAP, Scenario 3 on deterministic MLPC, big dataset, 2nd loop