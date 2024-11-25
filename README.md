# Runtime Environment
- 2 NVIDIA 3090 Ti GPUs 
- Ubuntu 20.04.5
- CUDA 11.3 (with CuDNN of the corresponding version)
- Python 3.9
- PyTorch 1.11.0
- PyTorch Geometric 2.0.4
# Dataset
WikiSQL: [WikiSQL](https://github.com/salesforce/WikiSQL)

Spider: [Spider](https://github.com/taoyds/spider)

SpiderComGen: [Google Drive](https://drive.google.com/drive/folders/1xDgnK700hvQIiuniJu05Yqq-F8vEfEKt?usp=drive_link)

# Experiment
1. Step into the directory `src_code`:
    ```angular2html
    cd src_code
    ```

2. Pre-process the train/valid/test data:
   ```angular2html
   python s1_preprocessor.py
    ```

3. Run the model for training, validation, and testing:
    ```angular2html
   python s2_model.py
   ```
4. Eval the model:
    ```angular2html
    python s3_eval.py"
    ```

**Note that:** 
- All the parameters are set in `src_code/config.py`.
- If a model has been trained, you can set the parameter "train_mode" in `config.py` to "False". Then you can predict the testing data directly by using the model that has been saved in `data/Spider/model/`.
