## Using DeepSpeed in Training of MiniGPT-4

We leverage the capabilities of DeepSpeed, a cutting-edge deep learning optimization library developed by Microsoft. 
DeepSpeed enhances efficiency and scalability in training large-scale deep learning models. 

Note: DeepSpeed has been tested only on first pretraining stage.

To initiate the training script, simply follow the prescribed steps outlined in 
[MiniGPT4_Train.md](MiniGPT4_Train.md)

You need to update [train_configs/minigpt4_stage1_pretrain.yaml](train_configs/minigpt4_stage1_pretrain.yaml) by setting
'dist_framework' to "deepspeed" and updating 'ds_config_path' to point to your deepspeed config file.

Sample deepspeed configuration file is available at [train_configs/deepspeed_config.json](train_configs/deepspeed_config.json)

Note: Nvidia Apex AMP cannot be combined with the fp16 mode in deepspeed. In addition it mode is not currently 
compatible with ZeRO; thus you need to ensure setting run.amp: False in [train_configs/minigpt4_stage1_pretrain.yaml](train_configs/minigpt4_stage1_pretrain.yaml)
in case you have fp16 or ZeRO enabled in your deepspeed configuration.
