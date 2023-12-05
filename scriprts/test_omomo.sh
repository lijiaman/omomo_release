python trainer_hand_foot_manip_diffusion.py \
--window=120 \
--batch_size=32 \
--project="./omomo_runs" \
--exp_name="stage2_manip_set1" \
--run_whole_pipeline \
--add_hand_processing \
--data_root_folder="./data" \
--checkpoint="./pretrained_models/stage1/model.pt" \
--fullbody_checkpoint="./pretrained_models/stage2/model.pt" \
--for_quant_eval 
