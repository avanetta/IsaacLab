0. record dataset on robot

###############
1. Add gripper to state (smooth opening possible)
python act_copy/modify_hdf5.py (with MODE="add_gripper_smooth")

2. Prepare dataset for annotation (add env name, object root_pose, ...)
python act_copy/modify_hdf5.py (with MODE="prepare_metadata")
###############

1.+2. python act_copy/preprocess_recorded_hdf5.py --input act_copy/data/dataset_real_dynamics.hdf5 --output act_copy/data/dataset_real_dynamics_preprocessed.hdf5 --task Isaac-Unplug-Franka-IK-Abs-Mimic-RGB-v0

3. Annotate demos with Mimic
python scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --device cuda --task Isaac-Unplug-Franka-IK-Abs-Mimic-RGB-v0 --auto --input_file ./act_copy/data/dataset_real_dynamics_preprocessed.hdf5 --output_file ./datasets/annotated_dataset.hdf5 --enable_cameras

4. Generate datasets with Mimic
python  scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --device cuda --enable_cameras --num_envs 1 --generation_num_trials 50 --input_file ./datasets/annotated_dataset.hdf5 --output_file ./datasets/generated_dataset_small_50demos.hdf5

4.1 In case of module import version problem execute in your cell
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

5. Play generated demos in the task environment
python scripts/imitation_learning/robomimic/play_hdf5_actions.py --task Isaac-Unplug-Franka-IK-Abs-Mimic-RGB-v0 --hdf5_path datasets/generated_dataset_small_50demos.hdf5 --enable_cameras --num_demos 10

###############
6. Clean generated datasets (reduce state to 9d, rename images, ...)
python act_copy/modify_hdf5.py (with MODE="clean_for_sim")

7. Merge original dataset and generated dataset:
python act_copy/modify_hdf5.py (with MODE="merge")
###############

6.+7. 
python act_copy/postprocess_generated_hdf5.py --gen ./datasets/generated_dataset_50demos_01std_005range.hdf5 --rec ./act_copy/data/dataset_real_dynamics_with_gripper_smooth_metadata.hdf5 --output ./act_copy/data/merged_data.hdf5

8. Train ACT policy
python3 act_copy/imitate_episodes.py --ckpt_dir act_copy/ckpt  --policy_class ACT --task_name sim_usbc \
--batch_size 16 --seed 1 --num_epochs 3000  --lr 1e-5 --chunk_size 32 --kl_weight 0.1 --context_length 4

9. Play trained ACT policy in sim environment
python scripts/imitation_learning/robomimic/play_cvae.py   --task Isaac-Unplug-Franka-IK-Abs-Mimic-RGB-v0   --checkpoint ./act_copy/ckpt/policy_best.ckpt   --data_path act_copy/ckpt/   --enable_cameras --context_length 4