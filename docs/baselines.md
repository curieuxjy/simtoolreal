# Baselines

## Kinematic Retargeting Baseline

### Hand Pose Extraction

To run this baseline, we first need to perform additional processing on the data, which includes (1) using SAM2 to extract hand masks and (2) using hamer_depth to extract 3D hand poses. This will give `hand_masks` and `hand_pose_trajectory` directories for each demo.

TODO: Add instructions for how to run SAM2 and hamer_depth.

### Visualize the Hand Pose Extraction and Retargeting

```
export DEMO_DIR=dextoolbench/data/hammer/claw_hammer/swing_down

python baselines/visualize_demo_with_hand.py \
--object-path assets/urdf/dex_tool_bench/hammer/claw_hammer/claw_hammer.urdf \
--object-poses-json-path $DEMO_DIR/poses.json \
--hand-poses-dir $DEMO_DIR/hand_pose_trajectory/ \
--visualize-hand-meshes \
--retarget-robot \
--save-retargeted-robot-to-file \
--rgb-path $DEMO_DIR/rgb/ \
--depth-path $DEMO_DIR/depth/ \
--cam-intrinsics-path $DEMO_DIR/cam_K.txt
```

This will generate a `retargeted_robot/<timestamp>.npz` file, which contains the retargeted robot poses.

### Replay the Retargeted Robot

You can then replay this trajectory with:

```
python deployment/replay_trajectory.py \
--file_path retargeted_robot/<timestamp>.npz
```

## Fixed Grasp Baseline

### Installation

Create a new conda environment for `pyroki`:

```
conda create --name pyroki_env python=3.10                            
conda activate pyroki_env

git clone https://github.com/chungmin99/pyroki.git
cd pyroki
pip install -e .

cd <this_repo>
pip install -e .
```

### Add Robot Spheres

Model the robot's collision geometry as a set of spheres:

```
python baselines/create_robot_spheres_interactive.py
```

### Test out TrajOpt

```
python baselines/test_trajopt_sharpa.py
```

### Run Trajopt on a Demo

```
export DEMO_DIR=dextoolbench/data/hammer/claw_hammer/swing_down
python baselines/visualize_demo_with_hand_trajopt.py \
--object-path assets/urdf/dex_tool_bench/hammer/claw_hammer/claw_hammer.urdf \
--object-poses-json-path $DEMO_DIR/poses.json \
--hand-poses-dir $DEMO_DIR/hand_pose_trajectory/ \
--visualize-hand-meshes \
--retarget-robot \
--retarget-robot-using-object-relative-pose \
--rgb-path $DEMO_DIR/rgb/ \
--depth-path $DEMO_DIR/depth/ \
--cam-intrinsics-path $DEMO_DIR/cam_K.txt
```

### Run on Real Robot

To run this on the real robot, we run the `rl_policy_node.py` and `goal_pose_node.py` to make the initial grasp, but we modify it to not update the goal pose beyond the first one.

```
python deployment/goal_pose_node.py \
--object_category hammer \
--object_name claw_hammer \
--task_name swing_down \
--success_threshold 0.0
```

```
python deployment/rl_policy_node.py \
--policy_path pretrained_policy \
--object_name claw_hammer
```

When the object has been lifted and the first goal pose has been reached, run the following to stop the policy and to save to a file `trajopt_inputs.json` that contains the inputs to TrajOpt:

```
pkill -USR1 -f rl_policy
```

Next, we run TrajOpt to generate the retargeted robot poses.

```
python baselines/run_trajopt.py
```

This will generate a `trajopt_outputs.json` file, which contains the retargeted robot poses. When this is generated, we can continue `rl_policy_node.py` to run the retargeted robot poses.