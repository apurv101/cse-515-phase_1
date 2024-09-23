# Download the dataset and prepare python venv
```bash
sudo apt update ; sudo apt install -y ffmpeg python3 python3-pip
python3 -m venv .venv
./downloadDataset.sh
./.venv/bin/pip3 install -r requirements.txt
```

# Task 1
```bash
./.venv/bin/python3 task_1.py --video ./target_videos/cartwheel/Bodenturnen_2004_cartwheel_f_cm_np1_le_med_0.avi  

./.venv/bin/python3 task_1.py --video ./target_videos/sword_exercise/Blade_Of_Fury_-_Scene_1_sword_exercise_f_cm_np1_ri_med_3.avi

./.venv/bin/python3 task_1.py --video ./target_videos/sword/AHF_longsword_against_Rapier_and_Dagger_Fight_sword_f_cm_np2_ri_bad_0.avi

./.venv/bin/python3 task_1.py --video ./target_videos/drink/CastAway2_drink_u_cm_np1_le_goo_8.avi
```

# Task 2a
```bash
./.venv/bin/python3 task_2.py --task 2a --folder ./hmdb51_org_stips
```

# Task 2b
```bash
./.venv/bin/python3 task_2.py --task 2b --video ./target_videos/cartwheel/Bodenturnen_2004_cartwheel_f_cm_np1_le_med_0.avi --stip ./hmdb51_org_stips/cartwheel/Bodenturnen_2004_cartwheel_f_cm_np1_le_med_0.avi.txt

./.venv/bin/python3 task_2.py --task 2b --video ./target_videos/sword_exercise/Blade_Of_Fury_-_Scene_1_sword_exercise_f_cm_np1_ri_med_3.avi  --stip ./hmdb51_org_stips/sword_exercise/Blade_Of_Fury_-_Scene_1_sword_exercise_f_cm_np1_ri_med_3.avi.txt

./.venv/bin/python3 task_2.py --task 2b --video ./target_videos/sword/AHF_longsword_against_Rapier_and_Dagger_Fight_sword_f_cm_np2_ri_bad_0.avi --stip ./hmdb51_org_stips/sword/AHF_longsword_against_Rapier_and_Dagger_Fight_sword_f_cm_np2_ri_bad_0.avi.txt

./.venv/bin/python3 task_2.py --task 2b --video ./target_videos/drink/CastAway2_drink_u_cm_np1_le_goo_8.avi --stip ./hmdb51_org_stips/drink/CastAway2_drink_u_cm_np1_le_goo_8.avi.txt
```

# Task 2c
```bash
./.venv/bin/python3 task_2.py --task 2c --video ./target_videos/cartwheel/Bodenturnen_2004_cartwheel_f_cm_np1_le_med_0.avi --stip ./hmdb51_org_stips/cartwheel/Bodenturnen_2004_cartwheel_f_cm_np1_le_med_0.avi.txt

./.venv/bin/python3 task_2.py --task 2c --video ./target_videos/sword_exercise/Blade_Of_Fury_-_Scene_1_sword_exercise_f_cm_np1_ri_med_3.avi  --stip ./hmdb51_org_stips/sword_exercise/Blade_Of_Fury_-_Scene_1_sword_exercise_f_cm_np1_ri_med_3.avi.txt

./.venv/bin/python3 task_2.py --task 2c --video ./target_videos/sword/AHF_longsword_against_Rapier_and_Dagger_Fight_sword_f_cm_np2_ri_bad_0.avi --stip ./hmdb51_org_stips/sword/AHF_longsword_against_Rapier_and_Dagger_Fight_sword_f_cm_np2_ri_bad_0.avi.txt

./.venv/bin/python3 task_2.py --task 2c --video ./target_videos/drink/CastAway2_drink_u_cm_np1_le_goo_8.avi --stip ./hmdb51_org_stips/drink/CastAway2_drink_u_cm_np1_le_goo_8.avi.txt
```

# Task 3
```bash
./.venv/bin/python3 task_3.py --video ./target_videos/cartwheel/Bodenturnen_2004_cartwheel_f_cm_np1_le_med_0.avi --output_file histogramCartwheel.txt

./.venv/bin/python3 task_3.py --video ./target_videos/sword_exercise/Blade_Of_Fury_-_Scene_1_sword_exercise_f_cm_np1_ri_med_3.avi --output_file histogramSwordExercise.txt

./.venv/bin/python3 task_3.py --video ./target_videos/sword/AHF_longsword_against_Rapier_and_Dagger_Fight_sword_f_cm_np2_ri_bad_0.avi --output_file histogramSword.txt

./.venv/bin/python3 task_3.py --video ./target_videos/drink/CastAway2_drink_u_cm_np1_le_goo_8.avi --output_file histogramDrink.txt
```

# Task 4
```bash
./.venv/bin/python3 task_4.py --directory ./target_videos --output_file demoVideoFeatures.pkl
```

# Task 5
```bash
./.venv/bin/python3 task_5.py --directory ./target_videos --key_name cartwheel/Bodenturnen_2004_cartwheel_f_cm_np1_le_med_0.avi --m 5

./.venv/bin/python3 task_5.py --directory ./target_videos --key_name sword_exercise/Blade_Of_Fury_-_Scene_1_sword_exercise_f_cm_np1_ri_med_3.avi --m 5

./.venv/bin/python3 task_5.py --directory ./target_videos --key_name sword/AHF_longsword_against_Rapier_and_Dagger_Fight_sword_f_cm_np2_ri_bad_0.avi --m 5

./.venv/bin/python3 task_5.py --directory ./target_videos --key_name drink/CastAway2_drink_u_cm_np1_le_goo_8.avi --m 5
```