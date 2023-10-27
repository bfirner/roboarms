set terminal pngcairo enhanced

set xlabel "epoch"
set ylabel "error (m)"

set datafile separator ","

pos_0_query="grep target_xyz_position-0 train_log.txt | nl | awk '{print $1 \", \" $3 \" \" $4 \" \" $5}'"
pos_1_query="grep target_xyz_position-1 train_log.txt | nl | awk '{print $1 \", \" $3 \" \" $4 \" \" $5}'"
pos_2_query="grep target_xyz_position-2 train_log.txt | nl | awk '{print $1 \", \" $3 \" \" $4 \" \" $5}'"

set output "maximum_coordinate_errors.png"

set logscale y

plot "< ".pos_0_query u 1:(abs($4)) w lp title "x position maximum error", \
     "< ".pos_1_query u 1:(abs($4)) w lp title "y position maximum error", \
     "< ".pos_2_query u 1:(abs($4)) w lp title "z position maximum error", \
     "< ".pos_0_query u 1:(abs($2)) w lp title "x position average error", \
     "< ".pos_1_query u 1:(abs($2)) w lp title "y position average error", \
     "< ".pos_2_query u 1:(abs($2)) w lp title "z position average error"

joint_0_query="grep target_arm_position-0 train_log_joints.txt | nl | awk '{print $1 \", \" $3 \" \" $4 \" \" $5}'"
joint_1_query="grep target_arm_position-1 train_log_joints.txt | nl | awk '{print $1 \", \" $3 \" \" $4 \" \" $5}'"
joint_2_query="grep target_arm_position-2 train_log_joints.txt | nl | awk '{print $1 \", \" $3 \" \" $4 \" \" $5}'"
joint_3_query="grep target_arm_position-3 train_log_joints.txt | nl | awk '{print $1 \", \" $3 \" \" $4 \" \" $5}'"
joint_4_query="grep target_arm_position-4 train_log_joints.txt | nl | awk '{print $1 \", \" $3 \" \" $4 \" \" $5}'"

set ylabel "angle error ({\260})"
set output "maximum_joint_errors.png"

plot "< ".joint_0_query u 1:(abs($4)) w lp title "waist angle maximum error", \
     "< ".joint_1_query u 1:(abs($4)) w lp title "shoulder angle maximum error", \
     "< ".joint_2_query u 1:(abs($4)) w lp title "elbow angle maximum error", \
     "< ".joint_3_query u 1:(abs($4)) w lp title "wrist angle maximum error", \
     "< ".joint_4_query u 1:(abs($4)) w lp title "wrist rotation maximum error"
