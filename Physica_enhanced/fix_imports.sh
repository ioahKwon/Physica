#!/bin/bash
# Fix all relative imports to absolute imports

cd /egr/research-zijunlab/kwonjoon/Code/Physica_enhanced

# Fix retargeting module
sed -i 's/from \.\.core\.smpl_model/from core.smpl_model/g' retargeting/joint_synthesis.py
sed -i 's/from \.\.core/from core/g' retargeting/retargeting_pipeline.py

# Fix optimization module  
sed -i 's/from \.\.core/from core/g' optimization/shape_optimizer.py
sed -i 's/from \.\.core/from core/g' optimization/pose_optimizer.py
sed -i 's/from \.\.utils/from utils/g' optimization/pose_optimizer.py

# Fix all __init__ files to use absolute imports
find . -name "__init__.py" -exec sed -i 's/from \./from /g' {} \;

echo "Fixed all imports!"
