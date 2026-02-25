#!/bin/bash
# 批量热力图生成器启动脚本

echo "========================================"
echo "  批量热力图生成器"
echo "========================================"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
pip3 install -q pandas numpy matplotlib seaborn 2>/dev/null

# 运行
python3 heatmap_generator.py "$@"
