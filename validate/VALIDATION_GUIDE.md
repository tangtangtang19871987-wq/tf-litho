# TF-Litho 验证指南

## 目录
1. [验证目标](#验证目标)
2. [验证方法](#验证方法)
3. [运行验证](#运行验证)
4. [结果解读](#结果解读)
5. [常见问题诊断](#常见问题诊断)
6. [性能基准](#性能基准)

## 验证目标

本验证套件旨在确保 TensorFlow 实现 (`tf-litho`) 与原始 PyTorch 实现 (`TorchLitho-Lite`) 在以下方面保持一致：

- **数值精度**：输出结果的数值差异在可接受范围内
- **梯度计算**：梯度图的计算结果一致性
- **功能完整性**：所有核心功能正确实现
- **性能特性**：合理的性能表现

## 验证方法

### 1. 数值一致性验证
- 使用相同的输入数据和参数
- 比较输出图像的逐像素差异
- 计算相对误差和绝对误差
- 设置合理的容差阈值（通常为 1e-5 到 1e-3）

### 2. 梯度验证
- 使用相同的损失函数
- 比较梯度图的数值差异
- 验证梯度方向的一致性
- 检查梯度幅值的合理性

### 3. 功能验证
- 测试边界条件和异常情况
- 验证不同参数组合的正确性
- 确保批处理模式正常工作

### 4. 性能基准
- 测量单次模拟的执行时间
- 比较内存使用情况
- 评估不同输入尺寸的扩展性

## 运行验证

### 基本验证
```bash
# 运行完整的验证套件
python validate/validation_results.py

# 运行特定测试
python validate/validation_results.py --test numerical_consistency
python validate/validation_results.py --test gradient_validation
python validate/validation_results.py --test performance_benchmark
```

### 详细选项
```bash
# 显示详细输出
python validate/validation_results.py --verbose

# 保存结果到文件
python validate/validation_results.py --output validation_report.json

# 指定容差阈值
python validate/validation_results.py --tolerance 1e-4

# 仅运行快速测试
python validate/validation_results.py --quick
```

## 结果解读

### 成功标准
- **数值一致性**：相对误差 < 1e-3，绝对误差 < 1e-5
- **梯度一致性**：相对误差 < 1e-2，绝对误差 < 1e-4
- **功能测试**：所有测试用例通过
- **性能基准**：执行时间在合理范围内

### 输出格式
验证脚本会生成以下信息：

1. **摘要报告**：整体通过/失败状态
2. **详细指标**：
   - 最大绝对误差 (Max Absolute Error)
   - 平均绝对误差 (Mean Absolute Error)  
   - 最大相对误差 (Max Relative Error)
   - 平均相对误差 (Mean Relative Error)
3. **可视化结果**：保存对比图像用于人工检查
4. **性能统计**：执行时间和内存使用

### 文件输出
- `validation_results/`：包含所有验证结果
  - `numerical_comparison.png`：数值对比图
  - `gradient_comparison.png`：梯度对比图
  - `error_maps/`：误差分布图
  - `performance_stats.json`：性能统计数据
  - `validation_report.json`：完整验证报告

## 常见问题诊断

### 1. 数值差异过大
**可能原因**：
- FFT 实现差异（零频位置、归一化）
- 复数运算精度差异
- 边界处理方式不同

**解决方案**：
- 检查 `utils.py` 中的 FFT 实现
- 验证频率网格构建是否一致
- 确认复数类型和精度设置

### 2. 梯度计算不一致
**可能原因**：
- 自动微分实现差异
- 复合函数链式法则处理不同
- 数值稳定性问题

**解决方案**：
- 使用 `tf.debugging.assert_near()` 进行逐层调试
- 比较中间变量的梯度
- 考虑使用数值梯度验证

### 3. 性能问题
**可能原因**：
- TensorFlow 图优化未启用
- 内存分配策略不同
- 并行计算配置问题

**解决方案**：
- 启用 XLA 编译优化
- 调整内存增长策略
- 使用 `tf.function` 装饰器

### 4. 内存溢出
**可能原因**：
- 大尺寸输入导致内存不足
- 梯度计算占用过多内存
- 缓存未及时释放

**解决方案**：
- 减小输入尺寸进行测试
- 使用 `tf.config.experimental.set_memory_growth(True)`
- 分批处理大输入

## 性能基准

### 测试配置
- **硬件**：自动检测当前系统
- **输入尺寸**：[256, 512, 1024, 2048]
- **重复次数**：5 次（取平均值）
- **内存监控**：记录峰值内存使用

### 预期性能
| 输入尺寸 | PyTorch (ms) | TensorFlow (ms) | 差异 (%) |
|----------|--------------|----------------|----------|
| 256      | ~50          | ~60            | < 20%    |
| 512      | ~150         | ~180           | < 25%    |
| 1024     | ~500         | ~600           | < 30%    |
| 2048     | ~2000        | ~2400          | < 35%    |

### 优化建议
- 对于生产环境，考虑使用 `tf.function` 装饰器
- 启用混合精度训练（如果适用）
- 使用 TCC 预计算减少重复计算

## 贡献和扩展

如果您发现新的验证用例或改进现有验证，请提交 Pull Request。特别欢迎：

- 新的测试用例
- 更严格的容差设置
- 额外的性能基准
- 跨平台兼容性测试

---

**注意**：本验证套件假设您已经正确安装了所有依赖项，并且可以访问 `TorchLitho-Lite` 仓库用于参考比较。