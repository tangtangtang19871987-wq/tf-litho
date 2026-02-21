# 自定义梯度实现文档

## 数学基础

### 1. Abbe模型梯度推导

#### 前向传播
Abbe模型的前向传播可以表示为：
$$I(\mathbf{r}) = \frac{1}{N_s} \sum_{s=1}^{N_s} \left| \mathcal{F}^{-1} \left\{ H_s(\mathbf{f}) \cdot \mathcal{F} \left\{ M(\mathbf{r}) \right\} \right\} \right|^2$$

其中：
- $M(\mathbf{r})$ 是掩模透射函数
- $H_s(\mathbf{f})$ 是第$s$个光源点对应的光学传递函数
- $N_s$ 是光源点数量
- $\mathcal{F}$ 表示傅里叶变换

#### 梯度计算
对掩模$M$的梯度为：
$$\frac{\partial I}{\partial M^*} = \frac{2}{N_s} \sum_{s=1}^{N_s} \mathcal{F}^{-1} \left\{ H_s^*(\mathbf{f}) \cdot \mathcal{F} \left\{ I_s(\mathbf{r}) \cdot \mathcal{F}^{-1} \left\{ H_s(\mathbf{f}) \cdot \mathcal{F} \left\{ M(\mathbf{r}) \right\} \right\} \right\} \right\}$$

其中$I_s(\mathbf{r})$是第$s$个光源点产生的强度分布。

### 2. Hopkins模型梯度推导

#### 前向传播
Hopkins模型的前向传播为：
$$I(\mathbf{r}) = \sum_{k=1}^{K} w_k \left| \phi_k(\mathbf{r}) \ast M(\mathbf{r}) \right|^2$$

其中：
- $\phi_k(\mathbf{r})$ 是第$k$个TCC特征函数
- $w_k$ 是对应的权重
- $K$ 是TCC压缩后的组件数量

#### 梯度计算
对掩模$M$的梯度为：
$$\frac{\partial I}{\partial M^*} = 2 \sum_{k=1}^{K} w_k \cdot \phi_k^*(-\mathbf{r}) \ast \left( \phi_k(\mathbf{r}) \ast M(\mathbf{r}) \cdot I_k(\mathbf{r}) \right)$$

其中$I_k(\mathbf{r}) = \left| \phi_k(\mathbf{r}) \ast M(\mathbf{r}) \right|^2$。

### 3. 离焦效应处理

当考虑离焦时，光学传递函数包含相位项：
$$H_s(\mathbf{f}) = P(\mathbf{f}) \cdot e^{i \cdot \text{OPD}(\mathbf{f})}$$

其中$\text{OPD}(\mathbf{f})$是光程差。梯度计算中需要保持相位信息的共轭。

## TensorFlow实现细节

### 自定义梯度装饰器
使用`@tf.custom_gradient`装饰器实现自定义前向和反向传播：

```python
@tf.custom_gradient
def abbe_custom_gradient(mask, params):
    # 前向传播
    aerial_image = abbe_forward(mask, params)
    
    def grad_fn(grad_output):
        # 反向传播（梯度计算）
        mask_grad = abbe_backward(grad_output, mask, params)
        return mask_grad, None  # None for non-differentiable parameters
    
    return aerial_image, grad_fn
```

### 复数运算处理
TensorFlow中复数梯度需要特别注意：
- 使用`tf.math.conj()`进行复共轭
- 确保FFT/IFFT的正确性
- 处理实部和虚部的梯度传播

### 批处理支持
梯度函数需要正确处理批处理维度：
- 对batch维度进行适当的求和或平均
- 保持梯度形状与输入一致

## 性能优化建议

1. **内存复用**：避免创建不必要的中间张量
2. **FFT缓存**：对于重复计算的FFT结果进行缓存
3. **并行计算**：利用TensorFlow的自动并行化能力
4. **数据类型**：使用`tf.float32`而非`tf.float64`以提高性能

## 验证方法

### 数值梯度验证
使用有限差分法验证解析梯度的正确性：
$$\frac{\partial f}{\partial x} \approx \frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}$$

### 梯度检查点
在关键位置添加梯度检查：
- 梯度范数不应过大或过小
- 梯度方向应与预期物理行为一致
- 批处理梯度应正确聚合

## 注意事项

1. **相位一致性**：确保前向和反向传播中的相位处理一致
2. **归一化因子**：注意FFT/IFFT的归一化因子
3. **边界效应**：考虑周期性边界条件的影响
4. **数值稳定性**：避免除零和溢出问题