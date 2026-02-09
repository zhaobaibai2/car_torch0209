# Buffer数据导入Phase2训练指南

## 方案一：修改配置文件（推荐）

### 单个文件加载：
```yaml
training:
  demo_data_path: "/path/to/your/existing/buffer.pkl"  # 单个buffer文件路径
  demo_data_dir: null
```

### 批量文件加载：
```yaml
training:
  demo_data_path: null
  demo_data_dir: "/path/to/your/buffers/"  # 批量加载目录（会加载目录下所有buffer_*.pkl和segment_*.pkl文件）
```

### 自动搜索最新文件：
```yaml
training:
  demo_data_path: null
  demo_data_dir: null  # 程序会自动搜索最新的buffer文件
```

### 启动程序：
```bash
python car_dacer_torch.py
```

程序会自动：
- 检测配置并选择合适的加载方式
- 合并多个buffer文件（如果使用批量加载）
- 直接跳过Phase1，进入Phase2离线BC训练

## 方案二：使用独立训练脚本

### 单个文件加载：
```bash
python load_buffer_and_train.py /path/to/buffer.pkl
```

### 批量文件加载：
```bash
python load_buffer_and_train.py /path/to/buffers/
```

### 自动搜索最新文件：
```bash
python load_buffer_and_train.py auto
```

### 限制最大更新次数：
```bash
python load_buffer_and_train.py /path/to/buffer.pkl 10000
```

## 加载优先级

程序按以下优先级加载数据：

1. **单个文件** (`demo_data_path`) - 如果指定了单个文件路径
2. **批量目录** (`demo_data_dir`) - 如果指定了目录路径
3. **自动搜索** - 如果前两者都未指定，自动搜索最新的buffer文件

## 支持的文件类型

### Buffer文件：
- `buffer_*.pkl` - 完整的buffer数据文件
- `segment_*.pkl` - 分段数据文件（也会被加载并合并）

### 文件格式要求：
文件应该是通过以下方式保存的：
```python
# 完整buffer格式
data = {
    'human_buffer': list(human_experiences),
    'pvp_buffer': list(pvp_experiences),
    'total_samples': total_samples,
    'total_interventions': total_interventions,
}

# 分段数据格式
data = {
    'human': list(segment_human_experiences),
    'pvp': list(segment_pvp_experiences),
    'iteration': iteration,
    'phase': phase,
}
```

## 批量加载特性

### 自动合并：
- 程序会自动合并多个文件的数据
- 按文件名排序加载，保证数据顺序
- 统计显示每个文件的加载情况

### 错误处理：
- 单个文件加载失败不会影响其他文件
- 会显示详细的加载日志和错误信息
- 最终显示加载总结

### 示例输出：
```
[批量加载] 找到 5 个文件
[批量加载] 正在加载: buffer_00001000.pkl
[批量加载] 完成: buffer_00001000.pkl (Human: 500, PVP: 500)
[批量加载] 正在加载: buffer_00002000.pkl
[批量加载] 完成: buffer_00002000.pkl (Human: 600, PVP: 600)
...
[批量加载] 总结: 加载了 5/5 个文件 (总Human: 2500, 总PVP: 2500)
```

## 常见问题

### Q: 如何指定多个特定文件？
A: 目前只能通过目录批量加载或单个文件加载。如需加载特定多个文件，可以：
1. 将文件放在同一个目录下使用批量加载
2. 修改脚本支持文件列表参数

### Q: buffer文件太大怎么办？
A: 程序会按照配置的`max_size`限制buffer大小，超出部分会被自动丢弃。

### Q: 如何验证所有数据都加载成功？
A: 程序会显示详细的加载日志，包括每个文件的数据量和最终的总数据量。

### Q: 可以混合加载不同格式的文件吗？
A: 可以，程序会自动识别buffer文件和segment文件并正确处理。

## 训练监控

程序会定期打印训练进度：
```
[进度] Iter: 10 | Phase2 Updates: 1000/100000 (1.0%) | BC Loss: 0.1234
```

TensorBoard中也会记录相应的训练指标。
