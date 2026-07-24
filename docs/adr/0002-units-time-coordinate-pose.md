# ADR-0002：时间、单位、坐标系与 Pose

- 状态：Accepted
- 冻结日期：2026-07-23
- 适用版本：ZPDS 0.1.x

## 决策

持久化标准如下：

| 项目 | 标准 |
| --- | --- |
| 时间戳 | 有符号 `int64` 纳秒（`ns`） |
| 时间区间 | 半开区间 `[start_ns, end_ns)` |
| Segment 本地原点 | `start_ns = 0` |
| 长度 | 米（`m`） |
| 角度 | 弧度（`rad`） |
| 坐标系 | 右手系 |
| 默认轴 | `x-forward, y-left, z-up` |
| 四元数顺序 | `xyzw` |
| Pose 记法 | `T_parent_child` |

`T_parent_child` 表示把 child frame 中的坐标变换到 parent frame。每个 Pose 字段必须
声明 parent/child frame，不能只写含义模糊的 `pose`。

## 原始单位和时钟

Raw 值不因标准化而被覆盖。来源单位不是标准单位时，Prepared 产物必须保存：

- 原值或可定位到原值的 source reference；
- 原始单位；
- 转换方法和生产者；
- 影响结果的配置哈希。

同理，MCAP log time、消息内时间、设备单调时钟等时钟域要分别登记。跨时钟对齐必须
有显式映射与误差，不能静默合并。

## 采样规则

Prepared 层默认保留各流原始频率。统一时间网格只存在于 `alignments/` 或 Export
视图中。nearest、linear、SLERP、ZOH 等非原样映射必须记录方法和 sample map；
严禁跨长 gap、clock reset 或无法证明的 frame mapping 插值。

## 后果

- Schema 中字段必须带单位，时间字段统一使用 `_ns` 后缀。
- 任何毫米/度数/wxyz 的输入都需要显式、可追踪的确定性转换。
- 单位或 frame 无法确认时进入 quarantine，不能靠默认猜测。
