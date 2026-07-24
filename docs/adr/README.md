# ZPDS 架构决策记录（ADR）

ADR 用来冻结那些会影响多个模块、数据兼容性或审计结果的决定。状态为
`Accepted` 的 ADR 是当前实现必须遵守的规则；修改规则时新增 ADR，不直接抹掉历史。

| ADR | 主题 | 状态 |
| --- | --- | --- |
| [0001](0001-data-objects-and-boundaries.md) | 数据对象、边界与语义单位 | Accepted |
| [0002](0002-units-time-coordinate-pose.md) | 时间、单位、坐标系与 Pose | Accepted |
| [0003](0003-decision-severity-origin.md) | Decision、Severity、Evidence 与 Origin | Accepted |
| [0004](0004-versioning-naming-migration.md) | 版本、命名和兼容迁移 | Accepted |

变更流程：提出新 ADR → 给出兼容性和迁移影响 → 评审通过 → 更新 Schema、配置、
最小样例和测试 → 再修改生产实现。
