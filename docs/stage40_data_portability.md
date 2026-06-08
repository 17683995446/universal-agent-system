# 阶段40: Agent数据导出与导入系统

> 遵循"图难于其易，为大于其细"的原则，实现多格式导出、智能导入、跨存储迁移。

## 核心组件

| 组件 | 功能 |
|------|------|
| DataExporter | JSON/Pickle/CSV/归档导出 |
| DataImporter | 多格式导入，冲突处理 |
| DataMigrator | 跨存储迁移，增量同步 |

## 导出格式

- **JSON**: 人类可读，通用性强
- **Pickle**: Python原生，支持复杂对象
- **CSV**: 表格格式，适合简单键值对
- **Archive**: 目录归档，每个条目独立文件

## 冲突策略

- **SKIP**: 跳过已存在的键
- **OVERWRITE**: 覆盖已存在的键
- **RENAME**: 自动重命名冲突键

## 测试覆盖: 34个测试全部通过

运行: `PYTHONPATH=/workspace python3 -m pytest tests/test_data_portability.py -v`
