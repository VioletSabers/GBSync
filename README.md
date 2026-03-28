# EasyRendering

从 [SimpleRendering](../01.SimpleRendering) 抽取的**纯色背景文本渲染**数据生成器，仅保留通用简单渲染管线。

## 功能

- 从多个 `jsonl` 语料（`content` 字段）按类型（中文/英文）采样。
- 按语料类型配置字体池与颜色池，支持最小文字/背景对比度约束。
- 布局：`mixed_line`、`segmented`、`vertical`，以及可选的 `title_subtitle` 模板（通过 `text.template_weights` / `text.style_templates`）。
- 多进程生成 PNG 与对应 parquet 标注。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行

```bash
python scripts/generate_dataset.py --config configs/render_config.yaml
```

按字体类别（`simple` / `complex`）：

```bash
python scripts/generate_dataset.py --config configs/render_config.yaml --font-category simple
```

字体预览：

```bash
python scripts/visualize_fonts.py --config configs/render_config.yaml --font-category simple
```

## 输出

默认写入配置中 `output.root_dir`（相对 `configs/render_config.yaml` 解析），含 `images/round_xxxx/` 与 `parquet/round_xxxx.parquet`。

## 配置说明

见 `configs/render_config.yaml` 内注释与字段；请根据本机环境修改字体路径（示例使用 macOS 系统 emoji 字体路径）。
