LayoutLMv3已经集成到了Huggingface Transformers里面。

但是这里我们使用微软的开源代码，原因如下：
- Transformers的实现和原始代码有差别，比如pad逻辑和输入字段、Tokenizer实现等
- 直接引入原始代码，方便后续改造