---
title: "Chapter 24. 自定义模型开发"
description: "从零构建自定义 Transformer 模型、继承 PreTrainedModel 基类、实现完整 BERT"
updated: "2026-01-22"
---

在前面的章节中，我们学习了如何使用 Hugging Face Transformers 库中的预训练模型。本章将深入探讨如何**从零构建自定义模型**，包括继承 `PreTrainedModel` 基类、实现完整的 BERT 模型、注册新架构、自定义注意力机制以及训练新的 Tokenizer。这些技能对于研究新架构、适配特殊任务或理解模型内部机制至关重要。

---

## 24.1 PreTrainedModel 基类

所有 Transformers 模型都继承自 `PreTrainedModel` 基类，它提供了：
- **权重加载与保存**：`from_pretrained()`, `save_pretrained()`
- **设备管理**：`.to(device)`, `.cuda()`, `.cpu()`
- **前向传播接口**：统一的输入输出格式
- **配置管理**：`PretrainedConfig` 对象

### 24.1.1 必须实现的方法

创建自定义模型时，必须实现以下方法：

```python
from transformers import PreTrainedModel, PretrainedConfig
import torch.nn as nn

class CustomConfig(PretrainedConfig):
    """
    自定义配置类，继承自 PretrainedConfig
    """
    model_type = "custom"  # 模型类型标识符
    
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


class CustomModel(PreTrainedModel):
    """
    自定义模型类，继承自 PreTrainedModel
    """
    config_class = CustomConfig  # 关联配置类
    base_model_prefix = "custom"  # 权重前缀
    
    def __init__(self, config):
        super().__init__(config)
        # 模型层定义
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder = nn.TransformerEncoder(...)
        # 初始化权重
        self.post_init()
    
    def _init_weights(self, module):
        """
        初始化权重（必须实现）
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        前向传播（必须实现）
        """
        # 实现具体逻辑
        embeddings = self.embeddings(input_ids)
        outputs = self.encoder(embeddings, mask=attention_mask)
        return outputs
```

**关键点**：
1. **`config_class`**：关联配置类，用于 `from_pretrained()` 自动加载
2. **`base_model_prefix`**：权重保存时的前缀，如 `custom.encoder.layer.0.weight`
3. **`_init_weights()`**：PyTorch 模块初始化策略（正态分布、Xavier 等）
4. **`post_init()`**：调用父类方法，自动应用 `_init_weights`

### 24.1.2 配置类（PretrainedConfig）

配置类必须包含：
- **`model_type`**：唯一标识符（用于 AutoModel 注册）
- **所有超参数**：层数、隐藏维度、dropout 等
- **`__init__()`**：调用 `super().__init__(**kwargs)` 以支持额外参数

**最佳实践**：
```python
class MyConfig(PretrainedConfig):
    model_type = "my_model"
    
    def __init__(self, hidden_size=768, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        # 自动支持 pad_token_id, bos_token_id 等通用参数
```

### 24.1.3 权重初始化（_init_weights）

不同层的初始化策略：

| 层类型 | 初始化方法 | 说明 |
|--------|------------|------|
| `nn.Linear` | `normal_(0, std)` | 标准差通常为 0.02 |
| `nn.Embedding` | `normal_(0, std)` | padding_idx 位置清零 |
| `nn.LayerNorm` | `weight=1.0, bias=0` | 标准做法 |
| `nn.Conv1d` | `kaiming_normal_()` | 适用于 CNN |

**示例：BERT 风格初始化**
```python
def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
```

---

## 24.2 从零实现一个 BERT

下面完整实现一个简化版 BERT 模型，包含 Embedding、Encoder、Pooler 和分类头。

### 24.2.1 Embedding Layer

BERT 的嵌入层由三部分组成：

```python
import torch
import torch.nn as nn
import math

class BERTEmbeddings(nn.Module):
    """
    BERT Embeddings = Token Embeddings + Position Embeddings + Token Type Embeddings
    """
    def __init__(self, config):
        super().__init__()
        # 1. Token Embeddings（词嵌入）
        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            padding_idx=config.pad_token_id
        )
        
        # 2. Position Embeddings（位置编码，学习式）
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, 
            config.hidden_size
        )
        
        # 3. Token Type Embeddings（句子类型，用于句子对任务）
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size,  # 通常为 2 (sentence A/B)
            config.hidden_size
        )
        
        # Layer Normalization 和 Dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Position IDs（固定不变，注册为 buffer）
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False
        )
    
    def forward(
        self, 
        input_ids, 
        token_type_ids=None, 
        position_ids=None
    ):
        seq_length = input_ids.size(1)
        
        # 1. 获取 position_ids（如果未提供）
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        # 2. 获取 token_type_ids（默认全 0）
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # 3. 计算三种嵌入
        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        # 4. 求和并归一化
        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
```

**关键设计**：
- **`position_ids`** 注册为 buffer，避免占用参数计数
- **三种嵌入求和**：与 Transformer 的 sinusoidal 不同，BERT 使用**可学习位置编码**
- **LayerNorm + Dropout**：在嵌入层就进行归一化，提升稳定性

### 24.2.2 Transformer Encoder Layer

实现单个 Transformer Encoder 层：

```python
class BERTSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention
    """
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) 必须能被 "
                f"num_attention_heads ({config.num_attention_heads}) 整除"
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Q, K, V 线性变换
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x):
        """
        将张量变形为 (batch, num_heads, seq_len, head_size)
        """
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        # 1. 线性变换得到 Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 2. 计算注意力分数：Q @ K^T / sqrt(d_k)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 3. 应用 attention_mask（padding mask）
        if attention_mask is not None:
            # attention_mask: (batch, 1, 1, seq_len)，扩展到多头
            attention_scores = attention_scores + attention_mask
        
        # 4. Softmax 归一化
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 5. 加权求和：Attention @ V
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # 6. 转置并合并多头
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_shape)
        
        return context_layer


class BERTSelfOutput(nn.Module):
    """
    Self-Attention 输出层（残差连接 + LayerNorm）
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # 残差连接
        return hidden_states


class BERTAttention(nn.Module):
    """
    完整 Attention 模块（Self-Attention + Output）
    """
    def __init__(self, config):
        super().__init__()
        self.self = BERTSelfAttention(config)
        self.output = BERTSelfOutput(config)
    
    def forward(self, hidden_states, attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


class BERTIntermediate(nn.Module):
    """
    前馈网络（FFN）的中间层
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()  # BERT 使用 GELU 激活
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BERTOutput(nn.Module):
    """
    前馈网络输出层（残差连接 + LayerNorm）
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTLayer(nn.Module):
    """
    单个 Transformer Encoder 层
    """
    def __init__(self, config):
        super().__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)
    
    def forward(self, hidden_states, attention_mask=None):
        # 1. Self-Attention
        attention_output = self.attention(hidden_states, attention_mask)
        
        # 2. Feed-Forward Network
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        
        return layer_output
```

**架构解析**：
- **Multi-Head Attention**：
  1. 线性变换 → Q, K, V
  2. 分割为多头：`(batch, seq, hidden)` → `(batch, heads, seq, head_dim)`
  3. 计算注意力分数：`QK^T / sqrt(d_k)`
  4. Softmax + Dropout
  5. 加权求和：`Attention @ V`
  6. 合并多头 → 线性变换
- **残差连接（Residual Connection）**：每个子层都添加 `LayerNorm(X + Sublayer(X))`
- **前馈网络（FFN）**：`GELU(W1 @ X) @ W2`，通常 `intermediate_size = 4 × hidden_size`

### 24.2.3 Pooler 与 Classification Head

```python
class BERTEncoder(nn.Module):
    """
    多层 Transformer Encoder
    """
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BERTLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, hidden_states, attention_mask=None):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states


class BERTPooler(nn.Module):
    """
    将 [CLS] token 的表示转换为句子表示
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states):
        # 取第一个 token（[CLS]）的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BERTModel(PreTrainedModel):
    """
    完整的 BERT 模型（无任务头）
    """
    config_class = BERTConfig
    base_model_prefix = "bert"
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)
        self.pooler = BERTPooler(config)
        
        self.post_init()
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        return_dict=True
    ):
        # 1. 处理 attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # 转换为 (batch, 1, 1, seq_len) 格式，用于广播
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
        
        # 2. Embedding 层
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )
        
        # 3. Encoder 层
        encoder_output = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask
        )
        
        # 4. Pooler 层
        pooled_output = self.pooler(encoder_output)
        
        if not return_dict:
            return (encoder_output, pooled_output)
        
        from transformers.modeling_outputs import BaseModelOutputWithPooling
        return BaseModelOutputWithPooling(
            last_hidden_state=encoder_output,
            pooler_output=pooled_output
        )


class BERTForSequenceClassification(PreTrainedModel):
    """
    带分类头的 BERT 模型
    """
    config_class = BERTConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        self.bert = BERTModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        self.post_init()
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        return_dict=True
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )
```

### 24.2.4 完整代码实现与测试

```python
# 配置类
class BERTConfig(PretrainedConfig):
    model_type = "custom_bert"
    
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        num_labels=2,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.num_labels = num_labels


# 测试模型
if __name__ == "__main__":
    # 创建配置（小模型用于测试）
    config = BERTConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        max_position_embeddings=128
    )
    
    # 实例化模型
    model = BERTForSequenceClassification(config)
    
    # 创建假数据
    batch_size = 4
    seq_length = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    labels = torch.randint(0, 2, (batch_size,))
    
    # 前向传播
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    print(f"Loss: {outputs.loss.item():.4f}")
    print(f"Logits shape: {outputs.logits.shape}")  # (4, 2)
    
    # 保存模型
    model.save_pretrained("./my_custom_bert")
    
    # 重新加载
    loaded_model = BERTForSequenceClassification.from_pretrained("./my_custom_bert")
    print("✅ 模型保存和加载成功！")
```

**输出示例**：
```
Loss: 0.6931
Logits shape: torch.Size([4, 2])
✅ 模型保存和加载成功！
```

---

## 24.3 添加新的模型架构

### 24.3.1 注册模型（AutoModel）

要让自定义模型支持 `AutoModel.from_pretrained()`，需要注册：

```python
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification

# 1. 注册配置
AutoConfig.register("custom_bert", BERTConfig)

# 2. 注册模型
AutoModel.register(BERTConfig, BERTModel)
AutoModelForSequenceClassification.register(BERTConfig, BERTForSequenceClassification)

# 现在可以使用 Auto 类加载
model = AutoModelForSequenceClassification.from_pretrained("./my_custom_bert")
```

### 24.3.2 配置 mapping

创建 `config.json` 文件：

```json
{
  "architectures": [
    "BERTForSequenceClassification"
  ],
  "model_type": "custom_bert",
  "vocab_size": 1000,
  "hidden_size": 128,
  "num_hidden_layers": 2,
  "num_attention_heads": 2,
  "intermediate_size": 512,
  "hidden_dropout_prob": 0.1,
  "attention_probs_dropout_prob": 0.1,
  "max_position_embeddings": 128,
  "type_vocab_size": 2,
  "initializer_range": 0.02,
  "layer_norm_eps": 1e-12,
  "pad_token_id": 0,
  "num_labels": 2,
  "transformers_version": "4.36.0"
}
```

### 24.3.3 上传到 Hub

```python
from huggingface_hub import login

# 登录
login(token="hf_xxx")

# 上传模型
model.push_to_hub("username/custom-bert")
config.push_to_hub("username/custom-bert")

# 其他人可以直接使用
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("username/custom-bert")
```

---

## 24.4 自定义 Attention

### 24.4.1 实现 Sparse Attention

稀疏注意力通过限制注意力范围来降低计算复杂度。

**Local Window Attention（局部窗口注意力）**：

```python
class LocalWindowAttention(nn.Module):
    """
    每个 token 只关注前后 window_size 范围内的 token
    """
    def __init__(self, config, window_size=128):
        super().__init__()
        self.window_size = window_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def create_local_mask(self, seq_length, device):
        """
        创建局部窗口 mask
        mask[i, j] = 1 if |i - j| <= window_size else 0
        """
        # 创建位置索引
        positions = torch.arange(seq_length, device=device)
        positions_expanded = positions.unsqueeze(1)  # (seq_len, 1)
        positions_tiled = positions.unsqueeze(0)     # (1, seq_len)
        
        # 计算距离
        distance = torch.abs(positions_expanded - positions_tiled)
        
        # 创建 mask：距离 <= window_size 的位置为 1
        mask = (distance <= self.window_size).float()
        
        # 转换为注意力 mask（0 → -inf，1 → 0）
        attention_mask = (1.0 - mask) * torch.finfo(torch.float32).min
        
        return attention_mask
    
    def forward(self, hidden_states):
        batch_size, seq_length, _ = hidden_states.size()
        
        # Q, K, V 变换
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        # 分头
        def transpose_for_scores(x):
            new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(new_shape)
            return x.permute(0, 2, 1, 3)
        
        query_layer = transpose_for_scores(query_layer)
        key_layer = transpose_for_scores(key_layer)
        value_layer = transpose_for_scores(value_layer)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 应用局部窗口 mask
        local_mask = self.create_local_mask(seq_length, hidden_states.device)
        attention_scores = attention_scores + local_mask.unsqueeze(0).unsqueeze(0)
        
        # Softmax
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 加权求和
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.num_attention_heads * self.attention_head_size,)
        context_layer = context_layer.view(new_shape)
        
        return context_layer


# 测试
config = BERTConfig()
attention = LocalWindowAttention(config, window_size=64)
hidden = torch.randn(2, 128, 768)
output = attention(hidden)
print(f"Output shape: {output.shape}")  # (2, 128, 768)
```

### 24.4.2 Strided Attention（跳跃注意力）

```python
class StridedAttention(nn.Module):
    """
    每 stride 个 token 关注一次（用于 Longformer、BigBird）
    """
    def __init__(self, config, stride=64):
        super().__init__()
        self.stride = stride
        # ... 其他初始化同上
    
    def create_strided_mask(self, seq_length, device):
        """
        mask[i, j] = 1 if (j % stride == 0) or (j == i)
        """
        positions = torch.arange(seq_length, device=device)
        strided_indices = positions % self.stride == 0
        
        # 每个位置可以关注：(1) 自己 (2) stride 的倍数位置
        mask = torch.zeros(seq_length, seq_length, device=device)
        mask[:, strided_indices] = 1.0  # 所有位置都可以关注 stride 倍数
        mask[torch.arange(seq_length), torch.arange(seq_length)] = 1.0  # 关注自己
        
        attention_mask = (1.0 - mask) * torch.finfo(torch.float32).min
        return attention_mask
```

### 24.4.3 与标准 Attention 对比

<div data-component="CustomAttentionComparator"></div>

| 类型 | 复杂度 | 优点 | 缺点 | 适用场景 |
|------|--------|------|------|----------|
| **Full Attention** | $O(n^2)$ | 全局感受野，效果最佳 | 长序列显存爆炸 | 短序列（≤512） |
| **Local Window** | $O(n \times w)$ | 计算高效，适合局部模式 | 缺乏全局信息 | 文档分类、代码 |
| **Strided** | $O(n \times \frac{n}{s})$ | 平衡局部与全局 | 实现复杂 | 长文档（Longformer） |
| **Sparse (Random)** | $O(n \times k)$ | 极致效率 | 不稳定，需大量实验 | 研究探索 |

---

## 24.5 自定义 Tokenizer

### 24.5.1 Tokenizer 基类

Hugging Face 支持三种主要 Tokenizer：
- **WordPiece**（BERT）：基于词频的子词切分
- **BPE**（GPT-2）：字节对编码
- **Unigram**（T5）：概率模型

### 24.5.2 训练新 tokenizer

使用 `tokenizers` 库从头训练：

```python
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

# 1. 创建 BPE 模型
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# 2. 设置预处理（分词策略）
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 3. 定义训练器
trainer = trainers.BpeTrainer(
    vocab_size=10000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# 4. 训练（从文件或迭代器）
files = ["corpus.txt"]
tokenizer.train(files, trainer)

# 5. 保存
tokenizer.save("my_tokenizer.json")

# 6. 包装为 PreTrainedTokenizer
from transformers import PreTrainedTokenizerFast

fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="my_tokenizer.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]"
)
```

**从迭代器训练**：
```python
def get_training_corpus(dataset):
    """
    生成器，逐批返回文本
    """
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

# 从 HuggingFace 数据集训练
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

tokenizer.train_from_iterator(
    get_training_corpus(dataset),
    trainer=trainer,
    length=len(dataset)
)
```

### 24.5.3 添加特殊 token

```python
# 添加新的特殊 token
special_tokens = {"additional_special_tokens": ["[USER]", "[AGENT]"]}
tokenizer.add_special_tokens(special_tokens)

# 扩展模型 Embedding
model.resize_token_embeddings(len(tokenizer))

# 测试
text = "[USER] Hello! [AGENT] Hi there!"
tokens = tokenizer.tokenize(text)
print(tokens)  # ['[USER]', 'Hello', '!', '[AGENT]', 'Hi', 'there', '!']
```

### 24.5.4 保存与加载

```python
# 保存到目录
tokenizer.save_pretrained("./my_tokenizer")

# 生成文件：
# - tokenizer_config.json
# - tokenizer.json
# - special_tokens_map.json

# 加载
from transformers import AutoTokenizer
loaded_tokenizer = AutoTokenizer.from_pretrained("./my_tokenizer")
```

**上传到 Hub**：
```python
tokenizer.push_to_hub("username/my-tokenizer")

# 其他人使用
tokenizer = AutoTokenizer.from_pretrained("username/my-tokenizer")
```

---

## 24.6 模型构建可视化工具

<div data-component="ModelBuilderTool"></div>

该工具允许你通过拖拽方式组合模型组件：
- **Embedding Layer**：词嵌入 + 位置编码
- **Attention Layer**：标准/Sparse/Strided
- **Feed-Forward Network**：GELU/ReLU 激活
- **Pooler**：[CLS] pooling / Mean pooling
- **Classification Head**：线性层 + Softmax

**生成代码**：自动导出完整的 PyTorch 模型定义。

---

## 24.7 实战案例：自定义 Sentiment Analyzer

综合运用上述知识，构建一个情感分析模型：

```python
# 1. 定义配置
class SentimentConfig(PretrainedConfig):
    model_type = "sentiment_analyzer"
    
    def __init__(
        self,
        vocab_size=10000,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        num_classes=3,  # 负面、中性、正面
        dropout=0.3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout


# 2. 定义模型（使用 LSTM）
class SentimentAnalyzer(PreTrainedModel):
    config_class = SentimentConfig
    
    def __init__(self, config):
        super().__init__(config)
        
        # Embedding
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        # Bi-LSTM
        self.lstm = nn.LSTM(
            config.embedding_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        
        # 分类头
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_dim * 2, config.num_classes)  # *2 for bidirectional
        
        self.post_init()
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Embedding
        embedded = self.embedding(input_ids)  # (batch, seq, emb_dim)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded)  # lstm_out: (batch, seq, hidden*2)
        
        # 取最后一个时间步
        final_hidden = lstm_out[:, -1, :]  # (batch, hidden*2)
        
        # 分类
        final_hidden = self.dropout(final_hidden)
        logits = self.fc(final_hidden)  # (batch, num_classes)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )


# 3. 训练
from transformers import Trainer, TrainingArguments

config = SentimentConfig(vocab_size=5000, num_classes=3)
model = SentimentAnalyzer(config)

training_args = TrainingArguments(
    output_dir="./sentiment_model",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=5e-4,
    save_steps=500,
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # 需准备数据集
    eval_dataset=eval_dataset
)

trainer.train()

# 4. 保存和推理
model.save_pretrained("./sentiment_model")
loaded_model = SentimentAnalyzer.from_pretrained("./sentiment_model")
```

---

## 24.8 常见问题与最佳实践

### Q1: 自定义模型无法使用 `from_pretrained()`？

**原因**：未正确设置 `config_class` 或 `base_model_prefix`。

**解决**：
```python
class MyModel(PreTrainedModel):
    config_class = MyConfig  # 必须设置
    base_model_prefix = "my_model"
```

### Q2: 权重初始化后效果很差？

**原因**：初始化策略不当。

**建议**：
- **线性层**：`normal_(0, 0.02)` 或 `xavier_uniform_()`
- **Embedding**：`normal_(0, 0.02)`
- **LayerNorm**：`weight=1.0, bias=0`
- **避免**：过大的初始化标准差（如 1.0）

### Q3: 如何调试自定义模型？

```python
# 1. 检查参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# 2. 检查梯度流
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item():.4f}")

# 3. 使用 torchinfo
from torchinfo import summary
summary(model, input_size=(32, 128), dtypes=[torch.long])
```

### Q4: 自定义 Tokenizer 如何处理未登录词？

```python
# 设置 unk_token
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

# 测试
tokens = tokenizer.encode("未见过的词汇")
print(tokens.tokens)  # 包含 <unk> token
```

---

## 24.9 性能优化技巧

### 1. **Flash Attention 集成**

```python
from flash_attn import flash_attn_qkvpacked_func

class FlashSelfAttention(nn.Module):
    def forward(self, hidden_states):
        # 计算 Q, K, V
        qkv = self.qkv_proj(hidden_states)  # (batch, seq, 3 * hidden)
        qkv = qkv.reshape(batch, seq, 3, num_heads, head_dim)
        
        # 调用 Flash Attention
        output = flash_attn_qkvpacked_func(
            qkv, 
            dropout_p=self.dropout_prob, 
            softmax_scale=1.0 / math.sqrt(head_dim)
        )
        
        return output.reshape(batch, seq, hidden)
```

### 2. **Gradient Checkpointing**

```python
class MyModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.gradient_checkpointing = False
    
    def forward(self, hidden_states):
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states
                )
            else:
                hidden_states = layer(hidden_states)
        return hidden_states

# 启用
model.gradient_checkpointing_enable()
```

### 3. **混合精度训练**

```python
# 使用 torch.amp 自动混合精度
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 24.10 章节总结

本章我们深入学习了自定义模型开发的全流程：

✅ **核心技能**：
- 继承 `PreTrainedModel` 和 `PretrainedConfig`
- 从零实现 BERT 架构（Embedding → Encoder → Pooler）
- 注册自定义模型到 AutoModel
- 实现 Sparse Attention（Local Window、Strided）
- 训练和保存自定义 Tokenizer

✅ **实战能力**：
- 构建完整的情感分析模型
- 集成 Flash Attention 和 Gradient Checkpointing
- 调试和优化自定义模型

✅ **最佳实践**：
- 权重初始化策略：`normal_(0, 0.02)`
- 残差连接：`LayerNorm(X + Sublayer(X))`
- 配置管理：所有超参数放入 Config 类

**下一章预告**：Chapter 25 将学习**自定义 Trainer 与训练循环**，包括重写 `compute_loss()`、实现自定义 Callback、完全自定义训练循环以及高级损失函数（Focal Loss、Contrastive Loss 等）。
