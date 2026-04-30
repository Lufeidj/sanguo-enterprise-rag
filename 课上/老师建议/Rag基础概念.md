# RAG & Embeddings & Vector Store

## 一、什么是检索增强的生成模型（RAG）

### 1.1 大模型目前固有的局限性

1. LLM的知识不是实时的
2. LLM 可能不知道你私有的领域/业务知识

### 1.2 检索增强生成

RAG（Retrieval Augmented Generation）顾名思义，通过**检索**的方法来增强**生成模型**的能力。

<video src="./assets/RAG.mp4" controls="controls" width=1024px style="margin-left: 0px"></video>

<div class="alert alert-success">
<b>类比：</b>你可以把这个过程想象成开卷考试。让 LLM 先翻书，再回答问题。
</div>

## 二、RAG系统的基本搭建流程

<img src="./assets/rag.png" style="margin-left: 0px" width=1024px>

搭建过程：

1. 文档加载，并按一定条件**切割**成片段
2. 将切割的文本片段灌入**检索引擎**
3. 封装**检索接口**
4. 构建**调用流程**：Query -> 检索 -> Prompt -> LLM -> 回复

## 三、向量检索

### 3.1 什么是向量

​    向量是一种有大小和方向的数学对象。它可以表示为从一个点到另一个点的有向线段。例如，二维空间中的向量可以表示为(x, y) ，表示从原点(0, 0)到点(x, y)的有向线段
<img src="./assets/vector.png" style="margin-left: 0px" width=800px>
​        

以此类推，我可以用一组坐标    表示一个 N维空间中的向量，N叫向量的维度

#### 3.1.1 文本向量（Text Embeddings）

1. 将文本转成一组 N维浮点数，即**文本向量**又叫 Embeddings
2. 向量之间可以计算距离，距离远近对应**语义相似度**大小

<br />
<img src="./assets/embeddings.png" style="margin-left: 0px" width=800px>
<br />

### 3.1.2 文本向量是怎么得到的（选）

1. 构建相关（正例）与不相关（负例）的句子对样本
2. 训练双塔式模型，让正例间的距离小，负例间的距离大

例如：

<img src="./assets/sbert.png" style="margin-left: 0px" width=500px>

<div class="alert alert-info">
<b>扩展阅读：https://www.sbert.net</b>
</div>

### 3.2 向量间的相似度计算

<img src="./assets/sim.png" style="margin-left: 0px" width=500px>


### 3.3 向量数据库

#### 一、原生专用向量库（AI RAG 主力）

1. **Milvus**：国产、分布式、大数据量、企业生产标配
2. **Qdrant**：轻量高性能、低延迟、中小项目常用
3. **Chroma**：轻量化、本地开发、调试原型快速搭建
4. **Weaviate**：向量 + 全文混合检索，多模态友好
5. **FAISS**：Meta 开源，离线向量检索、算法强，需二次开发

#### 二、传统数据库改造（低运维、存量项目首选）

1. **pgvector**（PostgreSQL）：向量 + 关系一体，中小 RAG 最简方案
2. **Elasticsearch**：全文 + 向量混合，文档检索、传统知识库改造
3. **Redis Vector**：内存级、超低延迟，高频小向量业务
