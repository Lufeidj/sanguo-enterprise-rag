# **构建一个基于RAG的问答系统**

使用**VibeCoding**来实现 

流程：

1.整理文档，梳理文档的结构，清洗数据

2.在Milvus中设计表格

​	文件切片的表格

​	问答对的表格

​	可以参考Milvus官方示例设计表格的结构和生成代码

3.把设计表格的代码存放到一个vdb_init.py文件中

4.生成数据

​	把数据insert到表格中

5.封装一个RAG的检索系统，根据用户的问题进行向量库的检索，检索出匹配度较高的内容，放在提示词中，交给LLM执行

6.让AI帮助进行测试，确保准确度

7.外层提供一个FastAPI访问的接口

8.制作一个前端页面





## 手撕RAG的具体的步骤

### 第一步：整理文档

​	使用<<三国演义>>，让AI帮我生成场景的问答对

### 第二步：设计表格

​	选择：可以在Milvus官方AI对话中创建表格

​			帮我创建Milvus的向量表，表创建在AI80数据库中，用于保存文件切片的数据，尽可能保存元数据的信息，同时也要创建一个稀疏向量的字段，最终实现混合检索，帮我实现BM25算法

​		参照vdb_init_milvus.py文件中创建表的方法，帮忙再创建一个基于问答对的milvus向量表，包含三个字段：问题，答案，推理过程（为什么得出答案的思考过程），放在vdb_init_milvus.py文件中，保持较低的耦合度，每个建表语句使用一个函数

   【可以使用codebuddy，claude code】

### 第三步：生成数据

​	**文件切片的表**

 1. 帮我在rag_demo目录下创建一个until的目录，在until目录下创建解析txt文件内容的脚本，入参是一个本地txt文件的地址，对txt本地文件进行解析，出参解析后的文本，类型是字符串str

 2. 切片操作

    在until目录下生成一个专门用于对文本进行切片的工具类，使用LangChain的封装的切片函数进行切片，对输入的文本进行切片，入参是文本内容，str类型，需要提供元数据信息，出参是list[dict]

​	**问答对的表**

​		让大模型生成，存放在datas目录下

3. 插入到表中

   在vdb_init_milvus.py文件中创建两个函数，分别在两张表中插入数据，问答对的数据从datas/qa_paris.json中读取，文件切片的数据从datas/三国演义.txt中读取

   帮我根据表结构生成对应的插入逻辑，向量数据的生成请参照：

   ```python
   import os
   from openai import OpenAI
   
   client = OpenAI(
       api_key=os.getenv("DASHSCOPE_API_KEY"),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
       base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
   )
   
   completion = client.embeddings.create(
       model="text-embedding-v4",
       input='衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买',
       dimensions=1024, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
       encoding_format="float"
   )
   
   print(completion.model_dump_json())
   ```

   key从.env文件中读取



### 第四步：RAG问答机器人

基于vdb_init_milvus.py中的表结构，创建一个新的文件夹core，在下面创建一个RAG问答的文件，参照官方文档的混合检索的案例，生成一个函数，对文件切片的表和问答对的表检索出的内容置入提示词的上下文，调用qwen的大模型进行回答

入参是用户的问题，出参不但有大模型回答，还需要加上引用的内容

混合检索的案例：

from pymilvus import AnnSearchRequest

query_text = "white headphones, quiet and comfortable"
query_dense_vector = generate_dense_vector(768)
query_multimodal_vector = generate_dense_vector(512)

search_param_1 = {
    "data": [query_dense_vector],
    "anns_field": "text_dense",
    "param": {"nprobe": 10},
    "limit": 2
}
request_1 = AnnSearchRequest(**search_param_1)

search_param_2 = {
    "data": [query_text],
    "anns_field": "text_sparse",
    "limit": 2
}
request_2 = AnnSearchRequest(**search_param_2)

search_param_3 = {
    "data": [query_multimodal_vector],
    "anns_field": "image_dense",
    "param": {"nprobe": 10},
    "limit": 2
}
request_3 = AnnSearchRequest(**search_param_3)

reqs = [request_1, request_2, request_3]



ranker = Function(
    name="rrf",
    input_field_names=[], # Must be an empty list
    function_type=FunctionType.RERANK,
    params={
        "reranker": "rrf", 
        "k": 100  # Optional
    }
)

res = client.hybrid_search(
    collection_name="my_collection",
    reqs=reqs,
    ranker=ranker,
    limit=2
)
for hits in res:
    print("TopK results:")
    for hit in hits:
        print(hit)

### 第五步：把RAG问答的函数封装成一个FastAPI接口，对外提供访问服务

### 第六步：生成一个前端，放到学生管理系统里，增加一个菜单

