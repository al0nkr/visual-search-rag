### Achieved 
- running tencent/HunyuanOCR with local setup and worked with test application

### TODOs 
- integrate RAG service
- find optimizations and how to enable vector dbs for searching 
- create pipeline to integrate inference server along with search window and vector db
- implement internet search 
- test out ocr capabilities 
- reduce vram usage as much as possible
- look into fine tuning and reducing weights size
- look into mlx integration with mac

### Operations 
- ```vllm serve tencent/HunyuanOCR --gpu-memory-utilization 0.6 ```
- works with images 32k input tokens ig 
- tested with 12gb VRAM RTX 4070 super