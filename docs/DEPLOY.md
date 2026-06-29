# 部署与演示指南

本文档覆盖三种场景：本地 Docker 启动、Cloudflare Tunnel 临时公网演示、长期生产方案对比。

---

## 1. 项目部署形态说明

本项目**不是**传统的 SPA（Single Page Application）架构：

- **不是** React / Vite / Next.js / Nuxt，**不需要** `npm run build`
- **不依赖** Nginx / Caddy 托管静态文件
- **不使用** 浏览器侧 JavaScript 直接请求后端

实际架构是：

| 角色 | 框架 | 端口 | 暴露范围 |
| --- | --- | --- | --- |
| **对外前端** | Streamlit（Python） | `8501` | 本机 / Cloudflare Tunnel |
| **后端 API** | FastAPI + uvicorn（Python） | `8000` | **仅本机回环** |

关键设计：

- Streamlit 是**服务端渲染**框架，所有用户交互通过 8501 端口的 WebSocket 通信
- Streamlit 服务端用 Python `requests.post(...)` 调用 FastAPI，**浏览器不直接请求 8000**
- Docker Compose 负责把前后端作为一个整体应用启动，通过内部 docker 网络互联

---

## 2. 本地 Docker Compose 启动

### 2.1 配置密钥

```bash
cp .env.example .env
```

编辑 `.env`，**必填**项：

```dotenv
SILICONFLOW_API_KEY=你的_SiliconFlow_API_Key
LLM_API_KEY=你的_LLM_API_Key
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_MODEL=glm-4-flash
```

其他项保留默认值即可。

### 2.2 一键启动

```bash
docker compose up -d --build
docker compose ps
```

等待 `backend` 和 `frontend` 两个容器的 Status 都变为 `Up (healthy)`，通常需要 20–40 秒（首次构建镜像时间会更长）。

### 2.3 访问入口

| 用途 | 地址 | 备注 |
| --- | --- | --- |
| **Streamlit 前端**（主入口） | <http://127.0.0.1:8501> | 推荐访问方式 |
| 后端 Swagger UI（本机调试） | <http://127.0.0.1:8000/docs> | 仅本机可访问 |
| 后端健康检查 | <http://127.0.0.1:8000/health> | 容器 healthcheck 使用 |

### 2.4 查看日志

```bash
docker compose logs frontend          # 仅前端
docker compose logs backend           # 仅后端
docker compose logs -f                # 实时跟踪全部
```

---

## 3. Cloudflare Tunnel 临时公网演示

### 3.1 适用场景

- 临时演示给不在同一局域网的同事 / 朋友 / 老师
- 不需要域名、不需要公网 IP、不需要修改防火墙
- 演示结束后关闭即可

### 3.2 启动 Tunnel

前提：本地 `docker compose up -d` 已经启动，`http://127.0.0.1:8501` 本机能打开。

```bash
cloudflared tunnel --url http://127.0.0.1:8501
```

终端会输出形如：

```
+--------------------------------------------------------------------------------------------+
|  Your quick Tunnel has been created! View it at (it may take some time to be reachable):  |
|  https://random-words-1234.trycloudflare.com                                              |
+--------------------------------------------------------------------------------------------+
```

把 `https://random-words-1234.trycloudflare.com` 这个地址（以及演示访问码，如果设置了）发给你信任的用户。

### 3.3 访问链路图

```
外部用户浏览器（不同局域网）
  ↓ HTTPS
https://xxxx.trycloudflare.com
  ↓ Cloudflare 边缘网络
cloudflared（运行在你本机）
  ↓
宿主机 127.0.0.1:8501
  ↓ Docker 端口映射
Docker frontend 容器（Streamlit，监听 0.0.0.0:8501）
  ↓ Python requests.post("http://backend:8000/api/...")
Docker 内部 bridge 网络 "rs-rag-net"
  ↓ DNS 解析服务名 "backend"
Docker backend 容器（FastAPI，监听 0.0.0.0:8000）
  ↓
Chroma 向量库 + 外部 LLM/Embedding API
```

### 3.4 关键事实

- **浏览器只与 Cloudflare 边缘通信**，不直接访问你的本机
- **Streamlit 容器**接收浏览器请求，所有 `requests.post` 都在容器内执行
- **后端 8000 端口**只绑定 `127.0.0.1`，**不会被 tunnel 暴露**，也不会被局域网其他人访问

---

## 4. 为什么只暴露 8501，不暴露 8000

### 4.1 外部用户只需要 Streamlit

Streamlit 在服务端调用 FastAPI（通过 docker 内部网络），浏览器不会直接请求 `localhost:8000`。

### 4.2 后端 8000 端口包含敏感操作

| 端点 | 风险 |
| --- | --- |
| `POST /api/documents/upload` | 任意文件上传（受扩展名白名单限制） |
| `POST /api/documents/ingest` | 触发 embedding API 调用（消耗配额） |
| `DELETE /api/documents/{doc_id}` | **删除知识库**（不可恢复） |
| `POST /api/agent/query` | 消耗 LLM API 配额 |

### 4.3 当前端口绑定策略

`docker-compose.yml` 中：

```yaml
backend:
  ports:
    - "127.0.0.1:8000:8000"   # 仅本机回环，局域网/公网无法访问

frontend:
  ports:
    - "8501:8501"             # 所有接口可访问（用于 Cloudflare Tunnel 接入）
```

注意：**容器内部 backend 仍监听 `0.0.0.0:8000`**，frontend 容器通过 docker 内部 DNS 用 `http://backend:8000` 访问不受影响。端口绑定收紧只发生在宿主机层面。

---

## 5. RAG_API_BASE 在不同场景的取值

| 场景 | `RAG_API_BASE` 取值 | 说明 |
| --- | --- | --- |
| **本地裸跑**（`streamlit run` + `uvicorn`） | `http://127.0.0.1:8000` | 默认值 |
| **Docker Compose 启动** | `http://backend:8000` | 由 `docker-compose.yml` 的 `environment` 显式设置 |
| **Cloudflare Tunnel 公网演示** | `http://backend:8000` | 同上，tunnel 暴露前端而非后端 |

### 重要约束

- ❌ **不要**把 `RAG_API_BASE` 改成 `https://xxxx.trycloudflare.com`
  - trycloudflare 地址是给**外部浏览器**用的，Streamlit 服务端不能用这个地址访问后端
- ❌ **不要**把 `RAG_API_BASE` 改成 `/api` 相对路径
  - Streamlit 是服务端 Python 代码，相对路径在服务端无意义；后端在不同端口（8000），不在 Streamlit（8501）的同源路径下
- ✅ **保持** `http://backend:8000`（docker 场景）或 `http://127.0.0.1:8000`（裸跑场景）

---

## 6. DEMO_PASSWORD 访问码

### 6.1 何时使用

- 通过 Cloudflare Tunnel 把链接发给别人时，**强烈建议**设置
- 本地开发或局域网测试时，**不设置**（保持原体验）

### 6.2 如何设置

编辑 `.env`：

```dotenv
DEMO_PASSWORD=your-secret-password
```

### 6.3 设置后的效果

- 访问者打开链接 → 先看到"🔒 访问验证"页面
- 必须输入正确访问码才能进入 Streamlit 主界面
- 验证状态保存在 `st.session_state`，单次会话内不会重复要求输入
- 输入错误会显示"访问码错误"，无法进入主界面

### 6.4 不设置的效果

- 访问者打开链接 → 直接看到 Streamlit 主界面
- 任何人都能上传文档、提问、删除知识库
- 仅适用于完全可信的环境（本机调试）

### 6.5 重要限制

⚠️ DEMO_PASSWORD **不是生产级鉴权**：

- 密码以明文形式存在 `.env` 中
- Streamlit 无登录会话过期机制（关闭浏览器即失效）
- 仅防止"陌生人随手点开链接"，不防恶意攻击
- 真正的生产方案需要反向代理层 basic-auth 或 OAuth

---

## 7. 安装 cloudflared

`cloudflared` 是 Cloudflare 官方的独立客户端，**不是 Python pip 包，不要写进 `requirements.txt`**。

### 7.1 各平台安装

| 平台 | 命令 |
| --- | --- |
| **macOS** | `brew install cloudflared` |
| **Windows** | `winget install --id Cloudflare.cloudflared` 或从 [官方 GitHub Releases](https://github.com/cloudflare/cloudflared/releases) 下载 `.exe` |
| **Linux (Debian/Ubuntu)** | 参考 [Cloudflare 官方文档](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/install/) 配置 apt 仓库后 `apt install cloudflared` |
| **Linux (其他)** | 从 [GitHub Releases](https://github.com/cloudflare/cloudflared/releases) 下载对应架构的二进制 |

### 7.2 验证安装

```bash
cloudflared --version
```

### 7.3 不需要注册 Cloudflare 账号

Quick Tunnel（`cloudflared tunnel --url`）模式**不需要**登录 Cloudflare 账号，开箱即用。

---

## 8. 常见问题排查

### 8.1 启动阶段

| 症状 | 原因与排查 |
| --- | --- |
| `docker compose up` 报端口占用 | 8501 或 8000 被占用；`netstat -an \| findstr :8501`（Windows）或 `lsof -i :8501`（macOS/Linux）查占用进程 |
| 容器一直不 healthy | `docker compose logs backend` 看是否密钥错误或依赖加载失败 |
| 容器启动后立刻退出 | 多半是 `.env` 缺失或格式错误；确认已执行 `cp .env.example .env` 并填好密钥 |
| 构建镜像失败 | 网络问题导致 `pip install` 超时；可配置国内 pip 镜像后重试 |

### 8.2 访问阶段

| 症状 | 原因与排查 |
| --- | --- |
| `http://127.0.0.1:8501` 打不开 | `docker compose ps` 看 frontend 是否 healthy；`docker compose logs frontend` 看启动错误 |
| `http://127.0.0.1:8000/docs` 打不开 | `docker compose ps` 看 backend 是否 healthy；该端口仅本机可访问 |
| 局域网其他机器无法访问 `http://<你的IP>:8501` | 检查防火墙是否放行 8501 端口；8501 端口本身已绑 `0.0.0.0`，不需要改 compose |
| 局域网其他机器**应该**无法访问 `http://<你的IP>:8000` | ✅ **这是预期行为**——后端仅本机回环，外部访问应该走前端 |

### 8.3 功能阶段

| 症状 | 原因与排查 |
| --- | --- |
| 页面能打开但问答失败 | `docker compose logs backend`；多半是 `.env` 中 LLM / Embedding 密钥未配置或填错 |
| 上传文件后入库卡住 | 大 PDF 入库超时设为 300 秒；Cloudflare Tunnel 长请求易被中断，**建议只上传 < 5MB 的小文件演示** |
| 提问返回拒答 | 检查知识库是否已入库文档：`http://127.0.0.1:8000/api/documents` |
| Rerank 报错 | `USE_RERANK=true` 时检查 SiliconFlow 配额；可临时设为 `false` 跳过 |

### 8.4 Cloudflare Tunnel 阶段

| 症状 | 原因与排查 |
| --- | --- |
| `cloudflared: command not found` | cloudflared 未安装；见 §7 |
| Tunnel 链接打开后白屏 | 等 5–10 秒等 WebSocket 握手完成 |
| 操作页面一段时间后无响应 | Cloudflare 边缘会关闭空闲 ws 连接（约 100 秒无活动）；**刷新页面**即可恢复 |
| Tunnel 链接打不开（502） | `docker compose ps` 确认 frontend 容器还在运行；本机 `curl http://127.0.0.1:8501` 验证 |
| 大文件上传/入库超时 | Quick Tunnel 对长请求不友好；改用小文件，或入库时暂时不通过 tunnel 操作 |

### 8.5 日志查看

```bash
docker compose logs frontend              # 仅前端
docker compose logs backend               # 仅后端
docker compose logs -f                    # 实时跟踪全部
docker compose logs -f --tail=100 backend # 跟踪 backend 最后 100 行
```

---

## 9. 关闭与清理

### 9.1 停止服务（保留数据）

```bash
# 1. 在 cloudflared 终端按 Ctrl + C 停止 tunnel
# 2. 停止 docker 服务（保留向量库卷）
docker compose down
```

下次 `docker compose up -d` 可恢复全部入库的文档。

### 9.2 完全清理（删除数据卷）

⚠️ **谨慎操作**：以下命令会**永久删除** Chroma 向量库和已上传的原始文档。

```bash
docker compose down -v
```

`-v` 标志会删除命名卷 `rs-rag-data`。需要重新入库所有文档才能恢复服务。

---

## 10. 和长期生产方案的区别

Cloudflare Quick Tunnel 是**临时演示方案**，与长期生产部署有以下区别：

| 维度 | Cloudflare Quick Tunnel（本文档方案） | 长期生产方案 |
| --- | --- | --- |
| 域名 | 临时 `xxxx.trycloudflare.com`，每次重启变化 | 固定自有域名（如 `rs-rag.example.com`） |
| TLS 证书 | Cloudflare 自动 | Caddy 自动 或 Cloudflare Named Tunnel |
| 鉴权 | `DEMO_PASSWORD` 简单口令 | 反向代理层 basic-auth / OAuth / SSO |
| 后端暴露 | 仅本机回环 | 反向代理统一入口 `/api/*` |
| 持久化 | 本机 docker volume | 云服务器持久化卷 + 备份 |
| 配置方式 | `cloudflared tunnel --url` | `cloudflared tunnel run <name>` + `~/.cloudflared/config.yml` 或 `Caddyfile` |
| 适用场景 | 临时演示、内部分享、PoC | 长期生产、对外服务 |

### 何时从演示方案迁移到生产方案

- 链接需要长期稳定（不每次变）→ 用 Named Tunnel 或自有域名
- 演示对象扩大到不可信用户 → 必须加正式鉴权
- 演示数据需要备份或多机共享 → 迁移到云服务器
- 需要监控、告警、日志聚合 → 引入完整运维栈

迁移路径参考：

1. **同一台机器升级**：保留 docker compose，加 Caddy 反向代理 + 自有域名
2. **迁移到云服务器**：拷贝镜像 + 数据卷到云主机，再走方案 1
3. **多副本部署**：拆分 Chroma 到独立服务（或改用 Qdrant/Milvus），后端水平扩展
