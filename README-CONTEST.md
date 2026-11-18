# 智训引擎 | 赛用 Harness SDK 1.0.4

赛用入口为 `/app/harness.py describe|train|predict`。模型访问经 `host_gateway_rpc`，协议为 `contest.harness.workspace.v2`。本版在与当前沙箱 CPU 和内存上限相匹配的 Docker 容器中，以本地宿主驱动真实 PyTRiO `Qwen/Qwen3.8-27B`，各完成 A、B、C 10 轮。台账分开记录本地宿主回执、PyTRiO 上游请求、平台待验证项。

## 构建和检查

在 Linux x86-64 或 WSL2 Ubuntu 中运行。验收使用本机的 `zhixun-python:ubuntu24` 基础镜像。若需要重新构建基础环境，先执行第一条。运行依赖锁定在 `requirements-contest.lock`，源码 ZIP 含构建轮子与成员 SHA-256。

```bash
docker build --platform linux/amd64 -f Dockerfile.contest-base -t zhixun-python:ubuntu24 .
docker build --network none --platform linux/amd64 \
  --build-arg PYTHON_IMAGE=zhixun-python:ubuntu24 -f Dockerfile.contest \
  -t zhixun-harness:contest-1.0.4 .
docker run --rm --network none --read-only --cpus 2 --memory 4g zhixun-harness:contest-1.0.4 describe
docker save -o harness.tar zhixun-harness:contest-1.0.4
docker load -i harness.tar
(cd vendor/contest-harness-sdk-1.0.4 && \
   python3 -m unittest tests.test_sdk tests.test_contract_regressions tests.test_transport_regressions -v)
```

## 本地闭环验收及恢复

将官方任务包中的 `checkpoint.npz`、`public/train.jsonl` 和 `public/validation.jsonl` 置于 `vendor/reference/{A,B,C}/` 的对应路径。B 使用 `cpu-b-rul-20261110-v1`。源码 ZIP 不转发任务数据或检查点。镜像基于平台挂载的官方父检查点启动，不从历史发布模型启动。

```bash
conda activate zhixun-harness
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# 一轮不计费的协议测试，包含无效回复、候选上传和独立预测。
python -m scripts.check_contest_104 --mode fixture --run-id contest-104-local-fixture --tasks A --rounds 1
# 在已获授权的费用台账所在项目中，通过运行环境注入 PYTRIO_API_KEY 后执行。
python -m scripts.check_contest_104 --mode live --run-id contest-104-release --rounds 10
```

本地宿主可用 `requirements-contest.lock` 和 `requirements-v12-trainer.lock` 在 Python 3.12 环境中安装依赖。真实跑法依赖已有 `runs/v12-trainer/cost-ledger.json` 的累计授权账本，从 `PYTRIO_API_KEY` 读取密钥。恢复同一运行时复用已经保存并匹配镜像及父链的完成回执。结果不明的上游请求保留原费用上界。定位到瞬时故障并核对原记录后，可在新宿主轮次设置 `CONTEST_104_RETRY_REQUEST` 为该单个失败请求标识，发起单独计量的一次重试。

本次本地验收前 26 个成功轮次使用 OpenAI 兼容接口，C 第 7 至 10 轮使用锁定的 PyTRiO 0.2.9 采样接口。复现 SDK 采样时设置 `CONTEST_104_PYTRIO_TRANSPORT=sdk`，并设置 `CONTEST_104_TOKENIZER` 指向已获取的 Qwen3.8-27B tokenizer 目录。源包不包含在线凭证或累计费用账本。平台运行直接通过 SDK 的宿主网关，无需本地验收桥接、PyTRiO SDK 或 tokenizer。

## 实现约束

- 容器仅读取平台挂载的训练数据、父模型和历史。工业拟合与统计量估计仅读取 `/data/train.jsonl` 的授权标签。公开验证评分由宿主执行，逐样本预测只用 `inputs`。
- 加载原生 `checkpoint.npz` 及适用的 `extension.npz`。扩展绑定原生 SHA-256，在预测结果中生效。新增支路保持原生权重及 Adam 状态原样，原生 Adam 续训保留标准化和目标缩放并累计步数。平台公告的 64 MiB 工业产物上限优先于 SDK 所允许的较大的运输配额。
- 训练尝试至多一个候选，以相同投影取得父子测量。Qwen 根据摘要给出选择和下一动作。分数改善和合法回复共同决定接纳。无可比较的指标、格式错误和动作耗尽时保留父模型。
- SDK 上传采用 `chunked-v1`。真实测量或状态 RPC 中断时按 SDK 1.0.4 终止本轮会话。候选与状态只写 `/workspace`。日志走 stderr，预测在 EOF 前逐行回复。
- 本地闭环使用实际 PyTRiO 结果和完整公开验证样本。平台正式宿主、阶段变化、隐藏数据和正式得分需要平台验证。

## 证据和软件来源

`reports/contest-submission-1.0.4/contest-104-release.json` 是同一镜像实测记录，含父子测量、状态摘要和真实 Qwen 返回。`acceptance-verification.json` 中有累计费用、资源和核验范围，`swanlab-verification.json` 记录云端复核。SDK 归档随赛方下发，未改动其代码和模板。参考 NumPy 模型及特征解析来自官方 A CPU 任务包，A/B/C 使用相同实现。工业扩展与调度为本项目实现。PyTRiO 凭证和 SwanLab 凭证由运行环境提供。
