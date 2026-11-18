# contest_harness

选手侧 SDK，版本 1.0.4，使用 `contest.harness.workspace.v2` 协议并要求支持分块上传的新宿主。SDK 推荐在镜像中提供以下固定入口：

```sh
python3 /app/harness.py describe
python3 /app/harness.py train
python3 /app/harness.py predict
```

在 SDK 目录安装：

```sh
pip install .
```

把 `template/harness.py` 复制到镜像的 `/app/harness.py`，并在 `/app/contest_submission.py` 中提供 `train(rnd)` 和 `predict(sample, model_dir)`。使用其他模块名时，设置 `CONTEST_SUBMISSION`。完整提交要求见下载包中的 `HARNESS_SDK.md`（仓库内位于 `docs/HARNESS_SDK.md`）。固定入口是 SDK 推荐接法，不表示宿主完全禁用其他 adapter。

`describe` 会导入选手模块，检查训练和预测函数及任务声明；导入失败会终止，不会进入训练。它不是完整的训练代码审查，通过后仍需保证训练流程正确。

训练时通过 `rnd.model()` 或 `rnd.transport()` 调用平台模型。官方提示由宿主附加到 system 消息，不要在选手消息中重复粘贴。传输层兼容工具调用、多个 choices 和可选的流式 usage；流式响应是在一次非流式调用完成后转换的事件流，不是实时 token 流。

预测时，`model_dir` 是 `/model` 根目录，`CONTEST_ARTIFACT_PATH` 指向选中的检查点文件。状态目录 `/state` 只读，先用 `rnd.state_workspace()` 复制到可写目录，再用 `rnd.save_state()` 保存。同轮多次保存都绑定本轮初始状态，不以前一次保存为父状态。

RPC 发出后若被异常、信号或超时中断，其宿主处理结果未知；本轮会话不可再使用，后续调用报 `RPC_OUTCOME_UNKNOWN`，不会消费迟到的响应或重试。不要捕获超时后继续同一轮；候选验证可能超过选手自行设置的时限。

`example/contest_submission.py` 只展示协议和模型调用，不是可得分的训练方案。
